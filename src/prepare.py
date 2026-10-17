"""Stream several datasets, mix them by token ratio, tokenize with the GPT-2 tokenizer and
write uint16 shards. Memory stays small (one shard at a time) and the job is resumable.

Run on a CPU-only Kaggle session (it does not use GPU quota):
    python -m src.prepare --target_tokens 3e9 --hf_repo yourname/my-tokens

With --hf_repo, every shard (and state.json) is uploaded to a private HF dataset repo as it
is finished, so a dead session loses at most one shard and can be restarted with the same command.
"""
import argparse
import collections
import json
import os
import time

import numpy as np

EOT = 50256  # GPT-2 <|endoftext|>, appended after every document

# ratio = share of total tokens. Dataset ids / config names / text fields: verify they still
# exist on the Hub if a source fails to load.
SOURCES = {
    "fineweb_edu": dict(ratio=0.55, path="HuggingFaceFW/fineweb-edu", name="sample-10BT", field="text"),
    "cosmopedia":  dict(ratio=0.15, path="HuggingFaceTB/smollm-corpus", name="cosmopedia-v2", field="text"),
    "code":        dict(ratio=0.10, path="codeparrot/codeparrot-clean", name=None, field="content"),
    "math":        dict(ratio=0.08, path="HuggingFaceTB/finemath", name="finemath-4plus", field="text"),
    "wikipedia":   dict(ratio=0.07, path="wikimedia/wikipedia", name="20231101.en", field="text"),
    "fineweb":     dict(ratio=0.05, path="HuggingFaceFW/fineweb", name="sample-10BT", field="text"),
}


class Source:
    def __init__(self, name, spec, tok, skip_docs=0, batch_docs=256):
        from datasets import load_dataset
        kwargs = dict(split="train", streaming=True)
        ds = (load_dataset(spec["path"], spec["name"], **kwargs) if spec["name"]
              else load_dataset(spec["path"], **kwargs))
        if skip_docs:
            print(f"[{name}] skipping {skip_docs:,} docs already consumed in earlier runs...")
            ds = ds.skip(skip_docs)
        self.name, self.field, self.tok = name, spec["field"], tok
        self.it = iter(ds)
        self.batch_docs = batch_docs
        self.buf = collections.deque()
        self.docs = skip_docs  # docs popped so far (including earlier runs)

    def _refill(self):
        texts = []
        for _ in range(self.batch_docs):
            try:
                row = next(self.it)
            except StopIteration:
                break
            texts.append(row[self.field] or "")
        if not texts:
            return
        enc = self.tok(texts, add_special_tokens=False, return_attention_mask=False)["input_ids"]
        for ids in enc:
            # None = too short to keep (still counts as a consumed doc for resume bookkeeping)
            self.buf.append(None if len(ids) < 32 else np.array(ids + [EOT], dtype=np.uint16))

    def next(self):
        if not self.buf:
            self._refill()
            if not self.buf:
                raise EOFError(self.name)
        self.docs += 1
        return self.buf.popleft()


def save_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--target_tokens", type=float, default=3e9, help="training tokens (excludes val)")
    ap.add_argument("--val_tokens", type=float, default=10e6)
    ap.add_argument("--shard_tokens", type=float, default=100e6)
    ap.add_argument("--hf_repo", default="", help="private HF *dataset* repo for shards + state")
    args = ap.parse_args()
    target, val_target, shard_target = int(args.target_tokens), int(args.val_tokens), int(args.shard_tokens)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.model_max_length = 10 ** 9  # silence the 1024-token warning; we pack, not truncate

    shard_dir = os.path.join(args.out_dir, "shards")
    os.makedirs(shard_dir, exist_ok=True)
    state_path = os.path.join(args.out_dir, "state.json")

    api = None
    if args.hf_repo:
        from huggingface_hub import HfApi, hf_hub_download
        from src.utils import setup_hf_token
        setup_hf_token()
        api = HfApi()
        api.create_repo(args.hf_repo, repo_type="dataset", private=True, exist_ok=True)
        if not os.path.exists(state_path):
            try:
                hf_hub_download(args.hf_repo, "state.json", repo_type="dataset", local_dir=args.out_dir)
                print("Restored state.json from the Hub; resuming.")
            except Exception:
                pass

    if os.path.exists(state_path):
        with open(state_path) as f:
            state = json.load(f)
    else:
        state = {"shard_idx": 0, "val_done": False, "train_tokens": 0, "docs": {}, "emitted": {}}

    sources = {n: Source(n, spec, tok, state["docs"].get(n, 0)) for n, spec in SOURCES.items()}
    emitted = {n: state["emitted"].get(n, 0) for n in SOURCES}

    def upload(path):
        if api is None:
            return
        rel = os.path.relpath(path, args.out_dir).replace(os.sep, "/")
        api.upload_file(path_or_fileobj=path, path_in_repo=rel, repo_id=args.hf_repo, repo_type="dataset")

    def commit_state():
        state["docs"] = {**state["docs"], **{n: s.docs for n, s in sources.items()}}
        state["emitted"] = dict(emitted)
        save_json(state_path, state)
        upload(state_path)

    def build_chunk(limit):
        """Collect docs until >= limit tokens, always taking from the source furthest below
        its target ratio, so the mix stays exact."""
        parts, n, next_report = [], 0, 25_000_000
        while n < limit and sources:
            name = min(sources, key=lambda k: emitted[k] / SOURCES[k]["ratio"])
            try:
                arr = sources[name].next()
            except EOFError:
                print(f"[{name}] exhausted; its share will be redistributed.")
                del sources[name]
                continue
            if arr is None:
                continue
            parts.append(arr)
            n += len(arr)
            emitted[name] += len(arr)
            if n >= next_report:
                print(f"   ...{n / 1e6:.0f}M / {limit / 1e6:.0f}M tokens in this chunk")
                next_report += 25_000_000
        return np.concatenate(parts) if parts else None

    # ---- validation set (held out: taken from the stream first, never reused in training) ----
    if not state["val_done"]:
        print(f"Building validation set (~{val_target / 1e6:.0f}M tokens)...")
        chunk = build_chunk(val_target)
        path = os.path.join(args.out_dir, "val.bin")
        chunk.tofile(path)
        upload(path)
        state["val_done"] = True
        commit_state()

    # ---- training shards ----
    while state["train_tokens"] < target:
        t0 = time.time()
        chunk = build_chunk(shard_target)
        if chunk is None:
            print("All sources exhausted.")
            break
        path = os.path.join(shard_dir, f"shard_{state['shard_idx']:05d}.bin")
        chunk.tofile(path)
        upload(path)                       # upload the shard BEFORE state, so state never points at a missing file
        state["shard_idx"] += 1
        state["train_tokens"] += len(chunk)
        commit_state()
        mix = {n: f"{emitted[n] / max(1, sum(emitted.values())):.0%}" for n in SOURCES}
        print(f"wrote {path} ({len(chunk) / 1e6:.0f}M tokens, {time.time() - t0:.0f}s) | "
              f"total {state['train_tokens'] / 1e9:.2f}B / {target / 1e9:.2f}B | mix {mix}")

    print("Done.")


if __name__ == "__main__":
    main()
