## README.md

`markdown

`

## benchmark.py

`python
import gc
import time
import torch

from src.config import LLMConfig
from src.models.modeling import CustomLLM

SESSION_SECONDS = 12 * 3600


def make_config(size, seq_len):
    if size == 355:
        return LLMConfig(hidden_size=1024, num_hidden_layers=24, num_attention_heads=16,
                         intermediate_size=2816, max_position_embeddings=seq_len)
    if size == 125:
        return LLMConfig(hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
                         intermediate_size=2048, max_position_embeddings=seq_len)
    raise ValueError(size)


def bench(size, micro_bs, seq_len=512, accum=8, checkpointing=True):
    """Time `accum` micro-batches plus ONE optimizer step (one real training step)."""
    tag = f"{size}M | micro_bs={micro_bs} | seq={seq_len} | ckpt={checkpointing}"
    model = opt = x = y = None
    try:
        config = make_config(size, seq_len)
        model = CustomLLM(config).cuda()
        if checkpointing:
            model.gradient_checkpointing_enable()  # make sure your class has this
        model.train()

        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)
        scaler = torch.amp.GradScaler("cuda")

        x = torch.randint(0, config.vocab_size, (micro_bs, seq_len), device="cuda")
        y = torch.randint(0, config.vocab_size, (micro_bs, seq_len), device="cuda")

        def micro():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _, loss = model(x, y)
            scaler.scale(loss / accum).backward()

        def train_step():
            for _ in range(accum):
                micro()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

        for _ in range(2):  # warmup (also allocates optimizer state)
            train_step()

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        n_steps = 3
        for _ in range(n_steps):
            train_step()
        torch.cuda.synchronize()
        dt = time.time() - t0

        tokens = n_steps * accum * micro_bs * seq_len  # no extra multipliers
        tps = tokens / dt
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"[{tag}] params={n_params:.0f}M  {tps:,.0f} tok/s  "
              f"peak VRAM={peak:.1f} GB  ~{tps * SESSION_SECONDS / 1e6:,.0f}M tokens per 12h")
    except torch.cuda.OutOfMemoryError:
        print(f"[{tag}] OOM")
    finally:
        del model, opt, x, y
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    print("GPU:", torch.cuda.get_device_name(0))
    for mbs in (1, 4, 8):
        bench(355, mbs)
    for mbs in (4, 8, 16):
        bench(125, mbs)
    bench(125, 8, checkpointing=False)  # checkpointing costs ~25-30% speed; see if it fits without

`

## configs/125m_config.yaml

`yaml
# ~125M params (with tied embeddings). All "steps" below are OPTIMIZER steps.

# Architecture
vocab_size: 50304
hidden_size: 768
num_hidden_layers: 12
num_attention_heads: 12
intermediate_size: 2048
max_position_embeddings: 512
initializer_range: 0.02
rms_norm_eps: 0.00001
tie_word_embeddings: true

# Optimization
batch_size: 16
gradient_accumulation_steps: 8     # 16 x 8 = 128 seqs x 512 = 65,536 tokens per optimizer step
learning_rate: 0.0005
min_lr: 0.00005
warmup_steps: 1000
max_steps: 61000                   # ~4B tokens. SET FROM YOUR BENCHMARKED tok/s, do not change mid-run
weight_decay: 0.1
adam_beta1: 0.9
adam_beta2: 0.95
adam_epsilon: 0.00000001
max_grad_norm: 1.0
dtype: "float16"
gradient_checkpointing: true       # try false if benchmark.py shows it fits (about 25% faster)
seed: 1337

# Logging / eval / checkpoints
logging_steps: 10
eval_steps: 500
eval_batches: 100
checkpoint_minutes: 30
keep_checkpoints: 2
time_limit_hours: 11.5

# Paths
data_dir: "data/shards"
val_path: "data/val.bin"
output_dir: "checkpoints"

# Optional Hugging Face Hub persistence (leave "" to disable)
hf_repo: ""
hf_data_repo: ""

`

## configs/355m_config.yaml

`yaml
# ~355M params (with tied embeddings). All "steps" below are OPTIMIZER steps.

# Architecture
vocab_size: 50304            # GPT-2 vocab (50257) padded to a multiple of 64
hidden_size: 1024
num_hidden_layers: 24
num_attention_heads: 16
intermediate_size: 2816
max_position_embeddings: 512
initializer_range: 0.02
rms_norm_eps: 0.00001
tie_word_embeddings: true

# Optimization
batch_size: 4                      # micro-batch; raise if benchmark.py shows VRAM headroom
gradient_accumulation_steps: 32    # 4 x 32 = 128 seqs x 512 = 65,536 tokens per optimizer step
learning_rate: 0.0003
min_lr: 0.00003
warmup_steps: 1000
max_steps: 23000                   # ~1.5B tokens. SET FROM YOUR BENCHMARKED tok/s, do not change mid-run
weight_decay: 0.1
adam_beta1: 0.9
adam_beta2: 0.95
adam_epsilon: 0.00000001
max_grad_norm: 1.0
dtype: "float16"                   # T4 has no bf16
gradient_checkpointing: true
seed: 1337

# Logging / eval / checkpoints
logging_steps: 5
eval_steps: 250
eval_batches: 100
checkpoint_minutes: 30
keep_checkpoints: 2
time_limit_hours: 11.5             # save + exit cleanly before Kaggle's 12h cutoff

# Paths
data_dir: "data/shards"
val_path: "data/val.bin"
output_dir: "checkpoints"

# Optional Hugging Face Hub persistence (leave "" to disable)
hf_repo: ""                        # e.g. "yourname/my-355m-run"  (private model repo, checkpoints)
hf_data_repo: ""                   # e.g. "yourname/my-tokens"    (private dataset repo, shards)

`

## inference.py

`python
import torch
import torch.nn.functional as F
from src.config import LLMConfig
from src.models.modeling import CustomLLM
from src.tokenizer import LLMTokenizer

class LLMInferenceEngine:
    def __init__(self, config_yaml: str, weights_path: str):
        self.config = LLMConfig.from_yaml(config_yaml)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = LLMTokenizer()
        
        self.model = CustomLLM(self.config)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def generate(self, prompt: str, max_new_tokens: int = 300, temperature: float = 0.7) -> str:
        # 1. We MUST use the exact format we trained on
        system_prompt = "You are a helpful AI assistant. When asked to list things, use bullet points. When asked for code, use markdown code blocks. Use paragraphs for explanations."
        formatted_prompt = f"<|System|>\n{system_prompt}\n<|User|>\n{prompt}\n<|Assistant|>\n"
        
        input_ids = torch.tensor([self.tokenizer.encode(formatted_prompt)], dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                input_cond = input_ids[:, -self.config.max_position_embeddings:]
                logits, _ = self.model(input_cond)
                
                # Apply temperature and top-k sampling for better quality
                logits = logits[:, -1, :] / temperature
                v, _ = torch.topk(logits, min(40, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat((input_ids, next_token), dim=1)
                
                # Stop if we hit the end of text token
                if next_token.item() == self.tokenizer.tokenizer.eos_token_id:
                    break
                
        # Decode and extract only the assistant's reply
        full_text = self.tokenizer.decode(input_ids[0].tolist())
        reply = full_text.split("<|Assistant|>\n")[-1].split("<|endoftext|>")[0]
        return reply.strip()

if __name__ == "__main__":
    engine = LLMInferenceEngine("configs/355m_config.yaml", "checkpoints/step_1000.pt")
    output = engine.generate("What is Python? List 3 bullet points why it is popular.", max_new_tokens=200)
    print(f"\n--- Model Output ---\n{output}")
    
`

## requirements.txt

`text
torch>=2.2.0
numpy>=1.24.0
pyyaml>=6.0
transformers>=4.38.0
datasets
huggingface_hub

`

## src/__init__.py

`python

`

## src/config.py

`python
import dataclasses
from dataclasses import dataclass

import yaml


@dataclass
class LLMConfig:
    # ---- architecture ----
    vocab_size: int = 50304
    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    intermediate_size: int = 2816
    max_position_embeddings: int = 512
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True

    # ---- optimization (all steps are OPTIMIZER steps) ----
    batch_size: int = 4
    gradient_accumulation_steps: int = 32
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 1000
    max_steps: int = 23000
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    dtype: str = "float16"
    gradient_checkpointing: bool = True
    seed: int = 1337

    # ---- logging / eval / checkpoints ----
    logging_steps: int = 5
    eval_steps: int = 250
    eval_batches: int = 100
    checkpoint_minutes: int = 30
    keep_checkpoints: int = 2
    time_limit_hours: float = 11.5

    # ---- paths ----
    data_dir: str = "data/shards"
    val_path: str = "data/val.bin"
    output_dir: str = "checkpoints"

    # ---- optional Hugging Face Hub persistence ----
    hf_repo: str = ""
    hf_data_repo: str = ""

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "LLMConfig":
        with open(yaml_path, "r") as f:
            raw = yaml.safe_load(f) or {}

        fields = {f.name: f for f in dataclasses.fields(cls)}
        unknown = set(raw) - set(fields)
        if unknown:
            raise ValueError(f"Unknown config keys in {yaml_path}: {sorted(unknown)}")

        # YAML reads "1e-5" (no dot) as a string, so cast numbers/strings by declared type.
        for key, value in raw.items():
            ftype = fields[key].type
            if ftype in (int, float, str) and value is not None:
                raw[key] = ftype(value)
        return cls(**raw)

`

## src/dataset.py

`python
import numpy as np
import torch


class ShardedTokens:
    """Many uint16 .bin files viewed as one pool of non-overlapping training blocks.

    A block is block_size + 1 tokens (inputs = first block_size, targets = shifted by one);
    consecutive blocks share only that one boundary token.
    """

    def __init__(self, files, block_size):
        if not files:
            raise ValueError("No data files given.")
        self.block_size = block_size
        self.maps = [np.memmap(f, dtype=np.uint16, mode="r") for f in files]
        counts = [(len(m) - 1) // block_size for m in self.maps]
        self.cum = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        self.n_blocks = int(self.cum[-1])
        if self.n_blocks == 0:
            raise ValueError("Data files are too small for this block size.")

    def __len__(self):
        return self.n_blocks

    def get(self, i):
        s = int(np.searchsorted(self.cum, i, side="right") - 1)
        off = int(i - self.cum[s]) * self.block_size
        chunk = np.asarray(self.maps[s][off: off + self.block_size + 1], dtype=np.int64)
        return chunk[:-1], chunk[1:]


class BatchLoader:
    """Deterministic shuffled batches. The only state is `pos` (samples consumed), so a
    resumed run continues exactly where it stopped. Each epoch uses a new permutation
    seeded by (seed + epoch).

    Do NOT add or remove shards between a checkpoint and its resume: the permutation
    depends on the total number of blocks.
    """

    def __init__(self, tokens, batch_size, seed=1337):
        self.tokens = tokens
        self.batch_size = batch_size
        self.seed = seed
        self.pos = 0
        self._epoch = -1
        self._order = None

    @property
    def n_blocks(self):
        return self.tokens.n_blocks

    @property
    def epoch(self):
        return self.pos // self.tokens.n_blocks

    def _set_epoch(self, epoch):
        rng = np.random.default_rng(self.seed + epoch)
        self._order = rng.permutation(self.tokens.n_blocks)
        self._epoch = epoch

    def next_batch(self):
        n = self.tokens.n_blocks
        xs, ys = [], []
        for _ in range(self.batch_size):
            epoch, p = divmod(self.pos, n)
            if epoch != self._epoch:
                self._set_epoch(epoch)
            x, y = self.tokens.get(int(self._order[p]))
            xs.append(x)
            ys.append(y)
            self.pos += 1
        return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))

    def state_dict(self):
        return {"pos": self.pos, "n_blocks": self.tokens.n_blocks, "seed": self.seed}

    def load_state_dict(self, state):
        if state.get("n_blocks") != self.tokens.n_blocks:
            print(f"WARNING: shard set changed (checkpoint had {state.get('n_blocks')} blocks, "
                  f"now {self.tokens.n_blocks}). Data order will not match the original run.")
        self.pos = int(state["pos"])
        self._epoch = -1

`

## src/models/__init__.py

`python
from .modeling import CustomLLM

__all__ = ["CustomLLM"]

`

## src/models/attention.py

`python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

def apply_rope(q, k, cos, sin):
    cos = cos[None, None, :, :].to(dtype=q.dtype)
    sin = sin[None, None, :, :].to(dtype=q.dtype)
    
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)
        
    q_out = (q * cos) + (rotate_half(q) * sin)
    k_out = (k * cos) + (rotate_half(k) * sin)
    return q_out, k_out

class CausalFlashAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        
        self.rope = RotaryEmbedding(self.head_dim, config.max_position_embeddings)

    def forward(self, x):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(T)
        q, k = apply_rope(q, k, cos, sin)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.o_proj(out)
`

## src/models/layers.py

`python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Llama-style)."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight

class SwiGLU(nn.Module):
    """Swish Gated Linear Unit (Llama-style MLP block)."""
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))

`

## src/models/modeling.py

`python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from .layers import RMSNorm, SwiGLU
from .attention import CausalFlashAttention

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalFlashAttention(config)
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size)
        self.ln1 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ln2 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class CustomLLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False # ADDED
        
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if getattr(config, "tie_word_embeddings", True):
            self.token_embeddings.weight = self.lm_head.weight
        self.apply(self._init_weights)
        print(f"Model Parameter Count: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def forward(self, input_ids, targets=None):
        x = self.token_embeddings(input_ids)
        
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                # THIS SAVES THE VRAM
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
                
        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss
`

## src/prepare.py

`python
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

`

## src/tokenizer.py

`python
import os
from transformers import AutoTokenizer

class LLMTokenizer:
    """Wrapper around a fast Hugging Face tokenizer (e.g., gpt2)."""
    def __init__(self, model_name: str = "gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

`

## src/trainer.py

`python
import glob
import math
import os
import re
import threading
import time
from dataclasses import asdict

import torch

_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16}


class LLMTrainer:
    """All step counts (step, warmup_steps, max_steps, eval_steps, ...) are OPTIMIZER steps;
    each one consumes batch_size * gradient_accumulation_steps sequences."""

    def __init__(self, config, model, train_loader, val_loader=None):
        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA GPU is required (enable a GPU accelerator in Kaggle).")
        self.cfg = config
        self.device = torch.device("cuda")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.amp_dtype = _DTYPES[config.dtype]
        self.scaler = torch.amp.GradScaler("cuda", enabled=(config.dtype == "float16"))
        self.step = 0
        self.tokens_per_step = (config.batch_size * config.gradient_accumulation_steps
                                * config.max_position_embeddings)
        self._upload_thread = None
        self._uploading_path = None

        if config.gradient_checkpointing:
            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()
            else:
                print("WARNING: model has no gradient_checkpointing_enable(); running without it.")

        self.optimizer = self._build_optimizer()

    # ------------------------------------------------------------------ setup
    def _build_optimizer(self):
        c = self.cfg
        params = [p for p in self.model.parameters() if p.requires_grad]
        decay = [p for p in params if p.dim() >= 2]      # matmul + embedding weights
        no_decay = [p for p in params if p.dim() < 2]    # norms / biases
        groups = [
            {"params": decay, "weight_decay": c.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=c.learning_rate, betas=(c.adam_beta1, c.adam_beta2),
                                 eps=c.adam_epsilon, fused=True)

    def get_lr(self, step):
        c = self.cfg
        if step < c.warmup_steps:
            return c.learning_rate * (step + 1) / c.warmup_steps
        if step >= c.max_steps:
            return c.min_lr
        progress = (step - c.warmup_steps) / max(1, c.max_steps - c.warmup_steps)
        return c.min_lr + 0.5 * (1.0 + math.cos(math.pi * progress)) * (c.learning_rate - c.min_lr)

    # ------------------------------------------------------------------ eval
    @torch.no_grad()
    def evaluate(self):
        if self.val_loader is None:
            return None
        self.model.eval()
        self.val_loader.pos = 0  # same validation blocks every time
        n = min(self.cfg.eval_batches, max(1, self.val_loader.n_blocks // self.val_loader.batch_size))
        total = 0.0
        for _ in range(n):
            x, y = self.val_loader.next_batch()
            x, y = x.to(self.device), y.to(self.device)
            with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
                _, loss = self.model(x, y)
            total += loss.item()
        self.model.train()
        return total / n

    # ------------------------------------------------------------------ train
    def train(self):
        cfg = self.cfg
        self.model.train()
        os.makedirs(cfg.output_dir, exist_ok=True)

        start = time.time()
        deadline = start + cfg.time_limit_hours * 3600
        last_ckpt = start
        last_log = start
        steps_at_last_log = self.step
        print(f"Training from step {self.step} to {cfg.max_steps} "
              f"({self.tokens_per_step:,} tokens/step, {cfg.time_limit_hours}h time limit)")

        while self.step < cfg.max_steps:
            lr = self.get_lr(self.step)
            for g in self.optimizer.param_groups:
                g["lr"] = lr

            loss_sum = torch.zeros((), device=self.device)
            for _ in range(cfg.gradient_accumulation_steps):
                x, y = self.train_loader.next_batch()
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
                    _, loss = self.model(x, y)
                    loss = loss / cfg.gradient_accumulation_steps
                self.scaler.scale(loss).backward()
                loss_sum += loss.detach().float()

            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            self.step += 1

            now = time.time()
            if self.step == 1 or self.step % cfg.logging_steps == 0:
                n = self.step - steps_at_last_log
                dt = now - last_log
                tps = n * self.tokens_per_step / dt
                eta_h = (cfg.max_steps - self.step) * (dt / n) / 3600
                print(f"step {self.step}/{cfg.max_steps} | loss {loss_sum.item():.4f} | lr {lr:.2e} "
                      f"| gnorm {float(grad_norm):.2f} | {tps:,.0f} tok/s | ETA {eta_h:.1f}h "
                      f"| epoch {self.train_loader.epoch}")
                last_log, steps_at_last_log = now, self.step

            if self.val_loader is not None and self.step % cfg.eval_steps == 0:
                val_loss = self.evaluate()
                print(f"== step {self.step} | val loss {val_loss:.4f} | val ppl {math.exp(val_loss):.1f}")

            if (now - last_ckpt) >= cfg.checkpoint_minutes * 60:
                self.save_checkpoint()
                last_ckpt = time.time()

            if now >= deadline:
                print("Session time limit reached: saving and exiting. Re-run train.py to resume.")
                self.save_checkpoint()
                self.wait_for_upload()
                return

        print("Training complete.")
        self.save_checkpoint()
        self.save_final()
        self.wait_for_upload()

    # ------------------------------------------------------------ checkpoints
    def _local_checkpoints(self):
        files = glob.glob(os.path.join(self.cfg.output_dir, "step_*.pt"))
        def step_of(p):
            return int(re.search(r"step_(\d+)\.pt$", p).group(1))
        return sorted(files, key=step_of)

    def save_checkpoint(self):
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        path = os.path.join(self.cfg.output_dir, f"step_{self.step}.pt")
        tmp = path + ".tmp"
        torch.save({
            "step": self.step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "train_loader": self.train_loader.state_dict(),
            "config": asdict(self.cfg),
        }, tmp)
        os.replace(tmp, path)  # atomic: a crash mid-write never corrupts an existing checkpoint
        print(f"--- saved checkpoint {path} ---")
        # prune only AFTER the new one is safely on disk
        for old in self._local_checkpoints()[:-self.cfg.keep_checkpoints]:
            if old != self._uploading_path:
                try:
                    os.remove(old)
                except OSError:
                    pass
        self._upload_async(path)

    def save_final(self):
        path = os.path.join(self.cfg.output_dir, "final_model_fp16.pt")
        torch.save({k: v.half() for k, v in self.model.state_dict().items()}, path)
        print(f"Saved final fp16 weights to {path}")

    def resume(self):
        """Load the newest local checkpoint, else try the Hub. Returns True if resumed."""
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        for tmp in glob.glob(os.path.join(self.cfg.output_dir, "*.tmp")):
            os.remove(tmp)  # leftovers from an interrupted save

        path, downloaded = None, False
        local = self._local_checkpoints()
        if local:
            path = local[-1]
        elif self.cfg.hf_repo:
            path = self._download_remote()
            downloaded = path is not None
        if path is None:
            print("No checkpoint found. Starting from scratch.")
            return False

        print(f"Resuming from {path}")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self.train_loader.load_state_dict(ckpt["train_loader"])
        self.step = int(ckpt["step"])
        del ckpt
        torch.cuda.empty_cache()
        if downloaded:
            os.remove(path)  # don't keep a duplicate 4+ GB file on disk
        print(f"Resumed at optimizer step {self.step}")
        return True

    # -------------------------------------------------------------- Hub sync
    def _download_remote(self):
        try:
            from huggingface_hub import hf_hub_download
            return hf_hub_download(repo_id=self.cfg.hf_repo, filename="latest.pt", repo_type="model",
                                   local_dir=self.cfg.output_dir)
        except Exception as e:
            print(f"No remote checkpoint available ({type(e).__name__}).")
            return None

    def _upload_async(self, path):
        if not self.cfg.hf_repo:
            return
        if self._upload_thread is not None and self._upload_thread.is_alive():
            print("Previous upload still running; skipping this one.")
            return
        self._uploading_path = path

        def work():
            try:
                from huggingface_hub import HfApi
                api = HfApi()
                api.create_repo(self.cfg.hf_repo, repo_type="model", private=True, exist_ok=True)
                api.upload_file(path_or_fileobj=path, path_in_repo="latest.pt",
                                repo_id=self.cfg.hf_repo, repo_type="model")
                try:  # overwriting a big file keeps old versions in git history; squash to free space
                    api.super_squash_history(repo_id=self.cfg.hf_repo, repo_type="model")
                except Exception:
                    pass
                print(f"--- uploaded {os.path.basename(path)} to hf://{self.cfg.hf_repo} ---")
            except Exception as e:
                print(f"Upload failed: {e}")
            finally:
                self._uploading_path = None

        self._upload_thread = threading.Thread(target=work, daemon=True)
        self._upload_thread.start()

    def wait_for_upload(self):
        if self._upload_thread is not None and self._upload_thread.is_alive():
            print("Waiting for checkpoint upload to finish...")
            self._upload_thread.join()

`

## src/utils.py

`python
import os


def setup_hf_token():
    """Make HF_TOKEN available from the environment or Kaggle Secrets (secret named HF_TOKEN)."""
    if os.environ.get("HF_TOKEN"):
        return True
    try:
        from kaggle_secrets import UserSecretsClient
        os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
        return True
    except Exception as e:
        print(f"Could not load HF_TOKEN ({e}). Hub upload/download will be skipped or fail.")
        return False

`

## tests/test_attention.py

`python
import torch
from src.config import LLMConfig
from src.models.attention import CausalFlashAttention

def test_attention_shapes():
    config = LLMConfig(hidden_size=256, num_attention_heads=4, max_position_embeddings=128)
    attn = CausalFlashAttention(config)
    
    # Mock Batch size 2, Sequence length 64, Hidden size 256
    x = torch.randn(2, 64, 256)
    out = attn(x)
    
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"
    print("Attention tensor dimensions test: PASSED")

if __name__ == "__main__":
    test_attention_shapes()

`

## tests/test_forward.py

`python
import torch
from src.config import LLMConfig
from src.models.modeling import CustomLLM

def test_model_forward():
    config = LLMConfig(hidden_size=128, num_hidden_layers=2, num_attention_heads=2, max_position_embeddings=64)
    model = CustomLLM(config)
    
    input_ids = torch.randint(0, config.vocab_size, (2, 32))
    logits, _ = model(input_ids)
    
    assert logits.shape == (2, 32, config.vocab_size), f"Wrong output shape: {logits.shape}"
    print("Full forward pass layer dimension test: PASSED")

if __name__ == "__main__":
    test_model_forward()

`

## train.py

`python
import argparse
import glob
import os

import torch

from src.config import LLMConfig
from src.dataset import BatchLoader, ShardedTokens
from src.models.modeling import CustomLLM
from src.trainer import LLMTrainer
from src.utils import setup_hf_token


def find_shards(data_dir):
    return sorted(glob.glob(os.path.join(data_dir, "shard_*.bin")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/355m_config.yaml")
    args = ap.parse_args()

    cfg = LLMConfig.from_yaml(args.config)
    torch.manual_seed(cfg.seed)

    if cfg.hf_repo or cfg.hf_data_repo:
        setup_hf_token()

    # ---- data ----
    shard_files = find_shards(cfg.data_dir)
    if not shard_files and cfg.hf_data_repo:
        from huggingface_hub import snapshot_download
        root = os.path.dirname(os.path.normpath(cfg.data_dir)) or "."
        print(f"Downloading shards from hf://{cfg.hf_data_repo} into {root} ...")
        snapshot_download(repo_id=cfg.hf_data_repo, repo_type="dataset", local_dir=root)
        shard_files = find_shards(cfg.data_dir)
    if not shard_files:
        raise SystemExit(f"No shard_*.bin files in {cfg.data_dir}. Run `python -m src.prepare` first.")

    seq_len = cfg.max_position_embeddings
    train_tokens = ShardedTokens(shard_files, seq_len)
    train_loader = BatchLoader(train_tokens, cfg.batch_size, seed=cfg.seed)
    print(f"Train: {len(shard_files)} shards, ~{train_tokens.n_blocks * seq_len / 1e9:.2f}B tokens")

    val_loader = None
    if os.path.exists(cfg.val_path):
        val_tokens = ShardedTokens([cfg.val_path], seq_len)
        val_loader = BatchLoader(val_tokens, cfg.batch_size, seed=0)
        print(f"Val: ~{val_tokens.n_blocks * seq_len / 1e6:.1f}M tokens")
    else:
        print(f"WARNING: {cfg.val_path} not found; validation disabled.")

    planned = cfg.max_steps * cfg.batch_size * cfg.gradient_accumulation_steps * seq_len
    print(f"Plan: {cfg.max_steps} steps = ~{planned / 1e9:.2f}B tokens "
          f"({planned / (train_tokens.n_blocks * seq_len):.2f} epochs)")

    # ---- model ----
    model = CustomLLM(cfg)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {n_params:.0f}M")

    trainer = LLMTrainer(cfg, model, train_loader, val_loader)
    trainer.resume()
    trainer.train()


if __name__ == "__main__":
    main()

`
