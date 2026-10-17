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
