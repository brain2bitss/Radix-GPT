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
