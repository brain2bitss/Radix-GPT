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
