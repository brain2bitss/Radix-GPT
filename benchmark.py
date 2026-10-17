import torch
import time
import gc
from src.config import LLMConfig
from src.models.modeling import CustomLLM

def benchmark_model(model_size, micro_batch, seq_len, steps=15):
    print(f"\n--- Testing {model_size}M Model | Micro-batch: {micro_batch} | Seq Len: {seq_len} ---")
    
    # 1. Setup Config
    if model_size == 355:
        config = LLMConfig(hidden_size=1024, num_hidden_layers=24, num_attention_heads=16, intermediate_size=2816, max_position_embeddings=seq_len)
    elif model_size == 125:
        config = LLMConfig(hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=2048, max_position_embeddings=seq_len)
    else:
        raise ValueError("Model size must be 125 or 355")

    try:
        # 2. Setup Model
        model = CustomLLM(config).to("cuda")
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        model.train()
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        scaler = torch.cuda.amp.GradScaler()
        
        # 3. Setup Fake Data
        x = torch.randint(0, config.vocab_size, (micro_batch, seq_len)).cuda()
        y = torch.randint(0, config.vocab_size, (micro_batch, seq_len)).cuda()
        
        # 4. Warmup (Compile CUDA kernels)
        for _ in range(3):
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                _, loss = model(x, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()
        
        # 5. Measurement Loop
        for _ in range(steps):
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                _, loss = model(x, y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        # 6. Calculate Stats
        total_time = end_time - start_time
        # tokens/sec is based ONLY on micro_batch * seq_len * steps
        tokens_processed = steps * micro_batch * seq_len
        tokens_per_sec = tokens_processed / total_time
        
        peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
        
        print(f"Speed: {tokens_per_sec:,.0f} tokens/sec")
        print(f"Peak VRAM: {peak_vram:.2f} GB")
        print(f"Est. tokens per 12h session: {(tokens_per_sec * 43200) / 1_000_000:.1f}M")
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"FAILED: CUDA Out of Memory at micro-batch {micro_batch}.")
        else:
            print(f"FAILED with RuntimeError: {e}")
    finally:
        # Cleanup
        if 'model' in locals(): del model
        if 'optimizer' in locals(): del optimizer
        if 'x' in locals(): del x
        if 'y' in locals(): del y
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    SEQ_LEN = 512 # Standard test sequence length
    
    # Test 125M Model (Faster, smaller)
    benchmark_model(125, micro_batch=4, seq_len=SEQ_LEN)
    benchmark_model(125, micro_batch=16, seq_len=SEQ_LEN)
    
    # Test 355M Model (Bigger, heavier)
    benchmark_model(355, micro_batch=2, seq_len=SEQ_LEN)
    benchmark_model(355, micro_batch=4, seq_len=SEQ_LEN)
    benchmark_model(355, micro_batch=8, seq_len=SEQ_LEN)
