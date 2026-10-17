## README.md

```markdown
```

## configs/355m_config.yaml

```yaml
# Architecture Hyperparameters (Calculates to ~240M parameters)
vocab_size: 50257        
hidden_size: 896         
num_hidden_layers: 20    
num_attention_heads: 14  
intermediate_size: 2432  
max_position_embeddings: 512  # CHANGED from 1024 to 512 for speed
initializer_range: 0.02  
rms_norm_eps: 0.00001    

# Training Hyperparameters
batch_size: 4                 # CHANGED from 2 to 4
gradient_accumulation_steps: 4 # CHANGED from 16 to 4
learning_rate: 0.0003    
weight_decay: 0.1        
adam_beta1: 0.9
adam_beta2: 0.95
adam_epsilon: 0.00000001 
max_grad_norm: 1.0       
epochs: 2                
logging_steps: 10
checkpoint_steps: 500

# File Paths
data_path: "data/tokenized/train.bin"
output_dir: "checkpoints"
```

## inference.py

```python
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
    
```

## requirements.txt

```text
torch>=2.2.0
numpy>=1.24.0
pyyaml>=6.0
transformers>=4.38.0
```

## src/__init__.py

```python
```

## src/config.py

```python
import yaml
from dataclasses import dataclass

@dataclass
class LLMConfig:
    vocab_size: int = 50257
    hidden_size: int = 896          # Updated
    num_hidden_layers: int = 20     # Updated
    num_attention_heads: int = 14   # Updated
    intermediate_size: int = 2432   # Updated
    max_position_embeddings: int = 1024
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    
    batch_size: int = 2             # Updated
    gradient_accumulation_steps: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    epochs: int = 2
    logging_steps: int = 10
    checkpoint_steps: int = 500
    
    data_path: str = "data/tokenized/train.bin"
    output_dir: str = "checkpoints"

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "LLMConfig":
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
```

## src/dataset.py

```python
import torch
import numpy as np
from torch.utils.data import Dataset

class MemoryMappedDataset(Dataset):
    def __init__(self, data_path: str, block_size: int):
        self.block_size = block_size
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')

    def __len__(self):
        return len(self.data) - self.block_size - 1

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.block_size + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y
```

## src/models/__init__.py

```python
from .modeling import CustomLLM

__all__ = ["CustomLLM"]
```

## src/models/attention.py

```python
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
```

## src/models/layers.py

```python
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
```

## src/models/modeling.py

```python
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

        self.token_embeddings.weight = self.lm_head.weight
        self.apply(self._init_weights)
        print(f"Model Parameter Count: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing = True

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
```

## src/prepare.py

```python
import os
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

def clean_and_format_data(tokenizer, max_tokens=50_000_000):
    print("Initializing Multi-Dataset Blending Pipeline...")
    
    # Target allocations: 30% General, 50% Chat (Formatted), 20% Code (Formatted)
    wiki_target = int(max_tokens * 0.30)
    oasst_target = int(max_tokens * 0.50)
    code_target = int(max_tokens * 0.20)
    
    global_token_stream = []

    # =========================================================================
    # DATASET 1: Wikipedia (General Knowledge)
    # =========================================================================
    print(f"\nProcessing Wikipedia Stream (Target: {wiki_target:,} tokens)...")
    try:
        wiki_ds = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)
        wiki_tokens = 0
        for row in wiki_ds:
            # Prompting format to teach it how to summarize text
            formatted_text = f"<|System|>\nYou are a helpful assistant.\n<|User|>\nSummarize the following text:\n{row['text'][:1500]}\n<|Assistant|>\nHere is a summary:\n{row['text'][:400]}\n<|endoftext|>\n"
            tokens = tokenizer.encode(formatted_text, add_special_tokens=False)
            global_token_stream.extend(tokens)
            wiki_tokens += len(tokens)
            if wiki_tokens >= wiki_target:
                break
        print(f"-> Successfully extracted {wiki_tokens:,} Wikipedia tokens.")
    except Exception as e:
        print(f"Skipping Wikipedia: {e}")

    # =========================================================================
    # DATASET 2: OpenAssistant (Formatting & Conversation)
    # =========================================================================
    print(f"\nProcessing OpenAssistant Stream (Target: {oasst_target:,} tokens)...")
    try:
        oasst_ds = load_dataset("OpenAssistant/oasst1", split="train")
        en_oasst = oasst_ds.filter(lambda x: x["lang"] == "en")
        
        # We need to group by conversation ID to ensure the assistant replies to the right user
        message_tree = {}
        for row in en_oasst:
            message_tree.setdefault(row['message_id'], row)
            
        oasst_tokens = 0
        for row in en_oasst:
            role = "User" if row['role'] == 'prompter' else "Assistant"
            
            # Add a system prompt to teach formatting
            if role == "User":
                # Every time we see a user prompt, we start a new conversation block
                system_prompt = "<|System|>\nYou are a helpful AI assistant. When asked to list things, use bullet points. When asked for code, use markdown code blocks. Use paragraphs for explanations.\n"
                formatted_text = f"{system_prompt}<|User|>\n{row['text']}\n"
            else:
                formatted_text = f"<|Assistant|>\n{row['text']}\n<|endoftext|>\n"
                
            tokens = tokenizer.encode(formatted_text, add_special_tokens=False)
            global_token_stream.extend(tokens)
            oasst_tokens += len(tokens)
            
            if oasst_tokens >= oasst_target:
                break
        print(f"-> Successfully extracted {oasst_tokens:,} OpenAssistant tokens.")
    except Exception as e:
        print(f"Skipping OpenAssistant: {e}")

    # =========================================================================
    # DATASET 3: Code Feedback (Coding & Syntax)
    # =========================================================================
    print(f"\nProcessing Code Feedback Stream (Target: {code_target:,} tokens)...")
    try:
        code_ds = load_dataset("m-a-p/CodeFeedback-Filtered-Instruction", split="train", streaming=True)
        code_tokens = 0
        for row in code_ds:
            # Format code specifically with markdown blocks
            formatted_text = f"<|System|>\nYou are a helpful AI assistant. Provide code in ```python blocks.\n<|User|>\n{row['query']}\n<|Assistant|>\n```python\n{row['answer']}\n```\n<|endoftext|>\n"
            tokens = tokenizer.encode(formatted_text, add_special_tokens=False)
            global_token_stream.extend(tokens)
            code_tokens += len(tokens)
            if code_tokens >= code_target:
                break
        print(f"-> Successfully extracted {code_tokens:,} Coding tokens.")
    except Exception as e:
        print(f"Skipping Code Feedback: {e}")

    return global_token_stream

def main():
    output_path = "data/tokenized/train.bin"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # We are generating 50 MILLION tokens. This will take ~15-20 minutes.
    token_stream = clean_and_format_data(tokenizer, max_tokens=50_000_000)
    
    print(f"\nTotal Compiled Global Token Count: {len(token_stream):,}")
    
    # Shuffling
    print("Performing global macro-shuffling across data segments...")
    block_size = 2048
    num_blocks = len(token_stream) // block_size
    
    if num_blocks > 0:
        blocks = [token_stream[i*block_size : (i+1)*block_size] for i in range(num_blocks)]
        np.random.seed(42)
        np.random.shuffle(blocks)
        token_stream = [token for block in blocks for token in block]

    print(f"Serializing unified binary array matrix directly to: {output_path}")
    np_array = np.array(token_stream, dtype=np.uint16)
    np_array.tofile(output_path)
    print("Data compilation successfully completed. Ready for train.py execution loop!")

if __name__ == "__main__":
    main()
```

## src/tokenizer.py

```python
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
```

## src/trainer.py

```python
import os
import torch
from torch.utils.data import DataLoader

class LLMTrainer:
    def __init__(self, config, model, dataset):
        self.config = config
        self.model = model
        self.dataset = dataset
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # ACTIVATE GRADIENT CHECKPOINTING HERE
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
            print("Gradient Checkpointing ENABLED.")
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_epsilon,
            weight_decay=config.weight_decay
        )
        
        self.dataloader = DataLoader(
            self.dataset, 
            batch_size=config.batch_size, 
            shuffle=True
        )

    def train(self):
        self.model.train()
        os.makedirs(self.config.output_dir, exist_ok=True)
        step = 0
        global_step = 0
        accumulated_loss = 0.0
        
        # Proper precision selection: float16 for Tesla T4, bfloat16 for Ampere+ (A100/H100) or CPU
        if torch.cuda.is_available():
            autocast_device = "cuda"
            if torch.cuda.get_device_capability()[0] >= 8:
                autocast_dtype = torch.bfloat16
            else:
                autocast_dtype = torch.float16
        else:
            autocast_device = "cpu"
            autocast_dtype = torch.bfloat16
            
        use_scaler = (self.device.type == "cuda" and autocast_dtype == torch.float16)
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
        print(f"Starting training loop execution using precision: {autocast_dtype} (GradScaler: {use_scaler})", flush=True)
        
        for epoch in range(self.config.epochs):
            self.optimizer.zero_grad(set_to_none=True)
            
            for x, y in self.dataloader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                
                if step == 0:
                    print(f"Batch loaded. x shape: {x.shape}. Running forward pass...", flush=True)
                
                with torch.amp.autocast(device_type=autocast_device, dtype=autocast_dtype):
                    logits, loss = self.model(x, y)
                    loss = loss / self.config.gradient_accumulation_steps
                
                if step == 0:
                    print("Forward pass done! Running backward pass...", flush=True)
                    
                scaler.scale(loss).backward()
                accumulated_loss += loss.item() * self.config.gradient_accumulation_steps
                
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    
                    global_step += 1
                    
                    # Log on first step, then every logging_steps optimizer updates
                    if global_step == 1 or global_step % self.config.logging_steps == 0:
                        avg_loss = accumulated_loss / self.config.gradient_accumulation_steps
                        print(f"Epoch {epoch} | Step {global_step} (Batch {step + 1}) | Loss: {avg_loss:.4f}", flush=True)
                        
                    accumulated_loss = 0.0
                        
                    if self.config.checkpoint_steps > 0 and global_step % self.config.checkpoint_steps == 0:
                        checkpoint_path = os.path.join(self.config.output_dir, f"step_{global_step}.pt")
                        torch.save(self.model.state_dict(), checkpoint_path)
                        print(f"Saved sharded architectural weights to: {checkpoint_path}", flush=True)
                    
                step += 1
```

## src/utils.py

```python
import torch

def save_checkpoint(model, optimizer, step, path: str):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step
    }
    torch.save(checkpoint, path)

def load_checkpoint(path: str, model, optimizer=None):
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint.get('step', 0)
```

## tests/test_attention.py

```python
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
```

## tests/test_forward.py

```python
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
```

## train.py

```python
import os
import subprocess
from src.config import LLMConfig
from src.models.modeling import CustomLLM
from src.dataset import MemoryMappedDataset
from src.trainer import LLMTrainer

def main():
    config = LLMConfig.from_yaml("configs/355m_config.yaml")
    
    # Check if data exists, if not, run the prepare script
    if not os.path.exists(config.data_path):
        print("Dataset not found. Running prepare.py...")
        subprocess.run(["python", "src/prepare.py"], check=True)
    
    print("Loading Dataset pipeline...")
    dataset = MemoryMappedDataset(config.data_path, config.max_position_embeddings)
    
    print("Initializing 355M Parameter Custom Architecture Foundation Model...")
    model = CustomLLM(config)
    
    print("Initializing Training Engine Execution Context...")
    trainer = LLMTrainer(config, model, dataset)
    
    print("Beginning Pre-training Iteration Loop...")
    trainer.train()

if __name__ == "__main__":
    main()
```

