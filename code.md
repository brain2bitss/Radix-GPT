## README.md

```markdown
```

## configs/355m_config.yaml

```yaml
# Architecture Hyperparameters (Calculates to ~355M parameters)
vocab_size: 50257        # Standard GPT-2 vocabulary size
hidden_size: 1024        # Hidden dimension size
num_hidden_layers: 24    # Total transformer blocks
num_attention_heads: 16  # Multi-head attention count
intermediate_size: 2816  # SwiGLU hidden dimension (~8/3 * hidden_size)
max_position_embeddings: 2048 # Context window length
initializer_range: 0.02  # Weights initialization standard deviation
rms_norm_eps: 1e-5       # Stability epsilon for RMSNorm

# Training Hyperparameters
batch_size: 4            # Micro-batch size per GPU
gradient_accumulation_steps: 8 # Simulates larger batch sizes
learning_rate: 3e-4      # Peak learning rate
weight_decay: 0.1        # AdamW regularization
adam_beta1: 0.9
adam_beta2: 0.95
adam_epsilon: 1e-8
max_grad_norm: 1.0       # Gradient clipping threshold
epochs: 3
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
    def __init__(self, config_yaml: str, weights_path: str = None):
        self.config = LLMConfig.from_yaml(config_yaml)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = LLMTokenizer()
        
        self.model = CustomLLM(self.config)
        if weights_path:
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def generate(self, prompt: str, max_new_tokens: int = 50, temperature: float = 0.7) -> str:
        input_ids = torch.tensor([self.tokenizer.encode(prompt)], dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Slice input if context length overflows model capabilities
                input_cond = input_ids[:, -self.config.max_position_embeddings:]
                logits, _ = self.model(input_cond)
                
                logits = logits[:, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)
                
                next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat((input_ids, next_token), dim=1)
                
        return self.tokenizer.decode(input_ids[0].tolist())

if __name__ == "__main__":
    # Self-test invocation using local randomly initialized weights
    engine = LLMInferenceEngine("configs/355m_config.yaml")
    output = engine.generate("Artificial Intelligence is", max_new_tokens=20)
    print(f"\nGenerated Output:\n{output}")
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
from dataclasses import dataclass, field

@dataclass
class LLMConfig:
    vocab_size: int = 50257
    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    intermediate_size: int = 2816
    max_position_embeddings: int = 2048
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    epochs: int = 3
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
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    
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
        
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.token_embeddings.weight = self.lm_head.weight
        
        self.apply(self._init_weights)
        print(f"Model Parameter Count: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")

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

def prepare_dataset(output_path="data/tokenized/train.bin", max_samples=50000):
    print("Downloading OpenAssistant dataset...")
    dataset = load_dataset("OpenAssistant/oasst1", split="train")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    filtered = dataset.filter(lambda x: x["lang"] == "en")
    subset = filtered.shuffle(seed=42).select(range(min(max_samples, len(filtered))))
    
    text_data = []
    for row in subset:
        role = "User" if row['role'] == 'prompter' else "Assistant"
        text_data.append(f"<|{role}|>\n{row['text']}\n<|endoftext|>\n")
        
    full_text = "".join(text_data)
    tokens = tokenizer.encode(full_text)
    arr = np.array(tokens, dtype=np.uint16)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    arr.tofile(output_path)
    print(f"Successfully created {output_path} with {len(tokens):,} tokens.")

if __name__ == "__main__":
    prepare_dataset()
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
            shuffle=True, 
            pin_memory=True if torch.cuda.is_available() else False,
            num_workers=2 if torch.cuda.is_available() else 0
        )

    def train(self):
        self.model.train()
        os.makedirs(self.config.output_dir, exist_ok=True)
        step = 0
        
        # Fixed the typo here
        autocast_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
        print(f"Starting training loop execution using precision: {autocast_dtype}")
        
        for epoch in range(self.config.epochs):
            self.optimizer.zero_grad(set_to_none=True)
            
            for x, y in self.dataloader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                
                with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=autocast_dtype):
                    logits, loss = self.model(x, y)
                    loss = loss / self.config.gradient_accumulation_steps
                
                loss.backward()
                
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    
                    if step % self.config.logging_steps == 0:
                        actual_loss = loss.item() * self.config.gradient_accumulation_steps
                        print(f"Epoch {epoch} | Step {step} | Loss: {actual_loss:.4f}")
                        
                if step % self.config.checkpoint_steps == 0 and step > 0:
                    checkpoint_path = os.path.join(self.config.output_dir, f"step_{step}.pt")
                    torch.save(self.model.state_dict(), checkpoint_path)
                    print(f"Saved sharded architectural weights to: {checkpoint_path}")
                    
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

