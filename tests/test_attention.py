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
