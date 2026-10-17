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
