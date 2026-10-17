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
            
        # Enterprise-grade type enforcement
        # This guarantees that optimizer variables are always floats
        float_keys = [
            "initializer_range", "rms_norm_eps", "learning_rate", 
            "weight_decay", "adam_beta1", "adam_beta2", 
            "adam_epsilon", "max_grad_norm"
        ]
        int_keys = [
            "vocab_size", "hidden_size", "num_hidden_layers", 
            "num_attention_heads", "intermediate_size", "max_position_embeddings", 
            "batch_size", "gradient_accumulation_steps", "epochs", 
            "logging_steps", "checkpoint_steps"
        ]

        for key in float_keys:
            if key in config_dict:
                config_dict[key] = float(config_dict[key])
                
        for key in int_keys:
            if key in config_dict:
                config_dict[key] = int(config_dict[key])

        return cls(**config_dict)