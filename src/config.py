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
