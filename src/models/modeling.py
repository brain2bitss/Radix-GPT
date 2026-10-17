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