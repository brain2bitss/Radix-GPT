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