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