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