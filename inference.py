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
    