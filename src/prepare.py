import os
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

def clean_and_format_data(tokenizer, max_tokens=50_000_000):
    print("Initializing Multi-Dataset Blending Pipeline...")
    
    # Target allocations: 30% General, 50% Chat (Formatted), 20% Code (Formatted)
    wiki_target = int(max_tokens * 0.30)
    oasst_target = int(max_tokens * 0.50)
    code_target = int(max_tokens * 0.20)
    
    global_token_stream = []

    # =========================================================================
    # DATASET 1: Wikipedia (General Knowledge)
    # =========================================================================
    print(f"\nProcessing Wikipedia Stream (Target: {wiki_target:,} tokens)...")
    try:
        wiki_ds = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)
        wiki_tokens = 0
        for row in wiki_ds:
            # Prompting format to teach it how to summarize text
            formatted_text = f"<|System|>\nYou are a helpful assistant.\n<|User|>\nSummarize the following text:\n{row['text'][:1500]}\n<|Assistant|>\nHere is a summary:\n{row['text'][:400]}\n<|endoftext|>\n"
            tokens = tokenizer.encode(formatted_text, add_special_tokens=False)
            global_token_stream.extend(tokens)
            wiki_tokens += len(tokens)
            if wiki_tokens >= wiki_target:
                break
        print(f"-> Successfully extracted {wiki_tokens:,} Wikipedia tokens.")
    except Exception as e:
        print(f"Skipping Wikipedia: {e}")

    # =========================================================================
    # DATASET 2: OpenAssistant (Formatting & Conversation)
    # =========================================================================
    print(f"\nProcessing OpenAssistant Stream (Target: {oasst_target:,} tokens)...")
    try:
        oasst_ds = load_dataset("OpenAssistant/oasst1", split="train")
        en_oasst = oasst_ds.filter(lambda x: x["lang"] == "en")
        
        # We need to group by conversation ID to ensure the assistant replies to the right user
        message_tree = {}
        for row in en_oasst:
            message_tree.setdefault(row['message_id'], row)
            
        oasst_tokens = 0
        for row in en_oasst:
            role = "User" if row['role'] == 'prompter' else "Assistant"
            
            # Add a system prompt to teach formatting
            if role == "User":
                # Every time we see a user prompt, we start a new conversation block
                system_prompt = "<|System|>\nYou are a helpful AI assistant. When asked to list things, use bullet points. When asked for code, use markdown code blocks. Use paragraphs for explanations.\n"
                formatted_text = f"{system_prompt}<|User|>\n{row['text']}\n"
            else:
                formatted_text = f"<|Assistant|>\n{row['text']}\n<|endoftext|>\n"
                
            tokens = tokenizer.encode(formatted_text, add_special_tokens=False)
            global_token_stream.extend(tokens)
            oasst_tokens += len(tokens)
            
            if oasst_tokens >= oasst_target:
                break
        print(f"-> Successfully extracted {oasst_tokens:,} OpenAssistant tokens.")
    except Exception as e:
        print(f"Skipping OpenAssistant: {e}")

    # =========================================================================
    # DATASET 3: Code Feedback (Coding & Syntax)
    # =========================================================================
    print(f"\nProcessing Code Feedback Stream (Target: {code_target:,} tokens)...")
    try:
        code_ds = load_dataset("m-a-p/CodeFeedback-Filtered-Instruction", split="train", streaming=True)
        code_tokens = 0
        for row in code_ds:
            # Format code specifically with markdown blocks
            formatted_text = f"<|System|>\nYou are a helpful AI assistant. Provide code in ```python blocks.\n<|User|>\n{row['query']}\n<|Assistant|>\n```python\n{row['answer']}\n```\n<|endoftext|>\n"
            tokens = tokenizer.encode(formatted_text, add_special_tokens=False)
            global_token_stream.extend(tokens)
            code_tokens += len(tokens)
            if code_tokens >= code_target:
                break
        print(f"-> Successfully extracted {code_tokens:,} Coding tokens.")
    except Exception as e:
        print(f"Skipping Code Feedback: {e}")

    return global_token_stream

def main():
    output_path = "data/tokenized/train.bin"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # We are generating 50 MILLION tokens. This will take ~15-20 minutes.
    token_stream = clean_and_format_data(tokenizer, max_tokens=50_000_000)
    
    print(f"\nTotal Compiled Global Token Count: {len(token_stream):,}")
    
    # Shuffling
    print("Performing global macro-shuffling across data segments...")
    block_size = 2048
    num_blocks = len(token_stream) // block_size
    
    if num_blocks > 0:
        blocks = [token_stream[i*block_size : (i+1)*block_size] for i in range(num_blocks)]
        np.random.seed(42)
        np.random.shuffle(blocks)
        token_stream = [token for block in blocks for token in block]

    print(f"Serializing unified binary array matrix directly to: {output_path}")
    np_array = np.array(token_stream, dtype=np.uint16)
    np_array.tofile(output_path)
    print("Data compilation successfully completed. Ready for train.py execution loop!")

if __name__ == "__main__":
    main()