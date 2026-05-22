import json
import collections
import re
from pathlib import Path
import sys

def reconstruct():
    print("🔍 Reconstructing Tokenizer Vocabulary...")
    
    # Files to process
    files = [
        Path("/home/agent-engineer/chatbot/massive_training_data.json"),
        Path("/home/agent-engineer/chatbot/jarvis_enhanced_training.json")
    ]
    
    word_counts = collections.Counter()
    
    for file_path in files:
        if not file_path.exists():
            print(f"⚠️ Skipping missing file: {file_path}")
            continue
            
        print(f"📖 Processing {file_path}...")
        with open(file_path, 'r') as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        text = item.get("text", "")
                        words = re.findall(r'\w+', text.lower())
                        word_counts.update(words)
                else:
                    print(f"⚠️ Unexpected format in {file_path}")
            except Exception as e:
                print(f"❌ Error reading {file_path}: {e}")

    # Special tokens
    pad_token = "<pad>"
    eos_token = "<eos>"
    unk_token = "<unk>" # Adding unk for completeness, though SimpleTokenizer didn't have it explicitly in the code I saw, but it used 0 for it.
    
    # In the code SimpleTokenizer uses:
    # pad: 0, eos: 1, next_id starts at 2
    
    word_to_id = {pad_token: 0, eos_token: 1}
    id_to_word = {0: pad_token, 1: eos_token}
    
    # Get top 7998 words (to make total 8000)
    top_words = [word for word, count in word_counts.most_common(7998)]
    
    next_id = 2
    for word in top_words:
        if word not in word_to_id:
            word_to_id[word] = next_id
            id_to_word[next_id] = word
            next_id += 1
            if next_id >= 8000:
                break
                
    vocab_data = {
        "vocab_size": 8000,
        "word_to_id": word_to_id,
        "id_to_word": id_to_word,
        "next_id": next_id
    }
    
    # Save to both locations to be safe
    output_paths = [
        Path("/home/agent-engineer/chatbot/tokenizer.json"),
        Path("/home/agent-engineer/chatbot/jarvis_v1_oracle/tokenizer.json")
    ]
    
    for out_path in output_paths:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(vocab_data, f, indent=2)
        print(f"✅ Tokenizer saved to {out_path} ({len(word_to_id)} tokens)")

if __name__ == "__main__":
    reconstruct()
