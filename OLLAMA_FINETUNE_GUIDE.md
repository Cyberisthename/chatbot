# Ollama Fine-Tuning Guide for Ben Lab

## 🎯 Goal

Fine-tune a small language model on your lab's documentation so it can:
- **Explain** concepts (PhaseDetector, TRI, RSI, QPR-R, bit systems)
- **Design** experiments (suggest parameters for maximum TRI, clustering, etc.)
- **Interpret** results (what does TRI=0.08 mean?)
- **Execute** experiments (optional: integrated with your Python PhaseDetector)

## 📂 What's Included

```
lab_corpus/                     # Documentation corpus
├── ARCHITECTURE.md             # Jarvis-5090X 5-layer stack
├── PHASE_DETECTOR.md           # Phase experiments, QPR-R
├── DISCOVERY_SUITE_README.md   # TRI, clustering, RSI
└── BIT_SYSTEM.md               # X/Y/Z/A/S/T/C/P/R bits + G-graph

ollama/                         # Ollama integration
├── README.md                   # Detailed guide
├── Modelfile.example           # Template for Ollama
└── ollama_lab_integration.py   # Python LLM ↔ PhaseDetector bridge

generate_lab_training_data.py   # Generates JSONL training data
lab_training_data.jsonl         # Generated training pairs (249+ examples)
```

## 🚀 Workflow

### Step 1: Generate Training Data ✅

This step is already done! The training data has been generated with 249 Q&A pairs.

```bash
python generate_lab_training_data.py
```

**Output**: `lab_training_data.jsonl`

**Sample questions covered**:
- "What is TRI?" → Detailed explanation
- "Design an experiment to maximize TRI" → Concrete parameter suggestions
- "Explain the four phase types" → Ising, SPT, trivial, pseudorandom
- "What is QPR-R?" → Complexity class explanation
- "Explain X-bit / Y-bit / Z-bit..." → All 9 bit systems

### Step 2: Fine-Tune on Cloud GPU

**Why cloud?** Fine-tuning requires GPU compute (~30-60 min on A100).

**Recommended platforms**:
- **RunPod**: $0.30-0.50/hr for A100
- **Lambda Labs**: Similar pricing
- **Vast.ai**: Spot instances even cheaper
- **Google Colab Pro**: If you have it

**Recommended base model**: Qwen2.5-1.5B (fast, Ollama-friendly)

#### Option A: Unsloth (Easiest)

```python
# On cloud GPU with Unsloth installed
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer

# Load base model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-1.5B-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    use_gradient_checkpointing=True,
)

# Load dataset
dataset = load_dataset("json", data_files="lab_training_data.jsonl")

# Format for training
def format_prompt(example):
    return {
        "text": f"User: {example['instruction']}\nAssistant: {example['output']}"
    }

dataset = dataset.map(format_prompt)

# Train
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    max_seq_length=2048,
    dataset_text_field="text",
    num_train_epochs=3,
    per_device_train_batch_size=4,
)

trainer.train()

# Save
model.save_pretrained("ben-lab-lora")
tokenizer.save_pretrained("ben-lab-lora")

# Merge LoRA + base
FastLanguageModel.save_merged_model(model, tokenizer, "ben-lab-merged")
```

#### Option B: Axolotl (More Control)

```yaml
# config.yaml
base_model: Qwen/Qwen2.5-1.5B
datasets:
  - path: lab_training_data.jsonl
    type: alpaca
    field_instruction: instruction
    field_output: output

adapter: lora
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj

num_epochs: 3
micro_batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 0.0002

# ... more config
```

```bash
accelerate launch -m axolotl.cli.train config.yaml
```

### Step 3: Convert to GGUF

After training, you have either:
- LoRA adapter (`ben-lab-lora/`) + base model
- Merged model (`ben-lab-merged/`)

Convert to GGUF:

```bash
# If merged
python convert_hf_to_gguf.py ben-lab-merged/ --outfile ben-lab-qwen-1.5b.gguf

# Quantize to Q4_K_M (4-bit, good quality/size balance)
./llama.cpp/quantize \
  ben-lab-qwen-1.5b.gguf \
  ben-lab-qwen-1.5b.Q4_K_M.gguf \
  Q4_K_M
```

**Output**: `ben-lab-qwen-1.5b.Q4_K_M.gguf` (~1.5 GB for 1.5B model)

### Step 4: Create Ollama Model

```bash
# Copy GGUF to project
cp ben-lab-qwen-1.5b.Q4_K_M.gguf /home/engine/project/ollama/

# Edit Modelfile.example if needed
# Update the FROM line to match your GGUF filename

cd /home/engine/project/ollama
ollama create ben-lab -f Modelfile.example
```

**Verify**:
```bash
ollama list
# Should show: ben-lab
```

### Step 5: Test the Model

#### Basic Chat

```bash
ollama run ben-lab
```

Try:
- "What is TRI?"
- "Explain the four phase types"
- "Design an experiment to maximize TRI"
- "What is the difference between X-bit and Y-bit?"

#### Integrated with PhaseDetector

```bash
cd /home/engine/project
python ollama/ollama_lab_integration.py
```

**Interactive session**:
```
You: What is TRI?
🤖 BEN-LAB: [Explains TRI]

You: run ising depth 12 bias 0.7
🤖 BEN-LAB: [Suggests experiment design]
🔬 [Executes with PhaseDetector]
✅ [Shows results]
🤖 BEN-LAB: [Interprets results]

You: tri ising
🔬 Measuring TRI for ising_symmetry_breaking...
TRI = 0.078423
```

## 📊 Expected Results

With 249 training pairs on Qwen2.5-1.5B (3 epochs):

### Before Fine-Tuning
- **Q**: "What is TRI?"
- **A**: Generic response about temperature or time

### After Fine-Tuning
- **Q**: "What is TRI?"
- **A**: "TRI (Time-Reversal Instability) measures how sensitive a phase is to bias reversal. It's calculated by running a phase experiment with bias (e.g., 0.7), then running the reverse (1-bias=0.3), and computing L2 distance between feature vectors. Low TRI (0.0001-0.001): time-symmetric. High TRI (0.05+): directional phase."

**Coverage**:
- ✅ All 4 phase types (Ising, SPT, trivial, pseudorandom)
- ✅ All 3 Discovery Suite experiments (TRI, clustering, RSI)
- ✅ All 9 bit systems (X/Y/Z/A/S/T/C/P/R)
- ✅ Jarvis-5090X architecture (5 layers)
- ✅ QPR-R complexity class
- ✅ Feature vector structure
- ✅ Practical how-tos (run experiments, interpret results)

## 🎛️ Configuration Notes

### Modelfile Parameters

```modelfile
PARAMETER temperature 0.35      # Lower = more deterministic
PARAMETER top_p 0.9             # Nucleus sampling
PARAMETER top_k 40              # Token candidates
PARAMETER num_ctx 8192          # Context window
```

**For lab use**:
- `temperature 0.3-0.4`: Precise technical answers
- `temperature 0.6-0.8`: Creative experiment design

### Training Hyperparameters

**Quick (30 min)**:
- Epochs: 3
- Batch size: 4
- Learning rate: 2e-4
- LoRA r=16, α=32

**Higher quality (60 min)**:
- Epochs: 5
- Batch size: 2 (with grad accumulation 4)
- Learning rate: 1e-4
- LoRA r=32, α=64

## 🧪 Auto-Scientist Loop

Once integrated, you can create a closed-loop AI scientist:

```python
assistant = OllamaLabAssistant()

# Step 1: Ask LLM to design experiment
design = assistant.chat("Find which phase has highest TRI")

# Step 2: Execute
params = assistant.parse_experiment_request(design)
result = assistant.run_experiment(params)

# Step 3: Interpret
interpretation = assistant.chat(f"Results: {result['summary']}. What does this mean?")

# Step 4: Iterate
next_experiment = assistant.chat(f"{interpretation}. What should I test next?")
```

**This is the RL scientist workflow automated via LLM.**

## 📈 Scaling Up

### Add More Training Data

Edit `generate_lab_training_data.py`:

```python
def generate_practical_qa() -> List[Dict[str, str]]:
    return [
        # ... existing questions ...
        {
            'instruction': "How do I classify a phase after running an experiment?",
            'output': """After running an experiment:
1. Extract feature vector from result['feature_vector']
2. If you have a trained classifier: detector.classify_phase(feature_vector=fv)
3. The classifier returns prediction and confidence
4. High confidence (>0.8): Strong signal
5. Low confidence (<0.5): Boundary region between phases
..."""
        },
        # Add 50-100 more custom Q&A pairs
    ]
```

Re-run: `python generate_lab_training_data.py`

### Use Larger Model

| Model | Size | VRAM | Inference Speed | Quality |
|-------|------|------|-----------------|---------|
| Qwen2.5-0.5B | 0.5B | 2 GB | ~40 tok/s | Basic |
| Qwen2.5-1.5B | 1.5B | 4 GB | ~25 tok/s | Good ✅ |
| Qwen2.5-3B | 3B | 8 GB | ~15 tok/s | Better |
| Llama-3.2-3B | 3B | 8 GB | ~15 tok/s | Better |
| Phi-3-mini | 3.8B | 10 GB | ~12 tok/s | Best |

**Recommendation**: Start with 1.5B, upgrade if needed.

## 🐛 Troubleshooting

### "Ollama model not found"
```bash
ollama list  # Check installed models
ollama pull qwen2.5:1.5b  # Test base model first
```

### "PhaseDetector import error"
```bash
cd /home/engine/project
python -c "from jarvis5090x import PhaseDetector; print('OK')"
```

### Training OOM
- Use 4-bit quantization: `load_in_4bit=True`
- Reduce batch size: `per_device_train_batch_size=1`
- Enable gradient checkpointing: `use_gradient_checkpointing=True`

### GGUF conversion fails
- Use llama.cpp's `convert-hf-to-gguf.py`
- Merge LoRA into base first (Unsloth: `save_merged_model`)
- Check model architecture compatibility

### Model gives wrong answers
- Check training loss (should decrease to <0.5)
- Increase epochs (try 5 instead of 3)
- Verify training data quality (`head lab_training_data.jsonl`)
- Try higher learning rate (2e-4 → 5e-4)

## 📚 Resources

- **Unsloth**: https://github.com/unslothai/unsloth
- **Axolotl**: https://github.com/OpenAccess-AI-Collective/axolotl
- **Ollama**: https://ollama.ai/
- **llama.cpp**: https://github.com/ggerganov/llama.cpp
- **Qwen2.5 models**: https://huggingface.co/Qwen

## 🎓 Next Level: Multi-Agent Lab

Once you have `ben-lab` working:

### Agent 1: Experiment Designer
- Input: Research question
- Output: Experiment parameters

### Agent 2: Result Interpreter
- Input: Experiment results
- Output: Scientific interpretation

### Agent 3: Hypothesis Generator
- Input: Past results
- Output: New hypotheses to test

### Agent 4: Paper Writer
- Input: Full experiment log
- Output: Draft paper sections

**Orchestrator**: Chains agents together via Ollama API.

## 📝 Citation

If you use this in research:

```
We fine-tuned a domain-specific language model (Qwen2.5-1.5B) on 249 instruction/output
pairs derived from synthetic quantum lab documentation. The resulting model, deployed via
Ollama, can design experiments, interpret results, and guide discovery workflows for
PhaseDetector, enabling automated phase classification and complexity analysis in the
QPR-R (Quantum Phase Recognition with Replay) framework.
```

## ✅ Checklist

- [x] Generate training data (`python generate_lab_training_data.py`)
- [ ] Upload `lab_training_data.jsonl` to cloud GPU
- [ ] Fine-tune base model (Qwen2.5-1.5B recommended)
- [ ] Convert to GGUF Q4_K_M
- [ ] Download GGUF to `ollama/`
- [ ] `ollama create ben-lab -f ollama/Modelfile.example`
- [ ] Test: `ollama run ben-lab`
- [ ] Test integration: `python ollama/ollama_lab_integration.py`
- [ ] (Optional) Extend training data and re-train

---

**Ben Lab is now ready to have its own AI assistant! 🚀**
