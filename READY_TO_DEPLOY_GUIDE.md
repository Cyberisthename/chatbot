# JARVIS Quantum AI - Hugging Face Ready

This repository has been organized for direct deployment to Hugging Face Spaces.

## Deployment Package
The fully trained model and UI are located in the `ready-to-deploy-hf/` directory.

### How to Deploy
1. Create a new Space on Hugging Face (choose Gradio SDK).
2. Copy the contents of `ready-to-deploy-hf/` to your Space repository.
3. Push to Hugging Face.

### Contents of `ready-to-deploy-hf/`
- `app.py`: Premium Gradio UI with Quantum State Analysis.
- `jarvis_quantum_llm.npz`: Fully trained model weights (trained from scratch).
- `tokenizer.json`: Trained tokenizer.
- `config.json`: Model architecture configuration.
- `src/`: Core Quantum Transformer library.
- `requirements.txt`: Necessary Python dependencies.

## Local Training
If you wish to train the model further, you can use `train_for_hf.py` in the root directory.
It will automatically update the artifacts in the `ready-to-deploy-hf/` folder.

## Scientific Background
This model is a Quantum-Inspired Transformer implemented in pure NumPy. It features:
- Quantum Superposition-based Attention
- Real-valued complex interference patterns
- From-scratch backpropagation through all layers
- No pre-trained weights or external AI libraries (pure science).
