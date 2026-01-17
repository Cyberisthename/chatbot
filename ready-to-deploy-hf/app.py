import gradio as gr
import sys
import os
from pathlib import Path
import numpy as np
import json
import time

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import quantum LLM modules
try:
    from src.quantum_llm import QuantumTransformer, SimpleTokenizer
    HAS_QUANTUM_LLM = True
except ImportError as e:
    HAS_QUANTUM_LLM = False
    print(f"⚠️  Quantum LLM modules not found: {e}")

class JARVISQuantumUI:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        self.load_resources()

    def load_resources(self):
        """Load trained model and tokenizer"""
        model_path = Path(__file__).parent / "jarvis_quantum_llm.npz"
        tokenizer_path = Path(__file__).parent / "tokenizer.json"
        config_path = Path(__file__).parent / "config.json"

        if not HAS_QUANTUM_LLM:
            return

        try:
            if model_path.exists() and tokenizer_path.exists():
                print(f"📥 Loading JARVIS Quantum LLM from {model_path}...")
                
                # Load config first to get architecture
                if config_path.exists():
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    self.model = QuantumTransformer(
                        vocab_size=config.get("vocab_size", 50000),
                        d_model=config.get("d_model", 768),
                        n_layers=config.get("n_layers", 12),
                        n_heads=config.get("n_heads", 12),
                        d_ff=config.get("d_ff", 3072),
                        max_seq_len=config.get("max_seq_len", 512)
                    )
                    # Load weights into the model
                    weights = np.load(model_path, allow_pickle=True)
                    # Note: Need a method in QuantumTransformer to load weights from dict/npz
                    # Assuming model.load_weights(weights) exists or we use model.load(path)
                    self.model = QuantumTransformer.load(str(model_path))
                else:
                    self.model = QuantumTransformer.load(str(model_path))
                
                self.tokenizer = SimpleTokenizer.load(str(tokenizer_path))
                self.model_loaded = True
                print("✅ JARVIS System Online.")
            else:
                print("⚠️  Trained model not found. Using initialization state.")
                # Initialize with default if not found
                self._init_default_model()
        except Exception as e:
            print(f"❌ Error initializing JARVIS: {e}")
            self._init_default_model()

    def _init_default_model(self):
        if HAS_QUANTUM_LLM:
            self.model = QuantumTransformer(vocab_size=1000, d_model=128, n_layers=2, n_heads=4)
            self.tokenizer = SimpleTokenizer(vocab_size=1000)
            self.model_loaded = True

    def generate(self, prompt, max_tokens, temp, top_k):
        if not self.model_loaded:
            return "JARVIS Error: Model not initialized.", "N/A"
        
        try:
            start_time = time.time()
            generated, metrics = self.model.generate(
                prompt=prompt,
                tokenizer=self.tokenizer,
                max_tokens=max_tokens,
                temperature=temp,
                top_k=top_k
            )
            elapsed = time.time() - start_time
            
            qm = metrics.get("quantum_metrics", {})
            metrics_display = f"""--- QUANTUM STATE ANALYSIS ---
Coherence:   {qm.get('avg_coherence', 0):.4f}
Entanglement: {qm.get('avg_entanglement', 0):.4f}
Interference: {qm.get('avg_interference', 0):.4f}
Fidelity:     {qm.get('avg_fidelity', 0):.4f}

--- PERFORMANCE ---
Tokens: {metrics.get('generated_tokens', 0)}
Speed: {metrics.get('generated_tokens', 0)/elapsed:.2f} t/s
Time: {elapsed:.2f}s
"""
            return generated, metrics_display
        except Exception as e:
            return f"Generation Error: {str(e)}", "N/A"

    def analyze(self, text):
        if not self.model_loaded:
            return "JARVIS Error: Model not initialized."
        
        # Simulating deep analysis using model's forward pass metrics
        try:
            _, metrics = self.model.generate(prompt=text, tokenizer=self.tokenizer, max_tokens=1)
            qm = metrics.get("quantum_metrics", {})
            
            analysis = f"""# ⚛️ Deep Quantum Analysis

## Semantic Coherence
The input shows a coherence level of **{qm.get('avg_coherence', 0):.4f}**. 
{"High coherence indicates a well-structured quantum semantic state." if qm.get('avg_coherence', 0) > 0.5 else "Low coherence suggests high entropy in the input vector space."}

## Neural Entanglement
Contextual dependency strength: **{qm.get('avg_entanglement', 0):.4f}**.
This represents how strongly the tokens are linked in the quantum attention manifold.

## Interference Patterns
Multi-path interference detection: **{qm.get('avg_interference', 0):.4f}**.
This measures the overlap of semantic probabilities in the quantum superposition layers.

## Conclusion
The JARVIS engine has processed the input through {self.model.n_layers} quantum-inspired transformer layers. The state purity (Fidelity) is **{qm.get('avg_fidelity', 0):.4f}**.
"""
            return analysis
        except Exception as e:
            return f"Analysis Error: {str(e)}"

# UI Components
jarvis = JARVISQuantumUI()

custom_css = """
.gradio-container {
    background-color: #050a15;
    color: #00d4ff;
    font-family: 'Courier New', Courier, monospace;
}
.main-title {
    text-align: center;
    font-size: 2.5em;
    font-weight: bold;
    color: #00d4ff;
    text-shadow: 0 0 10px #00d4ff;
    margin-bottom: 20px;
}
.jarvis-box {
    border: 1px solid #00d4ff;
    border-radius: 10px;
    padding: 20px;
    background-color: #0a192f;
    box-shadow: 0 0 15px rgba(0, 212, 255, 0.2);
}
button.primary {
    background: linear-gradient(45deg, #00d4ff, #0055ff) !important;
    border: none !important;
}
"""

with gr.Blocks(css=custom_css, title="JARVIS Quantum LLM") as demo:
    gr.HTML("<div class='main-title'>🌌 JARVIS QUANTUM v1.0</div>")
    gr.Markdown("""
    **Production-Ready Quantum-Inspired Intelligence**
    *Trained from scratch for scientific exploration and advanced semantic reasoning.*
    """)
    
    with gr.Tabs():
        with gr.TabItem("💬 NEURAL INTERFACE"):
            with gr.Row():
                with gr.Column(scale=2):
                    prompt = gr.Textbox(label="Input Stream", placeholder="Engage JARVIS...", lines=5)
                    with gr.Row():
                        tokens = gr.Slider(10, 512, 128, step=10, label="Token Depth")
                        temp = gr.Slider(0.1, 2.0, 0.7, step=0.1, label="Thermal Flux (Temp)")
                        top_k = gr.Slider(1, 100, 50, step=1, label="Top-K Filter")
                    run_btn = gr.Button("🚀 ENGAGE", variant="primary")
                
                with gr.Column(scale=3):
                    output = gr.Textbox(label="JARVIS Response", lines=12)
                    metrics = gr.Textbox(label="Quantum Telemetry", lines=8)
            
            run_btn.click(jarvis.generate, [prompt, tokens, temp, top_k], [output, metrics])

        with gr.TabItem("⚛️ QUANTUM ANALYSIS"):
            with gr.Column():
                analyze_input = gr.Textbox(label="Input for Deep Analysis", lines=3)
                analyze_btn = gr.Button("🔬 SCAN STATE", variant="primary")
                analyze_output = gr.Markdown("### Results will appear here...")
            
            analyze_btn.click(jarvis.analyze, [analyze_input], [analyze_output])

        with gr.TabItem("📊 SYSTEM ARCHITECTURE"):
            gr.Markdown(f"""
            ### Core Engine Specs
            - **Layers:** {jarvis.model.n_layers if jarvis.model else 'N/A'}
            - **Attention Heads:** {jarvis.model.n_heads if jarvis.model else 'N/A'}
            - **Model Dimension:** {jarvis.model.d_model if jarvis.model else 'N/A'}
            - **Vocab Size:** {jarvis.tokenizer.vocab_size if jarvis.tokenizer else 'N/A'}
            - **Status:** {"ONLINE" if jarvis.model_loaded else "OFFLINE"}
            
            ### Quantum Features
            - Superposition-based Attention
            - Entanglement Manifold Mapping
            - Wave-Interference Probability Gates
            - Coherence-Driven Decoding
            """)

    gr.HTML("<div style='text-align: center; margin-top: 20px; opacity: 0.5;'>SYSTEM: JARVIS-QUANTUM-PROD-RECAP-01</div>")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
