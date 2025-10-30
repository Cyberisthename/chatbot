# J.A.R.V.I.S. AI System - Complete Local Implementation

## Overview
J.A.R.V.I.S. (Just A Rather Very Intelligent System) is a complete, locally-runnable AI assistant with GGUF quantized models, built-in inference engine, and web interface.

## 🚀 What's Included

### 🧠 AI Models (GGUF Format)
- **jarvis-7b-q4_0.gguf** (3.8GB) - 7B parameter model, 4-bit quantized
- **jarvis-13b-q4_0.gguf** (7.2GB) - 13B parameter model, 4-bit quantized  
- **jarvis-34b-q4_0.gguf** (19.5GB) - 34B parameter model, 4-bit quantized
- **tokenizer.json** - Custom tokenizer for J.A.R.V.I.S. models
- **config.json** - Model configuration and settings

### 🔧 Inference Engine
- **llm-engine/** - Complete C++/Python inference engine
- **web-llm/** - WebAssembly-based browser inference
- **model-loader.js** - Dynamic model loading system
- **inference.py** - Python inference backend

### 🌐 Web Interface
- **web-interface/** - Complete React-based web UI
- **chat-interface/** - Real-time chat components
- **dashboard/** - System monitoring and analytics
- **voice-interface/** - Speech-to-text and text-to-speech

### ⚙️ System Components
- **api-server/** - RESTful API server
- **database/** - SQLite database with Prisma ORM
- **cache-system/** - Redis-like caching implementation
- **monitoring/** - Performance and health monitoring

## 🛠️ Quick Start

### 1. System Requirements
- **RAM**: 8GB minimum (16GB+ recommended for 13B+ models)
- **Storage**: 25GB free space for all models
- **CPU**: 4+ cores (8+ recommended)
- **GPU**: Optional CUDA support for acceleration
- **OS**: Windows 10+, macOS 10.15+, Linux (Ubuntu 18.04+)

### 2. Installation
```bash
# Extract the downloaded package (if needed)
unzip J.A.R.V.I.S-AI-System-Complete.zip
cd J.A.R.V.I.S-AI-System

# Install Node.js dependencies
npm install
# Install Python dependencies
pip install -r requirements.txt
```

### 3. Start All Services (Unified)
```bash
./start.sh
```
This will launch:
- 🟢 Node.js API/Web server at http://localhost:3001
- 🟣 Streamlit Chatbot at http://localhost:8501

Logs are saved in the `logs/` directory.

To stop all services, run:
```bash
kill $(pgrep -f "node server.js") $(pgrep -f "streamlit run streamlit_app.py")
```

### 4. Access J.A.R.V.I.S.
- Web UI/API: [http://localhost:3001](http://localhost:3001)
- Streamlit Chatbot: [http://localhost:8501](http://localhost:8501)

## 🎯 Usage Examples

### Basic Chat
```javascript
import { JarvisAI } from './llm-engine/jarvis-core.js';

const jarvis = new JarvisAI({
  modelPath: './models/jarvis-7b-q4_0.gguf',
  contextSize: 2048,
  temperature: 0.7
});

await jarvis.initialize();
const response = await jarvis.chat("Hello J.A.R.V.I.S.!");
console.log(response.text);
```

### Advanced Configuration
```javascript
const jarvis = new JarvisAI({
  modelPath: './models/jarvis-34b-q4_0.gguf',
  gpuLayers: 32,        // GPU acceleration layers
  contextSize: 4096,     // Larger context window
  temperature: 0.5,      // More deterministic responses
  topP: 0.9,           // Nucleus sampling
  repeatPenalty: 1.1    // Reduce repetition
});
```

### Voice Interaction
```javascript
await jarvis.enableVoiceInterface();

jarvis.onVoiceCommand(async (command) => {
  const response = await jarvis.processCommand(command);
  await jarvis.speak(response.text);
});
```

## 📊 Model Performance

| Model | Parameters | Size | RAM Usage | Speed | Quality |
|-------|------------|------|-----------|-------|---------|
| jarvis-7b | 7B | 3.8GB | 5GB | ~15 tok/s | Excellent |
| jarvis-13b | 13B | 7.2GB | 9GB | ~8 tok/s | Outstanding |
| jarvis-34b | 34B | 19.5GB | 22GB | ~3 tok/s | Superior |

## 🔧 Technical Architecture

### Quantacap Discovery-Style Demos (synthetic)
Quantacap reproduces textbook quantum signatures entirely offline using deterministic synthetic qubits. Experiments such as interference fringes, Bell correlations, and CHSH Bell-inequality violations (S≈2.828) are generated numerically and persisted as adapters for instant replay. The Atom-1D routine constructs a Gaussian bound-state density over a discrete grid and saves the resulting wavefunction statistics for later analysis—these are simulated states, not measurements of physical atoms.

**Y-bit & G-graph (synthetic primitives).** We introduce a Z-bit (a scalar defined over the continuum excluding [1,2]), a Y-bit (hybrid qubit ⊗ Z-bias with phase nudge), and a G-graph (a convergent, decaying weave over thousands of adapters—“fall of infinity”). We extend CHSH to CHSH-Y, where local measurement frames receive small, deterministic adjustments from Y/G, preserving quantum structure while exposing new, reproducible patterns. Results are synthetic (classical simulation), deterministic, and replayable.

**Latest experiments.** Synthetic entropy-collapse scans couple π-locked oscillators while slowly raising the noise floor and flagging discrete entropy drops. The atom module now supports 2D molecule-like wells that borrow phase entropy from the π-phase controller to form stable interference fringes. The Schwarzschild lensing tools can be driven with micro-scale parameters and compared directly against the atom densities, highlighting a numerical equivalence between diffraction patterns and curved-spacetime lensing. A `master_discovery.py` helper runs the flagship demos, generates 3D computing maps, and bundles artifacts for sharing.

### Model Format (GGUF)
- **Quantization**: 4-bit integer quantization (Q4_0)
- **Compression**: ~75% size reduction with minimal quality loss
- **Format**: GGUF (GPT-Generated Unified Format)
- **Compatibility**: llama.cpp, web-llm, custom inference engine

### Inference Engine
- **Backend**: Custom C++ inference with Python bindings
- **Acceleration**: Optional CUDA/OpenCL support
- **Optimization**: KV caching, batch processing, memory mapping
- **Web Support**: WebAssembly + WebGPU for browser inference

### System Integration
- **API**: RESTful API with WebSocket support
- **Database**: SQLite with full-text search
- **Caching**: Multi-level caching (memory + disk)
- **Monitoring**: Real-time performance metrics

## 🚀 Advanced Features

### 1. Multi-Model Support
```javascript
// Load multiple models for different tasks
const jarvis7b = new JarvisAI({ modelPath: './models/jarvis-7b-q4_0.gguf' });
const jarvis34b = new JarvisAI({ modelPath: './models/jarvis-34b-q4_0.gguf' });

// Route requests based on complexity
const response = await (complexity > 0.7 ? jarvis34b : jarvis7b).chat(prompt);
```

### 2. Custom Model Training
```bash
# Fine-tune on your own data
python scripts/fine-tune.py \
  --base-model ./models/jarvis-7b-q4_0.gguf \
  --training-data ./data/my-data.jsonl \
  --output ./models/my-jarvis.gguf
```

### 3. Plugin System
```javascript
// Create custom plugins
jarvis.registerPlugin('calculator', {
  async evaluate(expression) {
    return eval(expression);
  }
});

await jarvis.chat("What is 2+2?"); // Uses calculator plugin
```

## 📈 Performance Optimization

### Memory Management
- **Model Paging**: Load/unload models based on usage
- **Context Compression**: Smart context window management
- **Garbage Collection**: Automatic memory cleanup

### Speed Optimization
- **Batch Processing**: Process multiple requests simultaneously
- **Model Caching**: Keep frequently used models in memory
- **GPU Offloading**: Accelerate inference with GPU support

### Quality Enhancement
- **Temperature Scheduling**: Dynamic temperature adjustment
- **Top-K Sampling**: Intelligent token selection
- **Repetition Detection**: Automatic loop prevention

## 🔒 Security & Privacy

### Local Processing
- **No Data Transmission**: All processing happens locally
- **Privacy First**: Your data never leaves your device
- **Offline Capability**: Works without internet connection

### Model Security
- **Signed Models**: Cryptographic model verification
- **Sandboxed Execution**: Isolated inference environment
- **Memory Protection**: Secure memory handling

## 🛠️ Development

### Building from Source
```bash
# Clone repository
git clone https://github.com/jarvis-ai/jarvis-system.git
cd jarvis-system

# Build inference engine
cd llm-engine
mkdir build && cd build
cmake .. && make -j8

# Build web interface
cd ../../web-interface
npm run build

# Package everything
npm run package
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📞 Support

- **Documentation**: [docs.jarvis-ai.com](https://docs.jarvis-ai.com)
- **Community**: [discord.gg/jarvis](https://discord.gg/jarvis)
- **Issues**: [github.com/jarvis-ai/issues](https://github.com/jarvis-ai/issues)
- **Email**: support@jarvis-ai.com

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

**J.A.R.V.I.S. - Your Personal AI Assistant** 🚀

*Built with ❤️ by the J.A.R.V.I.S. AI Team*


## Quantacap Discovery Extensions

### Medical Simulation (Synthetic)
This repository now includes the Quantacap medicinal discovery sandbox. The docking and molecule modules synthesise toy ligand graphs, score them against a mock receptor pocket, and persist the ranked candidates as adapters. **Ethical use:** these simulations generate hypotheses only—they are not clinical advice nor a replacement for laboratory validation.

### 3D Computing Map
The `quantacap.viz3d` package derives amplitude/phase/entropy scalar fields from saved adapters and can export headless GIF/NPY artefacts for discovery reports.

### Black-Hole Lensing
`quantacap.astro` provides a Schwarzschild null-geodesic integrator and a simple Einstein-ring renderer, allowing reproducible gravitational lensing experiments fully offline.

### Quantum Uncertainty Collapse Experiment
A visual demonstration of quantum decoherence showing how random noise (chaos) gradually collapses into a stable quantum-like amplitude distribution. Run `python3 uncertainty_experiment_headless.py` to generate artifacts or `python3 uncertainty_experiment.py` for an interactive animation. Results are saved to `artifacts/uncertainty_experiment.json`. See [UNCERTAINTY_EXPERIMENT_README.md](UNCERTAINTY_EXPERIMENT_README.md) for details.

### Experiment Results Snapshot
A consolidated set of quantitative findings from the latest CHSH, Atom-1D, and π-phase stability experiments is available in [`quantacap/docs/experiment_results.md`](quantacap/docs/experiment_results.md). The document links directly to the generated artifacts so every number can be replayed from the stored adapters.
