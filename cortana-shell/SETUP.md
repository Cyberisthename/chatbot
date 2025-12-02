# Cortana 2.0 Setup Guide

Complete step-by-step guide to get Cortana 2.0 running with your AI infrastructure.

## Quick Start (5 minutes)

```bash
# 1. Install dependencies
cd cortana-shell
npm install

# 2. Make sure Jarvis is running
cd ../
node server.js  # In a separate terminal

# 3. Launch Cortana
cd cortana-shell
npm run dev
```

That's it! Cortana 2.0 will launch with default settings connecting to Jarvis.

## Detailed Setup

### Step 1: Prerequisites

#### Required
- **Node.js 16+**: [Download](https://nodejs.org/)
- **Jarvis Backend**: Your main project's `server.js` should be running at `http://localhost:3001`

#### Optional
- **Ollama** (for local LLM fallback): [Download](https://ollama.com/)
- **Whisper.cpp** (for voice): [GitHub](https://github.com/ggerganov/whisper.cpp)
- **Python 3.8+** (for Whisper alternative)

### Step 2: Install Cortana 2.0

```bash
cd /home/engine/project/cortana-shell

# Install Node.js dependencies
npm install

# This will install:
# - Electron (for desktop app)
# - React (for UI)
# - TypeScript (for type safety)
# - Axios (for HTTP requests)
# - And all other dependencies
```

### Step 3: Configure Backends

#### Option A: Use Default Settings

The default configuration connects to:
- Jarvis: `http://localhost:3001/api/chat`
- Ollama: `http://localhost:11434/api/generate`
- Infinite Capacity: `http://localhost:9000/api/heavy`

If your Jarvis is running at these URLs, you're ready to go!

#### Option B: Custom Configuration

Create a user config file:

**Windows:**
```powershell
mkdir "$env:APPDATA\cortana-shell\config"
copy config\default-config.yaml "$env:APPDATA\cortana-shell\config\user-config.yaml"
notepad "$env:APPDATA\cortana-shell\config\user-config.yaml"
```

**macOS/Linux:**
```bash
mkdir -p ~/.config/cortana-shell/config
cp config/default-config.yaml ~/.config/cortana-shell/config/user-config.yaml
nano ~/.config/cortana-shell/config/user-config.yaml
```

Edit the URLs to match your setup:

```yaml
mode: jarvis  # or "local" or "hybrid"
jarvis_api_url: http://localhost:3001/api/chat
local_llm_url: http://localhost:11434/api/generate
```

### Step 4: Launch Cortana

#### Development Mode (Recommended for testing)

```bash
npm run dev
```

This launches Cortana with:
- Hot reload (changes update instantly)
- DevTools open for debugging
- Verbose logging

#### Production Mode

```bash
# Build first
npm run build

# Then package as distributable
npm run package

# The .exe will be in dist/ folder
```

### Step 5: Verify Connection

1. Cortana window should appear
2. Check status bar:
   - **Green dot + "Ready"** = Connected to Jarvis ✅
   - **Orange warning** = Jarvis offline, using local fallback ⚠️
   - **Red + "Offline"** = No backends available ❌

3. Test by typing: `"Hello, what can you do?"`

If you see a response, you're all set! 🎉

## Advanced Setup

### A. Local LLM (Ollama) as Fallback

1. **Install Ollama:**
   ```bash
   # macOS/Linux
   curl -fsSL https://ollama.com/install.sh | sh

   # Windows
   # Download from https://ollama.com/download
   ```

2. **Start Ollama:**
   ```bash
   ollama serve
   ```

3. **Pull a model:**
   ```bash
   ollama pull llama2
   # or
   ollama pull mistral
   ```

4. **Update Cortana config:**
   ```yaml
   mode: hybrid  # Falls back to local if Jarvis is offline
   local_llm_url: http://localhost:11434/api/generate
   ```

### B. Voice Setup (Whisper STT)

#### Option 1: Whisper.cpp (C++ - Fast)

```bash
# Clone and build
git clone https://github.com/ggerganov/whisper.cpp.git
cd whisper.cpp
make

# Download model
bash ./models/download-ggml-model.sh base.en

# Run server
./server -m models/ggml-base.en.bin --port 9000 --host 127.0.0.1
```

Update Cortana config:
```yaml
voice:
  stt_mode: whisper_http
  stt_endpoint: http://localhost:9000/inference
```

#### Option 2: Python Whisper (Python - Accurate)

```bash
# Install Whisper
pip install openai-whisper flask flask-cors

# Create server.py
cat > whisper_server.py << 'EOF'
import whisper
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
model = whisper.load_model("base.en")

@app.route('/transcribe', methods=['POST'])
def transcribe():
    audio_file = request.files['audio']
    result = model.transcribe(audio_file)
    return jsonify({'text': result['text']})

if __name__ == '__main__':
    app.run(port=9000)
EOF

# Run server
python whisper_server.py
```

Update Cortana config:
```yaml
voice:
  stt_mode: whisper_http
  stt_endpoint: http://localhost:9000/transcribe
```

#### Option 3: Windows Native STT

```yaml
voice:
  stt_mode: windows  # No additional setup needed
```

### C. Wake Word Detection

#### Option 1: Porcupine (Recommended)

```bash
npm install @picovoice/porcupine-node

# Get API key from https://console.picovoice.ai/
```

Update `src/core/voice/WakeWordDetector.ts`:

```typescript
import { Porcupine } from '@picovoice/porcupine-node';

// In start() method:
const porcupine = new Porcupine(
  'YOUR_ACCESS_KEY',
  ['hey cortana'],  // keyword
  [0.5]  // sensitivity
);

// Listen for wake word...
```

#### Option 2: Snowboy (Legacy)

```bash
npm install snowboy

# Train model at https://snowboy.kitt.ai/
# Download trained model
```

Update `WakeWordDetector.ts` to use Snowboy.

### D. Infinite Capacity Backend (Future)

Create a simple stub server for testing:

```python
# infinite_capacity_server.py
from flask import Flask, request, jsonify
import time

app = Flask(__name__)

@app.route('/api/heavy', methods=['POST'])
def heavy_compute():
    data = request.json
    task = data.get('task_description', '')
    
    # Simulate heavy computation
    time.sleep(2)
    
    result = f"Processed task: {task}"
    return jsonify({'result': result})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(port=9000)
```

```bash
python infinite_capacity_server.py
```

Update config:
```yaml
infinite_capacity_url: http://localhost:9000/api/heavy
```

## Troubleshooting

### Issue: "Jarvis Offline" warning

**Diagnosis:**
```bash
# Test Jarvis manually
curl http://localhost:3001/api/health
```

**Solutions:**
1. Start Jarvis server: `node server.js`
2. Check if port 3001 is in use: `lsof -i :3001` (macOS/Linux) or `netstat -ano | findstr :3001` (Windows)
3. Update `jarvis_api_url` in config if using different port

### Issue: Voice not working

**STT not transcribing:**
- Check if Whisper server is running
- Verify `stt_endpoint` in config
- Test: `curl -X POST -F "audio=@test.wav" http://localhost:9000/transcribe`

**TTS not speaking:**
- Windows only feature (PowerShell + System.Speech)
- Check if TTS voices are installed: `Get-InstalledVoice` in PowerShell
- Try different voice in config: `tts_voice: Microsoft Zira Desktop`

**Microphone not capturing:**
- Grant microphone permissions to Electron
- Install `node-record-lpcm16` native dependencies: `npm rebuild node-record-lpcm16`

### Issue: Tools not executing

**"Tool not allowed" error:**
- Check `tools.enabled: true`
- Verify tool name in `allowed_commands`

**"Permission denied" error:**
- Windows: Run as Administrator
- macOS: Grant accessibility permissions to Electron

### Issue: Build/Package fails

**TypeScript errors:**
```bash
npm run lint
npm run format
```

**Native module errors:**
```bash
npm rebuild
npm run build
```

**Electron packaging errors:**
```bash
rm -rf node_modules dist
npm install
npm run build
npm run package
```

## Development Tips

### Hot Reload

Development mode (`npm run dev`) supports hot reload:
- UI changes reload automatically
- TypeScript changes recompile on save
- No need to restart app for most changes

### Debugging

**Main Process:**
Add in `src/main/index.ts`:
```typescript
console.log('Debug:', data);
```

**Renderer Process:**
- DevTools are open by default in dev mode
- Use `console.log()` in React components

**IPC Communication:**
```typescript
// Test IPC from DevTools console
await window.cortana.config.get()
await window.cortana.brain.sendMessage({ prompt: 'test' })
```

### Custom Tools

Add a new tool in `src/core/tools/ToolManager.ts`:

```typescript
private async myCustomTool(arg1: string): Promise<ToolExecutionResult> {
  // Your implementation
  return {
    tool: 'my_custom_tool',
    success: true,
    output: `Executed with ${arg1}`
  };
}
```

Update `execute()` method:
```typescript
case 'my_custom_tool': {
  result = await this.myCustomTool(toolCall.args.arg1 as string);
  break;
}
```

Add to config:
```yaml
tools:
  allowed_commands:
    - my_custom_tool
```

### UI Customization

**Change Colors:**
Edit `src/renderer/styles.css`:
```css
.app {
  background: linear-gradient(135deg, 
    rgba(YOUR_COLOR_1), 
    rgba(YOUR_COLOR_2)
  );
}
```

**Halo Animation:**
```css
.cortana-halo.idle .halo-ring {
  border-color: rgba(0, 195, 255, 0.8);
  animation: rotate 4s linear infinite;
}
```

## Next Steps

1. **Test all features:**
   - Text chat ✓
   - Voice input (if configured) ✓
   - System tools ✓
   - Reminders ✓

2. **Customize:**
   - Change personality prompt in config
   - Add custom tools
   - Adjust UI colors/animations

3. **Deploy:**
   - Package with `npm run package`
   - Distribute `.exe` to other machines
   - Share config guide with users

4. **Extend:**
   - Implement wake word detection
   - Connect Infinite Capacity backend
   - Add plugin system for custom features

## Support

- **Documentation:** See main README.md
- **Issues:** Check console logs and DevTools
- **Configuration:** All settings in config YAML file
- **Updates:** Pull latest changes from repository

---

**Happy Hacking!** 🚀 Now you have your own personal AI assistant!
