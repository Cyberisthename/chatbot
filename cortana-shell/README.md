# Cortana 2.0 – Jarvis Shell

A beautiful Windows desktop assistant that recreates the classic Cortana experience using your own AI infrastructure (Jarvis, local LLMs, and Infinite Capacity backend).

## Features

✨ **Classic Cortana UI**
- Beautiful blue halo/ring animation with state-based effects
- Listening/Thinking/Speaking visual indicators
- Fluent Design-inspired colors and glassmorphism effects
- Conversation history with chat bubbles
- Frameless, transparent window with drag support

🧠 **AI Brain Router**
- Primary: Jarvis API (your custom AI backend)
- Secondary: Local LLMs via Ollama
- Future: Infinite Capacity backend for heavy compute
- Automatic fallback between backends
- Real-time status indicators

🎤 **Voice Interaction**
- Wake word detection: "Hey Cortana" (extensible)
- Speech-to-Text via Whisper (local or HTTP)
- Text-to-Speech using Windows native voices
- Mic button and hotkey activation

🛠️ **System Control Tools**
- Open applications
- Open URLs
- Control volume (up/down/mute)
- Control media playback
- Create text notes (saved as Markdown)
- Set reminders with notifications

⚙️ **Configuration**
- YAML-based configuration
- Hot-reload support
- Mode switching (Jarvis/Local/Hybrid)
- Customizable hotkeys, voices, and UI settings

## Architecture

```
cortana-shell/
├── src/
│   ├── main/               # Electron main process
│   │   └── index.ts        # App initialization, window management, IPC handlers
│   ├── preload/            # IPC bridge
│   │   └── index.ts        # Secure context bridge for renderer
│   ├── renderer/           # Front-end UI
│   │   ├── index.html      # HTML entry point
│   │   ├── main.tsx        # React app with Cortana UI
│   │   └── styles.css      # Cortana animations & styling
│   └── core/               # Core business logic
│       ├── brain/          # AI router & backends
│       │   ├── BrainRouter.ts          # Routes requests to backends
│       │   ├── JarvisClient.ts         # Jarvis API client
│       │   ├── LocalLLMClient.ts       # Ollama/local LLM client
│       │   └── InfiniteCapacityClient.ts  # Future IC backend (stub)
│       ├── voice/          # Voice interaction
│       │   ├── VoiceManager.ts         # Voice orchestration
│       │   ├── WakeWordDetector.ts     # "Hey Cortana" detection (extensible)
│       │   ├── SpeechToText.ts         # Whisper STT integration
│       │   └── TextToSpeech.ts         # Windows TTS integration
│       ├── tools/          # System control
│       │   ├── ToolManager.ts          # Tool execution engine
│       │   └── ReminderService.ts      # Local reminder scheduler
│       ├── config/         # Configuration management
│       │   └── ConfigManager.ts        # YAML config with hot-reload
│       └── types/          # TypeScript types
│           └── index.ts
├── config/
│   └── default-config.yaml # Default configuration
├── assets/
│   ├── icons/              # App & tray icons
│   └── sounds/             # Sound effects (optional)
├── package.json
├── tsconfig.json
├── electron.vite.config.ts
└── README.md
```

## Installation

### Prerequisites

- Node.js 16+ and npm
- Windows 10+ (for native TTS/media control)
- **Jarvis Backend** running at `http://localhost:3001` (or configure different URL)
- (Optional) Ollama or local LLM server at `http://localhost:11434`

### Setup

```bash
# Navigate to the Cortana Shell directory
cd cortana-shell

# Install dependencies
npm install

# Development mode (with hot reload)
npm run dev

# Build for production
npm run build

# Package as distributable .exe
npm run package
```

## Configuration

Edit `config/default-config.yaml` or create a user config at:
- Windows: `%APPDATA%/cortana-shell/config/user-config.yaml`
- macOS: `~/Library/Application Support/cortana-shell/config/user-config.yaml`
- Linux: `~/.config/cortana-shell/config/user-config.yaml`

### Example Configuration

```yaml
# AI Mode: "jarvis" | "local" | "hybrid"
mode: jarvis

# Backend URLs
jarvis_api_url: http://localhost:3001/api/chat
local_llm_url: http://localhost:11434/api/generate
infinite_capacity_url: http://localhost:9000/api/heavy

# Voice Configuration
voice:
  wake_word_enabled: true
  wake_word: "hey cortana"
  stt_mode: whisper_local  # "whisper_local" | "whisper_http" | "windows"
  stt_endpoint: http://localhost:9000/whisper/transcribe
  tts_mode: windows
  tts_voice: Microsoft David Desktop

# UI Configuration
ui:
  hotkey: CommandOrControl+Alt+C
  theme: blue
  always_on_top: false
  start_minimized: false
  minimize_to_tray: true
  window_opacity: 0.95

# System Commands
tools:
  enabled: true
  allowed_commands:
    - open_app
    - open_url
    - control_volume
    - control_media
    - create_note
    - set_reminder

# Personality
personality:
  name: Cortana
  greeting: "Hello! I'm Cortana 2.0, powered by your personal AI infrastructure."
  prompt_prefix: "You are Cortana 2.0, a helpful AI assistant..."
```

## How to Connect Backends

### 1. Jarvis Backend (Primary)

Make sure your Jarvis server is running:

```bash
# In the main project directory
node server.js
# Or
npm start
```

Jarvis should expose a REST API at `http://localhost:3001/api/chat` that accepts:

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "Hello"}
  ],
  "context": {},
  "tools": []
}
```

**Response:**
```json
{
  "text": "Hello! How can I help you?",
  "tool_calls": []
}
```

### 2. Local LLM (Ollama)

Install and run Ollama:

```bash
# Install Ollama (see https://ollama.com)
ollama serve

# Pull a model
ollama pull llama2

# Ollama will be available at http://localhost:11434
```

### 3. Infinite Capacity Backend (Future)

This is a placeholder for your future experimental compute backend. The interface is already defined in `InfiniteCapacityClient.ts`:

```typescript
// Example stub implementation
await infiniteCapacity.runHeavyReasoning("Solve complex problem", { data: "..." });
await infiniteCapacity.runBatchSearch("Research topic", { depth: 5 });
```

To implement:
1. Create a REST API at `http://localhost:9000/api/heavy`
2. Accept `{ task_description: string, payload: object }`
3. Return `{ result: string }`

## Voice Setup

### Wake Word Detection

**Current Status:** Stub implementation (prints to console)

**To Implement:**

1. **Porcupine** (Recommended):
   ```bash
   npm install @picovoice/porcupine-node
   ```
   Update `WakeWordDetector.ts` to use Porcupine API.

2. **Snowboy** (Deprecated but works):
   ```bash
   npm install snowboy
   ```
   Train custom "Hey Cortana" model at https://snowboy.kitt.ai

3. **Custom Model:**
   Integrate any wake word library in `WakeWordDetector.ts`.

### Speech-to-Text (Whisper)

**Option 1: Local Whisper.cpp**

```bash
# Clone and build whisper.cpp
git clone https://github.com/ggerganov/whisper.cpp.git
cd whisper.cpp
make

# Download model
bash ./models/download-ggml-model.sh base.en

# Run Whisper server
./server -m models/ggml-base.en.bin --port 9000
```

Update config: `stt_mode: whisper_http` and `stt_endpoint: http://localhost:9000/inference`

**Option 2: Python Whisper**

```bash
pip install openai-whisper flask

# Create a simple Flask server (see implementation in SpeechToText.ts comments)
# Then set stt_endpoint to your Flask server
```

**Option 3: Windows STT**

Set `stt_mode: windows` to use Windows native speech recognition (basic, no additional setup needed).

### Text-to-Speech

**Windows:** Uses `System.Speech.Synthesis` via PowerShell (works out of the box).

**List Available Voices:**
```powershell
Add-Type -AssemblyName System.Speech
(New-Object System.Speech.Synthesis.SpeechSynthesizer).GetInstalledVoices() | ForEach-Object { $_.VoiceInfo.Name }
```

**Custom TTS:**
Implement in `TextToSpeech.ts` and set `tts_mode: custom`.

## Usage

### Basic Usage

1. **Launch Cortana:**
   - Run `npm run dev` (development) or launch the packaged app
   - Use hotkey `Ctrl+Alt+C` (or configured hotkey)
   - Click the tray icon

2. **Type a message:**
   - Type in the input box and press Enter
   - Cortana will process via Jarvis/Local LLM and respond

3. **Voice interaction:**
   - Click the mic button 🎙️ to start listening
   - Say your command
   - Cortana will transcribe and respond

4. **Wake word (when implemented):**
   - Say "Hey Cortana"
   - Window will appear and start listening

### Example Commands

```
"What can you do?"
"Open Chrome"
"Set volume to 50"
"Create a note called 'Meeting Notes' with content 'Discuss Q4 goals'"
"Set a reminder for tomorrow at 9am to call John"
"Tell me about quantum computing"
"Play music" (sends media play/pause)
```

### System Tools

Cortana can execute these tools automatically when detected in conversation:

- **open_app**: `"Open Chrome"`, `"Launch Steam"`
- **open_url**: `"Open github.com"`, `"Go to youtube.com"`
- **control_volume**: `"Volume up"`, `"Set volume to 75"`, `"Mute"`
- **control_media**: `"Play music"`, `"Next track"`, `"Pause"`
- **create_note**: `"Create a note called X with content Y"`
- **set_reminder**: `"Remind me to X at 3pm tomorrow"`

## Development

### Project Structure

- **Main Process** (`src/main/index.ts`): Manages the Electron app, windows, system integration
- **Preload** (`src/preload/index.ts`): Secure IPC bridge between main and renderer
- **Renderer** (`src/renderer/main.tsx`): React UI with Cortana animations
- **Core Logic** (`src/core/`): All business logic (AI, voice, tools, config)

### Adding a New Backend

1. Create client in `src/core/brain/` (e.g., `MyBackendClient.ts`)
2. Implement interface compatible with `BrainResponse`
3. Add to `BrainRouter.ts` as an option
4. Update config schema in `ConfigManager.ts`

### Adding a New Tool

1. Add tool implementation in `ToolManager.ts`
2. Add tool name to `default-config.yaml` allowed_commands
3. Tool will be auto-executed when AI requests it

### Customizing the UI

- **Animations**: Edit `src/renderer/styles.css`
- **Colors**: Change gradient/halo colors in CSS
- **Layout**: Modify `src/renderer/main.tsx` React components

## Building and Distribution

### Development Build

```bash
npm run dev
```

Opens Electron with hot-reload for fast iteration.

### Production Build

```bash
npm run build
```

Compiles TypeScript and bundles for production.

### Package as Installer

```bash
npm run package
```

Creates distributable in `dist/` folder:
- Windows: `Cortana-2.0-Setup-x.x.x.exe`
- Portable: `Cortana-2.0-x.x.x.exe`

### Distribution Files

The packaged app includes:
- Compiled Electron app
- Default configuration
- All assets (icons, sounds)

User data (config, notes, reminders) is stored in app data directory.

## Troubleshooting

### Cortana shows "Offline" or "Jarvis Offline"

- Ensure Jarvis backend is running at configured URL
- Check `jarvis_api_url` in config
- Test manually: `curl http://localhost:3001/api/health`

### Voice not working

- **Wake word:** Requires additional setup (see Voice Setup section)
- **STT:** Ensure Whisper server is running or set `stt_mode: windows`
- **TTS:** Windows only - check if Windows TTS voices are installed

### Tools not executing

- Check `tools.enabled: true` in config
- Verify tool name in `allowed_commands`
- Check console for permission errors

### Window not showing

- Press configured hotkey (`Ctrl+Alt+C`)
- Check system tray for Cortana icon
- Disable `start_minimized` in config

## Roadmap

- [x] Core UI with Cortana animations
- [x] Brain router (Jarvis/Local LLM/IC)
- [x] System tools (apps, volume, notes, reminders)
- [x] Voice interfaces (STT/TTS)
- [ ] Wake word detection implementation
- [ ] Advanced tool system (Python script execution, API calls)
- [ ] Infinite Capacity backend integration
- [ ] Robotics/external system control
- [ ] Multi-language support
- [ ] Plugin system for custom tools
- [ ] Visual customization (themes, colors)
- [ ] Mobile companion app

## License

MIT License - See LICENSE file for details.

## Credits

- Inspired by Microsoft Cortana (Windows 10/Mobile)
- Built with Electron, React, TypeScript
- Powered by your personal AI infrastructure (Jarvis, Ollama, etc.)

---

**Cortana 2.0** – Your AI, Your Rules, Your Shell 🚀
