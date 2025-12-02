# Cortana Shell 2.0 🎯

A modern desktop assistant application featuring the classic Cortana experience with AI capabilities powered by JARVIS backend and local LLM support.

## ✨ Features

### 🎨 Classic Cortana UI
- **Blue Halo Animation**: Iconic pulsing blue rings animation
- **Glassmorphism Design**: Modern frosted glass UI with blur effects
- **Transparent Window**: Seamless integration with your desktop
- **Smooth Animations**: Fluid transitions and visual effects

### 🗣️ Voice Interaction
- **Wake Word Detection**: Integrated models from original Cortana APK
  - `heycortana_enUS.table` (US English)
  - `heycortana_en-US.table` (Alternative US model)
  - `heycortana_enIN.table` (Indian English)
  - `heycortana_zhCN.table` (Simplified Chinese)
- **Speech-to-Text**: Multiple provider support (Whisper, Browser API)
- **Text-to-Speech**: System native voice synthesis
- **Global Hotkey**: `Ctrl+Alt+C` (configurable)

### 🧠 AI Backend Integration
- **JARVIS Backend**: Primary AI endpoint (`http://localhost:3001/api/chat`)
- **Ollama Local LLM**: Fallback local model support
- **Infinite Capacity**: Heavy computation backend (optional)
- **Intelligent Routing**: Automatic backend selection

### 🛠️ Productivity Tools
- `open_url` - Open URLs in default browser
- `open_app` - Launch desktop applications
- `control_volume` - System volume control
- `control_media` - Media playback controls
- `create_note` - Save notes to Markdown files
- `set_reminder` - Schedule reminders
- `search_web` - Web search integration
- `system_info` - System diagnostics

### 💾 Privacy & Storage
- **Local History**: Conversation storage with configurable retention
- **Offline Mode**: Full functionality without internet
- **No Telemetry**: Privacy-first design
- **Encrypted Storage**: Secure credential management

## 📦 Installation

### Prerequisites
- Node.js 16+
- npm or yarn
- Operating System: Windows 10+, macOS 10.15+, or Linux

### Setup

```bash
# Navigate to cortana-shell directory
cd cortana-shell

# Install dependencies
npm install

# Run in development mode
npm run dev

# Build for production
npm run build

# Package for distribution
npm run package

# Platform-specific packaging
npm run package:win    # Windows (NSIS + APPX/MSIX)
npm run package:mac    # macOS (DMG + ZIP)
npm run package:linux  # Linux (AppImage, DEB, RPM)
```

## ⚙️ Configuration

Edit `config.yaml` to customize:

```yaml
app:
  hotkey: "CommandOrControl+Alt+C"
  startMinimized: false
  
backend:
  jarvis:
    enabled: true
    url: "http://localhost:3001/api/chat"
  ollama:
    enabled: true
    url: "http://localhost:11434"
    model: "llama2"
    
voice:
  wakeWord:
    enabled: true
    phrase: "hey cortana"
    sensitivity: 0.5
  tts:
    provider: "system"
    voice: "Microsoft Zira Desktop"
    
ui:
  animations:
    halo: true
    particles: true
  glassmorphism:
    blur: 20
    opacity: 0.8
  colors:
    primary: "#0078D7"
    secondary: "#00BCF2"
```

The configuration file supports **hot-reload** - changes are applied instantly without restart.

## 🏗️ Architecture

```
cortana-shell/
├── src/
│   ├── main/              # Electron main process
│   │   ├── index.ts       # Application entry point
│   │   ├── preload.ts     # IPC bridge
│   │   └── services/      # Core services
│   │       ├── ConfigManager.ts    # Config hot-reload
│   │       ├── VoiceManager.ts     # Voice detection
│   │       ├── ToolManager.ts      # Tool execution
│   │       └── HistoryStore.ts     # Chat persistence
│   └── renderer/          # React UI
│       ├── App.tsx        # Main component
│       ├── styles.css     # Cortana styling
│       └── services/
│           └── BrainRouter.ts      # Backend routing
├── assets/
│   ├── wake-word-models/  # Voice detection models
│   ├── icons/             # App icons (from APK)
│   └── branding/          # Cortana branding assets
└── config.yaml            # User configuration
```

## 🎯 Usage

### Keyboard Shortcut
Press `Ctrl+Alt+C` (Windows/Linux) or `Cmd+Alt+C` (macOS) to show/hide Cortana.

### Voice Activation
Say "Hey Cortana" to wake the assistant (when enabled).

### Chat Interface
Type your questions or commands in the text input and press Enter or click the send button.

### System Tray
- Double-click tray icon to open
- Right-click for quick settings
- Toggle halo animation
- Quit application

## 📱 Building MSIX/APPX Package

The application is configured to build Windows Store-compatible APPX packages:

```bash
npm run package:win
```

This generates:
- `release/Cortana Shell Setup.exe` (NSIS installer)
- `release/Cortana Shell.appx` (Windows Store package)

The APPX package includes:
- App identity: `CortanaShell`
- Publisher: `CN=Cortana`
- Background color: `#0078D7` (Cortana blue)
- Store-ready manifest

## 🔧 Development

### Hot Reload
Development mode supports hot reload for both main and renderer processes:

```bash
npm run dev
```

### TypeScript
All code is written in TypeScript with strict type checking:

```bash
# Type check
npx tsc --noEmit

# Build main process
npm run build:main

# Build renderer
npm run build:renderer
```

### IPC Communication
Use the exposed `window.cortana` API in renderer:

```typescript
// Get configuration
const config = await window.cortana.config.get();

// Update configuration
window.cortana.config.update('ui.animations.halo', false);

// Execute tool
const result = await window.cortana.tools.execute('open_url', {
  url: 'https://example.com'
});
```

## 🎨 Assets from Original Cortana

This project incorporates authentic assets from the original Cortana APK:

- **Wake Word Models**: Original detection models for multi-language support
- **Icons**: High-resolution Cortana icons (xxhdpi assets)
- **Branding**: Official Cortana double-ring logo
- **UI Elements**: Voice button and profile layouts

All assets are used respectfully and for educational/personal use.

## 🚀 Roadmap

- [ ] Full voice pipeline (STT + TTS)
- [ ] Advanced wake word detection with Porcupine/Snowboy
- [ ] Plugin system for extensibility
- [ ] Cross-platform notifications
- [ ] Weather, calendar, and email integrations
- [ ] Multi-language support
- [ ] Context awareness
- [ ] Custom themes

## 📄 License

MIT License - see LICENSE file for details.

Original Cortana assets are property of Microsoft Corporation.

## 🙏 Acknowledgments

- Microsoft Cortana team for the original design inspiration
- Wake word models extracted from Cortana 3.2.0 APK
- JARVIS AI System for backend integration
- Electron and React communities

---

**Cortana Shell 2.0** - Classic assistant, modern technology 💙
