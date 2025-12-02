# Cortana Shell - Setup Guide

## Quick Start

### 1. Prerequisites

Make sure you have the following installed:
- **Node.js** 16.x or later
- **npm** 7.x or later (comes with Node.js)
- **Git** (for cloning)

### 2. Installation

```bash
# Clone or navigate to the project
cd cortana-shell

# Install all dependencies
npm install
```

This will install:
- Electron (desktop framework)
- React (UI library)
- TypeScript (type safety)
- Vite (build tool)
- electron-builder (packaging)
- Supporting libraries (axios, js-yaml, electron-store, etc.)

### 3. Running in Development

```bash
npm run dev
```

This starts:
- Electron main process with TypeScript compilation
- Vite dev server for hot-reload React UI
- Opens the Cortana Shell window

**Dev Mode Features:**
- Hot reload for UI changes
- DevTools automatically opened
- Source maps enabled
- Fast refresh for React components

### 4. Building for Production

```bash
npm run build
```

This compiles:
- TypeScript main process → `dist/main/`
- React renderer → `dist/renderer/`
- All assets copied to `dist/`

### 5. Packaging Executables

#### Windows (NSIS + MSIX/APPX)
```bash
npm run package:win
```

Generates in `release/`:
- `Cortana Shell Setup X.X.X.exe` - NSIS installer
- `Cortana Shell X.X.X.appx` - Windows Store package

#### macOS (DMG + ZIP)
```bash
npm run package:mac
```

Generates in `release/`:
- `Cortana Shell-X.X.X.dmg` - Disk image
- `Cortana Shell-X.X.X-mac.zip` - Archive

#### Linux (AppImage, DEB, RPM)
```bash
npm run package:linux
```

Generates in `release/`:
- `Cortana Shell-X.X.X.AppImage` - Universal Linux
- `cortana-shell_X.X.X_amd64.deb` - Debian/Ubuntu
- `cortana-shell-X.X.X.x86_64.rpm` - Fedora/RHEL

## 🔧 Configuration

### Backend Setup

The Cortana Shell requires a backend for AI functionality. Ensure one of these is running:

#### Option 1: JARVIS Backend (Recommended)
```bash
# From the main project directory
cd ..
npm start
```

JARVIS will start on `http://localhost:3001`

#### Option 2: Ollama (Local LLM)
```bash
# Install Ollama
curl https://ollama.ai/install.sh | sh

# Pull a model
ollama pull llama2

# Start server (runs on http://localhost:11434)
ollama serve
```

#### Option 3: Custom Backend
Edit `config.yaml`:
```yaml
backend:
  custom:
    enabled: true
    url: "http://your-backend:port/api/chat"
```

### Configuration File

The `config.yaml` file contains all settings:

```yaml
app:
  name: "Cortana Shell"
  version: "2.0.0"
  hotkey: "CommandOrControl+Alt+C"  # Global shortcut
  startMinimized: false
  startOnBoot: false

backend:
  jarvis:
    enabled: true
    url: "http://localhost:3001/api/chat"
    timeout: 30000
  ollama:
    enabled: true
    url: "http://localhost:11434"
    model: "llama2"

voice:
  wakeWord:
    enabled: true
    phrase: "hey cortana"
    sensitivity: 0.5
    modelPath: "./assets/wake-word-models/heycortana_enUS.table"
  stt:
    provider: "whisper"
    language: "en-US"
  tts:
    provider: "system"
    voice: "Microsoft Zira Desktop"

ui:
  animations:
    halo: true
    particles: true
  glassmorphism:
    enabled: true
    blur: 20
  colors:
    primary: "#0078D7"
```

**Hot Reload:** Changes to `config.yaml` are applied immediately without restart!

## 🎨 Assets

### Wake Word Models

Wake word detection models are copied from the original Cortana APK:

```
cortana-shell/assets/wake-word-models/
├── heycortana_enUS.table       (US English - 707KB)
├── heycortana_en-US.table      (US English alt - 845KB)
├── heycortana_enIN.table       (Indian English - 85KB)
└── heycortana_zhCN.table       (Simplified Chinese - 85KB)
```

These files enable offline voice detection.

### Icons & Branding

Extracted from Cortana APK:

```
cortana-shell/assets/
├── icons/
│   └── cortana.png             (High-res menu icon)
└── branding/
    └── cortanadouble.png       (Logo with double ring)
```

For packaging, additional icon formats are needed:
- Windows: `cortana.ico` (256x256, 16-bit)
- macOS: `cortana.icns` (multiple resolutions)
- Linux: `cortana.png` (512x512 recommended)

## 🚀 Development Workflow

### Project Structure

```
cortana-shell/
├── src/
│   ├── main/                   # Electron main process (Node.js)
│   │   ├── index.ts            # Entry point
│   │   ├── preload.ts          # Renderer bridge
│   │   └── services/
│   │       ├── ConfigManager.ts
│   │       ├── VoiceManager.ts
│   │       ├── ToolManager.ts
│   │       └── HistoryStore.ts
│   └── renderer/               # React UI (Browser)
│       ├── App.tsx             # Main component
│       ├── main.tsx            # React entry
│       ├── styles.css          # Cortana styling
│       └── services/
│           └── BrainRouter.ts  # Backend communication
├── assets/                     # Static resources
├── dist/                       # Compiled output
├── release/                    # Packaged executables
├── config.yaml                 # User configuration
├── package.json                # Dependencies & scripts
├── tsconfig.json               # TypeScript base config
├── tsconfig.main.json          # Main process config
├── tsconfig.renderer.json      # Renderer config
└── vite.config.ts              # Vite bundler config
```

### Available Scripts

```bash
npm run dev              # Development mode (hot reload)
npm run build            # Build for production
npm run build:main       # Compile main process only
npm run build:renderer   # Build React UI only
npm start                # Start built application
npm run package          # Package for current platform
npm run package:win      # Windows (NSIS + APPX)
npm run package:mac      # macOS (DMG + ZIP)
npm run package:linux    # Linux (AppImage, DEB, RPM)
```

### Adding a New Tool

1. Edit `src/main/services/ToolManager.ts`
2. Register your tool:

```typescript
this.register('my_tool', 'Description of my tool', async (args) => {
  // Tool implementation
  const result = await doSomething(args);
  return { status: 'ok', data: result };
});
```

3. Update `config.yaml`:

```yaml
tools:
  enabled:
    - my_tool
```

4. Use in UI via IPC:

```typescript
const result = await window.cortana.tools.execute('my_tool', {
  param1: 'value',
  param2: 123
});
```

### Debugging

#### Main Process
- DevTools automatically open in dev mode
- Console logs appear in terminal
- Use `console.log()` in main process code

#### Renderer Process
- DevTools accessible via `Ctrl+Shift+I`
- React DevTools available
- Use browser debugging tools

#### IPC Communication
Enable IPC debugging:

```typescript
// In main process
ipcMain.on('*', (event, ...args) => {
  console.log('IPC:', event.channel, args);
});
```

## 📦 Distributing

### Code Signing (Optional but Recommended)

#### Windows
Obtain a code signing certificate and configure:

```json
// In package.json build.win
"certificateFile": "path/to/cert.pfx",
"certificatePassword": "password"
```

#### macOS
Configure Apple Developer ID:

```json
// In package.json build.mac
"identity": "Developer ID Application: Your Name"
```

### Auto-Updates

Add electron-updater:

```bash
npm install electron-updater
```

Configure in `package.json`:

```json
"build": {
  "publish": {
    "provider": "github",
    "owner": "your-username",
    "repo": "cortana-shell"
  }
}
```

## 🐛 Troubleshooting

### Port Already in Use (Dev Mode)
```bash
# Kill existing Vite process
pkill -f vite
# Or change port in vite.config.ts
```

### Build Fails
```bash
# Clear caches
rm -rf dist/ node_modules/
npm install
npm run build
```

### APPX Packaging Issues (Windows)
- Ensure publisher matches certificate
- Check `appx.identityName` is unique
- Validate app manifest

### Electron Binary Not Found
```bash
# Reinstall Electron
npm install electron --save-dev
```

### IPC Not Working
- Check `contextIsolation: true` in BrowserWindow
- Verify preload script path is correct
- Ensure `nodeIntegration: false`

## 🔒 Security Best Practices

1. **Never disable `contextIsolation`** - Keeps renderer process sandboxed
2. **Keep `nodeIntegration: false`** - Prevents direct Node access from renderer
3. **Use preload scripts** - Safe IPC bridge between main and renderer
4. **Validate inputs** - Sanitize all user input before executing
5. **CSP Headers** - Content Security Policy for web content
6. **Update dependencies** - `npm audit` regularly

## 📚 Additional Resources

- [Electron Documentation](https://www.electronjs.org/docs)
- [React Documentation](https://react.dev)
- [TypeScript Handbook](https://www.typescriptlang.org/docs/)
- [Vite Guide](https://vitejs.dev/guide/)
- [electron-builder Docs](https://www.electron.build/)

## 💡 Tips

- Use `npm run dev` for rapid development
- Keep config.yaml in version control (without secrets)
- Test on all target platforms before release
- Enable source maps for debugging
- Use TypeScript strict mode for safety
- Profile with Chrome DevTools Performance tab

---

**Ready to build amazing AI experiences!** 🚀
