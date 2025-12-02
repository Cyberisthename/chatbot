# Cortana Shell - Changelog

## [2.0.0] - 2024-12-02

### 🎉 Initial Release - Complete Rebuild

#### Added from Original Cortana APK Assets
- **Wake Word Models**: Integrated 4 official wake word detection models
  - `heycortana_enUS.table` (707KB) - US English primary model
  - `heycortana_en-US.table` (845KB) - US English alternative
  - `heycortana_enIN.table` (85KB) - Indian English
  - `heycortana_zhCN.table` (85KB) - Simplified Chinese
- **Icons & Branding**: High-resolution icons from APK drawable resources
  - Cortana menu icon (xxhdpi quality)
  - Official Cortana double-ring logo
- **UI Inspiration**: Layout and design elements from original Cortana
  - Voice button styling
  - Profile circle layouts
  - Widget information structure

#### Core Features
- **Electron-based Desktop Application**
  - Cross-platform support (Windows, macOS, Linux)
  - Transparent frameless window with drag support
  - System tray integration with context menu
  - Global hotkey: `Ctrl+Alt+C` (customizable)
  
- **Classic Cortana UI/UX**
  - Iconic blue halo animation with pulsing rings
  - Glassmorphism design (backdrop blur, translucent panels)
  - Smooth fade-in animations for messages
  - Cortana blue color scheme (#0078D7, #00BCF2)
  - Custom title bar with window controls
  
- **AI Backend Integration**
  - BrainRouter for intelligent backend selection
  - JARVIS backend support (http://localhost:3001/api/chat)
  - Ollama local LLM integration (http://localhost:11434)
  - Infinite Capacity heavy computation backend
  - Automatic fallback and routing
  
- **Voice Interaction System**
  - VoiceManager with wake word detection support
  - Models loaded from original Cortana APK
  - STT support (Whisper, Browser API)
  - TTS integration (system native voices)
  - Voice button with active state animation
  
- **Productivity Tools**
  - `open_url`: Launch URLs in default browser
  - `open_app`: Execute desktop applications
  - `control_volume`: System volume adjustment
  - `control_media`: Media playback controls
  - `create_note`: Save Markdown notes to disk
  - `set_reminder`: Schedule notifications
  - `search_web`: Web search integration
  - `system_info`: Hardware and OS diagnostics
  
- **Configuration Management**
  - YAML-based configuration with hot-reload
  - ConfigManager service with file watching
  - Real-time config updates without restart
  - Comprehensive settings: app, backend, voice, ui, tools, privacy
  
- **Chat & History**
  - Persistent conversation storage with electron-store
  - HistoryStore service with 1000-message limit
  - Role-based messages (user/assistant)
  - Timestamps for all interactions
  - History clearing functionality
  
- **Developer Experience**
  - TypeScript throughout (strict mode)
  - React with hooks for UI
  - Vite for fast hot-reload development
  - IPC bridge with type-safe preload script
  - Separated main/renderer process configs
  - DevTools auto-open in development
  
#### Build & Packaging
- **electron-builder** configuration for all platforms
- **Windows**: NSIS installer + APPX/MSIX package
  - Microsoft Store ready with app manifest
  - APPX identity: `CortanaShell`
  - Background color: Cortana blue
- **macOS**: DMG disk image + ZIP archive
  - App category: Productivity
- **Linux**: AppImage, DEB, RPM packages
  - Category: Utility
  
#### Documentation
- Comprehensive README.md with features, architecture, usage
- Detailed SETUP.md with installation, configuration, development
- CHANGELOG.md tracking all changes
- Inline code documentation and comments
- TypeScript type definitions

#### Developer Tools
- Hot-reload for both main and renderer processes
- Concurrent development script
- Separate build scripts for main/renderer
- Platform-specific packaging commands
- Source maps for debugging

#### Security & Privacy
- Context isolation enabled (no direct Node access)
- Node integration disabled in renderer
- Preload script as IPC bridge
- Local-only data storage
- No telemetry collection
- Configurable privacy settings

### Technical Details

#### Dependencies
**Main:**
- `electron`: ^28.0.0
- `electron-store`: ^8.1.0 (persistent storage)
- `js-yaml`: ^4.1.0 (config parsing)
- `node-schedule`: ^2.1.1 (reminders)
- `axios`: ^1.6.0 (HTTP requests)
- `socket.io-client`: ^4.7.0 (real-time communication)

**Renderer:**
- `react`: ^18.2.0
- `react-dom`: ^18.2.0

**Dev:**
- `typescript`: ^5.3.0
- `vite`: ^5.0.0
- `electron-builder`: ^24.9.0
- `concurrently`: ^8.2.0

#### Architecture
```
Main Process (Node.js/Electron)
  ├── ConfigManager (YAML hot-reload)
  ├── VoiceManager (wake word detection)
  ├── ToolManager (productivity tools)
  └── HistoryStore (conversation persistence)
  
Renderer Process (React/Browser)
  ├── App Component (UI state management)
  └── BrainRouter (backend communication)
  
IPC Bridge (Preload Script)
  └── Exposed window.cortana API
```

#### File Structure
- `src/main/` - Electron main process (TypeScript)
- `src/renderer/` - React UI (TypeScript + JSX)
- `assets/` - Wake word models, icons, branding
- `config.yaml` - User configuration (hot-reload)
- `dist/` - Compiled output
- `release/` - Packaged executables

### Assets Attribution
All wake word models and icons are extracted from:
- **Source**: Cortana APK v3.2.0.12583 (en-US release)
- **Original Publisher**: Microsoft Corporation
- **Usage**: Educational and personal use
- **Acknowledgment**: Microsoft Cortana team for original design

### Future Roadmap
- Full voice pipeline implementation
- Advanced wake word detection (Porcupine/Snowboy)
- Plugin system for extensibility
- Weather, calendar, email integrations
- Multi-language UI support
- Custom theme engine
- Context-aware responses
- Calendar and reminder sync

---

**Cortana Shell 2.0** - Bringing back the classic assistant experience with modern AI capabilities! 💙
