# Cortana Shell - Quick Start Guide

Get up and running with Cortana Shell in 5 minutes!

## Prerequisites ✅

- Node.js 16+ installed
- npm 7+ installed
- JARVIS backend running (optional but recommended)

## Step 1: Navigate to Directory

```bash
cd cortana-shell
```

## Step 2: Install Dependencies

```bash
npm install
```

This takes ~2-3 minutes and installs:
- Electron (desktop framework)
- React (UI)
- TypeScript (type safety)
- Build tools

## Step 3: Start JARVIS Backend (Recommended)

In a separate terminal:

```bash
cd ..
npm start
```

This starts JARVIS on `http://localhost:3001`

**Skip this if**: You want to use Ollama or another backend.

## Step 4: Run Cortana Shell

```bash
npm run dev
```

This opens:
- Electron window with Cortana UI
- DevTools for debugging
- Vite dev server for hot-reload

## You're Ready! 🎉

Try these:
- Type a message and hit Enter
- Press `Ctrl+Alt+C` to show/hide
- Click the microphone button (voice not yet fully implemented)
- Right-click system tray icon for settings

## Common Issues

### "Cannot connect to backend"
➜ Start JARVIS backend: `cd .. && npm start`

### "Port already in use"
➜ Kill existing process: `pkill -f vite`

### "Electron not found"
➜ Reinstall: `npm install electron --save-dev`

## Next Steps

- Read [README.md](README.md) for full documentation
- Check [SETUP.md](SETUP.md) for configuration options
- See [INTEGRATION.md](INTEGRATION.md) for JARVIS integration
- Review [ASSETS.md](ASSETS.md) for asset information

## Building for Production

```bash
# Build
npm run build

# Package for your OS
npm run package

# Or platform-specific
npm run package:win    # Windows
npm run package:mac    # macOS
npm run package:linux  # Linux
```

---

**That's it!** You're running Cortana Shell 💙
