# Cortana Shell - JARVIS Integration Guide

## Overview

Cortana Shell integrates with the existing JARVIS AI System to provide a desktop assistant experience. This guide explains how the two systems work together.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Cortana Shell (Electron)                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   React UI  │  │ Voice Mgr   │  │  Tool Mgr   │         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
│         │                 │                 │                 │
│         └────────┬────────┴────────┬────────┘                │
│                  │  BrainRouter    │                         │
│                  └────────┬────────┘                         │
└───────────────────────────┼──────────────────────────────────┘
                            │ HTTP/WebSocket
┌───────────────────────────┼──────────────────────────────────┐
│                           ▼                                   │
│              ┌────────────────────────┐                      │
│              │   JARVIS API Server    │                      │
│              │  (localhost:3001)      │                      │
│              └────────┬───────────────┘                      │
│                       │                                       │
│    ┌──────────────────┼──────────────────┐                  │
│    │                  │                  │                   │
│    ▼                  ▼                  ▼                   │
│  ┌─────────┐    ┌──────────┐      ┌──────────┐            │
│  │ Ollama  │    │ Inference│      │ Socket.IO│            │
│  │  LLM    │    │  Backend │      │   Chat   │            │
│  └─────────┘    └──────────┘      └──────────┘            │
│                                                              │
│               JARVIS AI System                               │
└──────────────────────────────────────────────────────────────┘
```

## Starting Both Systems

### 1. Start JARVIS Backend (Required)

```bash
# From the main project directory
cd /home/engine/project
npm start
```

This starts:
- Node.js server on `http://localhost:3001`
- REST API at `/api/chat`
- Socket.IO chat server
- Python inference backend (if configured)

**Verify it's running:**
```bash
curl -X POST http://localhost:3001/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello"}'
```

### 2. Start Cortana Shell

```bash
# From the cortana-shell directory
cd cortana-shell
npm run dev
```

This starts:
- Electron desktop application
- React UI with hot-reload
- Connects to JARVIS backend automatically

## Communication Flow

### Chat Message Flow

1. **User Input** → Cortana Shell UI
2. **BrainRouter** → Selects backend (JARVIS/Ollama/Infinite Capacity)
3. **HTTP POST** → JARVIS API endpoint
4. **JARVIS Processing** → LLM inference + tools
5. **Response** → Returns to Cortana Shell
6. **UI Update** → Displays assistant message

### Example Request

```typescript
// In Cortana Shell
const response = await fetch('http://localhost:3001/api/chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    message: 'What is the weather today?',
    context: previousMessages,
  })
});

const data = await response.json();
// { reply: "I don't have access to real-time weather...", ... }
```

## Backend Configuration

### config.yaml Settings

```yaml
backend:
  jarvis:
    enabled: true
    url: "http://localhost:3001/api/chat"
    timeout: 30000
    
  ollama:
    enabled: true
    url: "http://localhost:11434"
    model: "llama2"
    
  infiniteCapacity:
    enabled: false
    url: "http://localhost:9000/api/heavy"
```

### Backend Selection Logic

The BrainRouter automatically selects a backend:

1. **Primary**: JARVIS (if enabled and url is set)
2. **Fallback**: Ollama (if JARVIS fails)
3. **Heavy Compute**: Infinite Capacity (for complex tasks)

## Tool Integration

### Available Tools

Cortana Shell provides tools that can be called by JARVIS:

```typescript
// ToolManager in Cortana Shell
tools = {
  'open_url': 'Open a URL in browser',
  'open_app': 'Launch desktop application',
  'control_volume': 'Adjust system volume',
  'create_note': 'Save note to Markdown file',
  'system_info': 'Get OS and hardware info',
  // ... more tools
}
```

### Tool Execution Flow

```
JARVIS Response → Contains tool call → Cortana Shell executes → Returns result
```

Example JARVIS response:
```json
{
  "reply": "Opening Google for you",
  "tool": "open_url",
  "args": { "url": "https://google.com" }
}
```

Cortana Shell handles execution:
```typescript
if (response.tool) {
  await window.cortana.tools.execute(response.tool, response.args);
}
```

## Voice Integration

### Wake Word Detection

Cortana Shell handles wake word detection locally:

```yaml
voice:
  wakeWord:
    enabled: true
    phrase: "hey cortana"
    modelPath: "./assets/wake-word-models/heycortana_enUS.table"
```

When wake word is detected:
1. Cortana Shell activates microphone
2. Records user speech
3. Sends to JARVIS for STT (if enabled)
4. Processes response and speaks back via TTS

### Speech-to-Text (STT)

**Option 1: Use JARVIS Backend**
```yaml
voice:
  stt:
    provider: "whisper"
    whisperUrl: "http://localhost:9000/api/transcribe"
```

**Option 2: Browser API**
```yaml
voice:
  stt:
    provider: "browser"
    language: "en-US"
```

**Option 3: System STT (Windows)**
```yaml
voice:
  stt:
    provider: "system"
```

### Text-to-Speech (TTS)

Cortana Shell handles TTS locally:

```yaml
voice:
  tts:
    provider: "system"
    voice: "Microsoft Zira Desktop"  # Windows
    rate: 1.0
    pitch: 1.0
```

## Data Sharing

### History Synchronization

Cortana Shell stores conversations locally. To sync with JARVIS:

```typescript
// Send history to JARVIS for context
const history = await window.cortana.history.get();

fetch('http://localhost:3001/api/context', {
  method: 'POST',
  body: JSON.stringify({ history })
});
```

### Configuration Sharing

Both systems can share configuration via YAML files:

```bash
# Cortana Shell config
cortana-shell/config.yaml

# JARVIS config
project/config.json
```

## Deployment Scenarios

### Scenario 1: Single Machine (Development)

```
Localhost:
  ├── JARVIS (port 3001)
  └── Cortana Shell (Electron app)
```

Configuration:
```yaml
backend:
  jarvis:
    url: "http://localhost:3001/api/chat"
```

### Scenario 2: Network Deployment

```
Server:
  └── JARVIS (192.168.1.100:3001)

Client Machines:
  ├── Cortana Shell 1
  ├── Cortana Shell 2
  └── Cortana Shell 3
```

Configuration:
```yaml
backend:
  jarvis:
    url: "http://192.168.1.100:3001/api/chat"
```

### Scenario 3: Cloud Backend

```
Cloud:
  └── JARVIS (https://jarvis.example.com/api/chat)

Local:
  └── Cortana Shell
```

Configuration:
```yaml
backend:
  jarvis:
    url: "https://jarvis.example.com/api/chat"
```

## API Endpoints

### JARVIS Endpoints Used by Cortana Shell

```
POST /api/chat
  Request: { message: string, context?: object }
  Response: { reply: string, tool?: string, args?: object }

GET /api/status
  Response: { status: 'online', models: [...] }

POST /api/context
  Request: { history: array }
  Response: { ok: true }

WebSocket /socket.io
  Events: message, typing, presence
```

## Troubleshooting

### Connection Issues

**Problem**: Cortana Shell can't connect to JARVIS

```bash
# Check if JARVIS is running
curl http://localhost:3001/api/status

# Check firewall
sudo ufw allow 3001/tcp

# Check JARVIS logs
tail -f logs/jarvis-node.log
```

**Problem**: "Disconnected" status in Cortana Shell

1. Verify JARVIS backend URL in `config.yaml`
2. Check CORS settings in JARVIS `server.js`
3. Test connectivity with curl/Postman

### Performance Issues

**Problem**: Slow responses

- Increase timeout in config.yaml
- Use Ollama for faster local inference
- Check JARVIS backend load

**Problem**: High memory usage

- Limit conversation history in HistoryStore
- Reduce context window size
- Restart Cortana Shell periodically

### Voice Issues

**Problem**: Wake word not detected

- Check model file exists: `assets/wake-word-models/`
- Verify microphone permissions
- Adjust sensitivity in config.yaml

**Problem**: TTS not working

- Windows: Install voice packs
- macOS: Check System Preferences → Accessibility
- Linux: Install `espeak` or `festival`

## Development Tips

### Adding Custom Endpoints

In JARVIS `server.js`:

```javascript
app.post('/api/cortana/custom', async (req, res) => {
  const { action, params } = req.body;
  // Custom logic
  res.json({ result: 'success' });
});
```

In Cortana Shell:

```typescript
const response = await fetch('http://localhost:3001/api/cortana/custom', {
  method: 'POST',
  body: JSON.stringify({ action: 'do_something', params: {} })
});
```

### Extending BrainRouter

```typescript
// Add custom backend
private async sendToCustom(prompt: string): Promise<string> {
  const url = this.config?.custom?.url;
  const response = await fetch(url, {
    method: 'POST',
    body: JSON.stringify({ prompt })
  });
  return (await response.json()).result;
}
```

### Testing Integration

```bash
# Terminal 1: JARVIS
cd /home/engine/project
npm start

# Terminal 2: Cortana Shell
cd cortana-shell
npm run dev

# Terminal 3: Test
curl -X POST http://localhost:3001/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello from Cortana Shell"}'
```

## Security Considerations

### Authentication

Add API key authentication:

```yaml
# config.yaml
backend:
  jarvis:
    url: "http://localhost:3001/api/chat"
    apiKey: "your-secret-key"
```

```typescript
// BrainRouter
const response = await fetch(url, {
  headers: {
    'Authorization': `Bearer ${apiKey}`
  }
});
```

### HTTPS

Use HTTPS for network deployments:

```yaml
backend:
  jarvis:
    url: "https://jarvis.example.com/api/chat"
    verifySsl: true
```

## Next Steps

1. ✅ Start JARVIS backend
2. ✅ Configure `cortana-shell/config.yaml`
3. ✅ Run Cortana Shell in dev mode
4. 🚀 Test chat interaction
5. 🎤 Enable voice features
6. 📦 Package for distribution

**Integration complete!** Cortana Shell now has the power of JARVIS AI 💪
