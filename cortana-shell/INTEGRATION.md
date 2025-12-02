# Cortana 2.0 ↔ Jarvis Integration Guide

How Cortana 2.0 desktop assistant integrates with your existing AI infrastructure.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Cortana 2.0 (Electron App)                  │
│                                                                 │
│  ┌───────────────┐  ┌────────────────┐  ┌──────────────────┐  │
│  │   React UI    │  │  VoiceManager  │  │  ToolManager     │  │
│  │  (Renderer)   │  │  (STT/TTS)     │  │  (OS Commands)   │  │
│  └───────┬───────┘  └────────┬───────┘  └────────┬─────────┘  │
│          │                   │                    │             │
│          └───────────────────┼────────────────────┘             │
│                              │                                  │
│                    ┌─────────▼─────────┐                        │
│                    │   BrainRouter     │                        │
│                    │  (Main Process)   │                        │
│                    └─────────┬─────────┘                        │
│                              │                                  │
│          ┌───────────────────┼────────────────────┐             │
│          │                   │                    │             │
│   ┌──────▼──────┐   ┌────────▼────────┐  ┌───────▼────────┐   │
│   │   Jarvis    │   │   Local LLM     │  │   Infinite     │   │
│   │   Client    │   │  (Ollama)       │  │   Capacity     │   │
│   └──────┬──────┘   └────────┬────────┘  └───────┬────────┘   │
└──────────┼───────────────────┼────────────────────┼────────────┘
           │                   │                    │
           │ HTTP              │ HTTP               │ HTTP
           │                   │                    │
┌──────────▼──────┐   ┌────────▼────────┐  ┌───────▼────────┐
│  Jarvis Server  │   │ Ollama Server   │  │   Future IC    │
│  (Node.js)      │   │                 │  │   Backend      │
│  Port: 3001     │   │ Port: 11434     │  │  Port: 9000    │
│                 │   │                 │  │                │
│  ┌───────────┐  │   │  ┌───────────┐  │  │                │
│  │  Express  │  │   │  │  llama2   │  │  │  (Your impl.)  │
│  │   API     │  │   │  │  mistral  │  │  │                │
│  │           │  │   │  │  ...      │  │  │                │
│  └─────┬─────┘  │   │  └───────────┘  │  │                │
│        │        │   │                 │  │                │
│  ┌─────▼─────┐  │   └─────────────────┘  └────────────────┘
│  │  Python   │  │
│  │ Inference │  │
│  │ Backend   │  │
│  └───────────┘  │
└─────────────────┘
```

## Communication Flow

### 1. User Input (Text or Voice)

```
User types "Hello" or speaks
  ↓
React UI captures input
  ↓
IPC: sendMessage({ prompt: "Hello" })
  ↓
Main Process: BrainRouter.sendMessage()
```

### 2. Brain Router Decision

```typescript
// BrainRouter.ts

if (config.mode === 'jarvis' || (config.mode === 'hybrid' && jarvisOnline)) {
  response = await jarvisClient.chat(messages);
}
else if (config.mode === 'local') {
  response = await localLLMClient.chat(messages);
}
```

### 3. Jarvis API Call

```typescript
// JarvisClient.ts

const response = await axios.post('http://localhost:3001/api/chat', {
  messages: [
    { role: 'system', content: 'You are Cortana 2.0...' },
    { role: 'user', content: 'Hello' }
  ],
  context: {},
  tools: []
});

// Response:
// {
//   text: "Hello! How can I assist you today?",
//   tool_calls: []
// }
```

### 4. Response Handling

```
BrainRouter receives response
  ↓
Tool execution (if any tool_calls)
  ↓
IPC: emit('brain:response', { reply: "..." })
  ↓
React UI displays message
  ↓
VoiceManager speaks response (if TTS enabled)
```

## Jarvis API Contract

Cortana 2.0 expects Jarvis to provide this API:

### Health Check

```http
GET /api/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00.000Z",
  "llm_ready": true
}
```

### Chat Endpoint

```http
POST /api/chat
Content-Type: application/json

{
  "messages": [
    { "role": "system", "content": "You are..." },
    { "role": "user", "content": "Hello" }
  ],
  "context": {},
  "tools": []
}
```

**Response:**
```json
{
  "text": "Hello! How can I help?",
  "tool_calls": [
    {
      "tool": "open_app",
      "args": { "name": "Chrome" }
    }
  ]
}
```

**Alternative Response Format:**
```json
{
  "message": {
    "content": "Hello! How can I help?"
  }
}
```

Both formats are supported by `JarvisClient.ts`.

## Tool Integration

When Jarvis returns tool calls, Cortana executes them:

### Example: Open Application

**Jarvis Response:**
```json
{
  "text": "Opening Chrome for you.",
  "tool_calls": [
    {
      "tool": "open_app",
      "args": { "name": "chrome" }
    }
  ]
}
```

**Cortana Execution:**
```typescript
// ToolManager.ts
await toolManager.execute({
  tool: 'open_app',
  args: { name: 'chrome' }
});

// Windows: cmd /c start "" "chrome"
// macOS: open -a "chrome"
// Linux: chrome &
```

### Example: Set Reminder

**Jarvis Response:**
```json
{
  "text": "I'll remind you tomorrow at 9am.",
  "tool_calls": [
    {
      "tool": "set_reminder",
      "args": {
        "text": "Call John",
        "time": "2024-01-02T09:00:00Z"
      }
    }
  ]
}
```

**Cortana Execution:**
```typescript
// ReminderService.ts
reminderService.addReminder("Call John", "2024-01-02T09:00:00Z");

// Schedules job with node-schedule
// Fires notification at specified time
```

## Supported Tools

| Tool | Args | Jarvis Should Return |
|------|------|---------------------|
| `open_app` | `{ name: string }` | Application name (e.g., "chrome", "notepad") |
| `open_url` | `{ url: string }` | Full URL with protocol (e.g., "https://github.com") |
| `control_volume` | `{ action: "up" \| "down" \| "mute" \| "unmute" \| "set", value?: number }` | Action + optional volume level |
| `control_media` | `{ action: "play" \| "pause" \| "next" \| "previous" }` | Media control command |
| `create_note` | `{ title: string, content: string }` | Note title and markdown content |
| `set_reminder` | `{ text: string, time: string }` | Reminder text and ISO timestamp |

## Configuration Sync

Cortana's behavior can be configured via YAML:

```yaml
# config/user-config.yaml

mode: jarvis  # or "local" or "hybrid"

jarvis_api_url: http://localhost:3001/api/chat

personality:
  prompt_prefix: "You are Cortana 2.0, powered by Jarvis..."

tools:
  enabled: true
  allowed_commands:
    - open_app
    - set_reminder
    # ... enable/disable specific tools
```

Jarvis doesn't need to be aware of this config - Cortana handles routing.

## Extending Integration

### Add New Tool to Jarvis

1. **Jarvis Side:**
   Update `server.js` to detect intent and return tool call:

   ```javascript
   // In chat handler
   if (message.includes('weather')) {
     return {
       text: "Fetching weather...",
       tool_calls: [{
         tool: 'get_weather',
         args: { location: 'New York' }
       }]
     };
   }
   ```

2. **Cortana Side:**
   Add tool implementation in `ToolManager.ts`:

   ```typescript
   case 'get_weather': {
     result = await this.getWeather(toolCall.args.location as string);
     break;
   }

   private async getWeather(location: string): Promise<ToolExecutionResult> {
     // Fetch from API
     const weather = await fetchWeather(location);
     return {
       tool: 'get_weather',
       success: true,
       output: `Weather in ${location}: ${weather}`
     };
   }
   ```

3. **Enable in Config:**
   ```yaml
   tools:
     allowed_commands:
       - get_weather
   ```

### Add Context from Cortana

Send system state to Jarvis for context-aware responses:

```typescript
// BrainRouter.ts
const context = {
  platform: process.platform,
  time: new Date().toISOString(),
  recent_tools: [...],
  user_preferences: {...}
};

await jarvis.chat(messages, context);
```

Jarvis can use this to provide better responses:

```javascript
// server.js
const { platform, time } = req.body.context;

if (platform === 'win32') {
  // Windows-specific suggestions
}
```

## Fallback Modes

### Hybrid Mode

If Jarvis is offline, automatically fallback to local LLM:

```yaml
mode: hybrid
```

```typescript
// BrainRouter.ts
let response = await jarvis.chat(messages);

if (response.error && config.mode === 'hybrid') {
  console.log('Jarvis offline, using local LLM fallback');
  response = await localLLM.chat(messages);
}
```

### Local-Only Mode

Bypass Jarvis entirely:

```yaml
mode: local
```

All requests go to Ollama/local LLM.

## Performance Optimization

### Connection Pooling

Jarvis client reuses HTTP connections:

```typescript
// JarvisClient.ts
this.client = axios.create({
  baseURL: 'http://localhost:3001',
  timeout: 30000,
  // Axios keeps connections alive by default
});
```

### Health Check Caching

Health checks run every 30 seconds:

```typescript
// BrainRouter.ts
setInterval(() => {
  this.checkJarvisHealth();
}, 30_000);
```

UI shows real-time status without spamming requests.

### Request Retry

Failed requests retry with backoff:

```typescript
// JarvisClient.ts
for (let attempt = 1; attempt <= this.retryAttempts; attempt++) {
  try {
    return await this.client.post('/api/chat', payload);
  } catch (error) {
    if (attempt < this.retryAttempts) {
      await delay(1000 * attempt);
    }
  }
}
```

## Debugging Integration

### Enable Debug Logging

Set in config:
```yaml
advanced:
  log_level: debug
```

### Test Manually

```bash
# Health check
curl http://localhost:3001/api/health

# Chat request
curl -X POST http://localhost:3001/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hello"}
    ]
  }'
```

### Inspect IPC

From Cortana DevTools console:

```javascript
// Get config
await window.cortana.config.get()

// Send test message
await window.cortana.brain.sendMessage({ prompt: 'test' })

// Check status
await window.cortana.brain.getStatus()
```

## Common Issues

### "Jarvis Offline"

- Check if Jarvis server is running: `curl http://localhost:3001/api/health`
- Verify port in config matches Jarvis server port
- Check firewall/antivirus blocking localhost connections

### Tools Not Executing

- Ensure tool name matches exactly in `allowed_commands`
- Check Jarvis response format includes correct tool names
- Verify tool implementation exists in `ToolManager.ts`

### Slow Responses

- Check Jarvis server logs for processing time
- Consider using smaller model (jarvis-7b vs jarvis-34b)
- Enable local LLM fallback for faster responses

## Future: Infinite Capacity Integration

When your IC backend is ready:

1. **Implement API:**
   ```python
   # infinite_capacity_server.py
   @app.route('/api/heavy', methods=['POST'])
   def heavy_compute():
       task = request.json['task_description']
       payload = request.json['payload']
       # Your heavy compute logic
       return jsonify({'result': '...'})
   ```

2. **Update Config:**
   ```yaml
   infinite_capacity_url: http://localhost:9000/api/heavy
   ```

3. **Route Complex Tasks:**
   ```typescript
   // Detect heavy tasks and route to IC
   if (isComplexReasoning(message)) {
     response = await infiniteCapacity.runHeavyReasoning(message, {});
   }
   ```

## Summary

- **Cortana** = Desktop UI + Voice + Tools + Router
- **Jarvis** = AI brain providing conversational responses
- **Integration** = Simple HTTP API contract
- **Extensibility** = Add tools, customize config, plug in backends

Both systems remain modular and can evolve independently!

---

For more details, see:
- Cortana: [`README.md`](README.md)
- Setup: [`SETUP.md`](SETUP.md)
- Jarvis: [`../README.md`](../README.md)
