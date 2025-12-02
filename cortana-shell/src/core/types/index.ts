export interface AppConfig {
  mode: 'jarvis' | 'local' | 'hybrid';
  jarvis_api_url: string;
  local_llm_url: string;
  infinite_capacity_url: string;
  voice: VoiceConfig;
  ui: UIConfig;
  tools: ToolsConfig;
  personality: PersonalityConfig;
  advanced: AdvancedConfig;
}

export interface VoiceConfig {
  wake_word_enabled: boolean;
  wake_word: string;
  stt_mode: 'whisper_local' | 'whisper_http' | 'windows';
  stt_endpoint?: string;
  tts_mode: 'windows' | 'custom';
  tts_voice?: string;
}

export interface UIConfig {
  hotkey: string;
  theme: string;
  always_on_top: boolean;
  start_minimized: boolean;
  minimize_to_tray: boolean;
  window_opacity: number;
}

export interface ToolsConfig {
  enabled: boolean;
  allowed_commands: string[];
}

export interface PersonalityConfig {
  name: string;
  greeting: string;
  prompt_prefix: string;
}

export interface AdvancedConfig {
  context_messages: number;
  request_timeout: number;
  retry_attempts: number;
  log_level: 'debug' | 'info' | 'warn' | 'error';
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: number;
  tool_calls?: ToolCall[];
}

export interface ToolCall {
  tool: string;
  args: Record<string, unknown>;
  result?: unknown;
  message?: string;
}

export interface ToolExecutionResult {
  tool: string;
  success: boolean;
  output?: unknown;
  error?: string;
}

export interface BrainResponse {
  reply: string;
  tool_calls?: ToolCall[];
  backend_used?: 'jarvis' | 'local' | 'infinite_capacity';
  error?: string;
}

export interface BrainRouterRequest {
  prompt: string;
  messages?: ChatMessage[];
  context?: Record<string, unknown>;
  forceBackend?: 'jarvis' | 'local' | 'infinite_capacity';
  allowTools?: boolean;
}

export interface BrainRouterResult extends BrainResponse {
  toolResults?: ToolExecutionResult[];
}

export interface VoiceState {
  listening: boolean;
  thinking: boolean;
  speaking: boolean;
}

export type CortanaState = 'idle' | 'listening' | 'thinking' | 'speaking' | 'offline';

export interface Reminder {
  id: string;
  text: string;
  time: string | Date;
  createdAt: number;
  status: 'pending' | 'completed' | 'cancelled';
}
