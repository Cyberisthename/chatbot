export interface CortanaConfig {
  app?: Record<string, unknown>;
  backend?: Record<string, unknown>;
  voice?: Record<string, unknown>;
  tools?: Record<string, unknown>;
  ui?: Record<string, unknown>;
  privacy?: Record<string, unknown>;
}

export interface CortanaChatEntry {
  role: 'user' | 'assistant';
  content: string;
  timestamp: number;
}

export interface CortanaToolResult {
  ok: boolean;
  result?: unknown;
  error?: string;
}

export interface CortanaAPI {
  config: {
    get: () => Promise<CortanaConfig>;
    update: (path: string, value: unknown) => void;
    onChange: (callback: (config: CortanaConfig) => void) => () => void;
  };
  history: {
    get: () => Promise<CortanaChatEntry[]>;
    add: (entry: CortanaChatEntry) => void;
  };
  voice: {
    toggle: (enabled: boolean) => Promise<boolean>;
    onWake: (callback: () => void) => () => void;
  };
  tools: {
    list: () => Promise<Array<{ name: string; description: string }>>;
    execute: (tool: string, args: Record<string, unknown>) => Promise<CortanaToolResult>;
  };
  ui: {
    onUpdate: (callback: (ui: Record<string, unknown>) => void) => () => void;
  };
  external: {
    open: (url: string) => void;
  };
  windowControls: {
    minimize: () => void;
    maximize: () => void;
    close: () => void;
    isMaximized: () => Promise<boolean>;
  };
}
