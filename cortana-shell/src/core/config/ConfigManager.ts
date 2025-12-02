import { EventEmitter } from 'node:events';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { watch } from 'node:fs/promises';
import path from 'node:path';
import YAML from 'yaml';
import merge from 'deepmerge';
import { z } from 'zod';
import type { AdvancedConfig, AppConfig, PersonalityConfig, ToolsConfig, UIConfig, VoiceConfig } from '../types/index.js';

const VoiceSchema = z.object({
  wake_word_enabled: z.boolean().default(true),
  wake_word: z.string().min(3).default('hey cortana'),
  stt_mode: z.enum(['whisper_local', 'whisper_http', 'windows']).default('whisper_local'),
  stt_endpoint: z.string().optional(),
  tts_mode: z.enum(['windows', 'custom']).default('windows'),
  tts_voice: z.string().optional()
});

const UISchema = z.object({
  hotkey: z.string().default('CommandOrControl+Alt+C'),
  theme: z.string().default('blue'),
  always_on_top: z.boolean().default(false),
  start_minimized: z.boolean().default(false),
  minimize_to_tray: z.boolean().default(true),
  window_opacity: z.number().min(0.2).max(1).default(0.95)
});

const ToolsSchema = z.object({
  enabled: z.boolean().default(true),
  allowed_commands: z.array(z.string()).default([
    'open_app',
    'open_url',
    'control_volume',
    'control_media',
    'create_note',
    'set_reminder'
  ])
});

const PersonalitySchema = z.object({
  name: z.string().default('Cortana'),
  greeting: z.string().default("Hello! I'm Cortana 2.0, powered by your personal AI infrastructure."),
  prompt_prefix: z
    .string()
    .default(
      'You are Cortana 2.0, a helpful AI assistant running on a custom AI stack. You are friendly, professional, and slightly witty. Provide clear, concise answers.'
    )
});

const AdvancedSchema = z.object({
  context_messages: z.number().min(1).max(50).default(10),
  request_timeout: z.number().min(5_000).max(120_000).default(30_000),
  retry_attempts: z.number().min(0).max(5).default(3),
  log_level: z.enum(['debug', 'info', 'warn', 'error']).default('info')
});

const AppConfigSchema = z
  .object({
    mode: z.enum(['jarvis', 'local', 'hybrid']).default('jarvis'),
    jarvis_api_url: z.string().url().default('http://localhost:3001/api/chat'),
    local_llm_url: z.string().url().default('http://localhost:11434/api/generate'),
    infinite_capacity_url: z.string().default('http://localhost:9000/api/heavy'),
    voice: VoiceSchema,
    ui: UISchema,
    tools: ToolsSchema,
    personality: PersonalitySchema,
    advanced: AdvancedSchema
  })
  .transform((value) => value satisfies AppConfig);

export interface ConfigManagerOptions {
  defaultConfigPath: string;
  userConfigDir: string;
  userConfigFilename?: string;
}

export class ConfigManager extends EventEmitter {
  private readonly defaultConfigPath: string;
  private readonly userConfigDir: string;
  private readonly userConfigPath: string;
  private currentConfig: AppConfig;
  private fileWatcher?: AsyncIterableIterator<unknown>;

  constructor(options: ConfigManagerOptions) {
    super();
    this.defaultConfigPath = options.defaultConfigPath;
    this.userConfigDir = options.userConfigDir;
    this.userConfigFilename = options.userConfigFilename ?? 'user-config.yaml';
    this.userConfigPath = path.join(this.userConfigDir, this.userConfigFilename);

    if (!existsSync(this.userConfigDir)) {
      mkdirSync(this.userConfigDir, { recursive: true });
    }

    this.currentConfig = this.loadConfigFromDisk();
    this.initializeWatcher();
  }

  private readonly userConfigFilename: string;

  private loadYamlFile<T>(filePath: string): T | undefined {
    if (!existsSync(filePath)) {
      return undefined;
    }

    const content = readFileSync(filePath, 'utf-8');
    if (!content.trim()) {
      return undefined;
    }

    return YAML.parse(content) as T;
  }

  private loadConfigFromDisk(): AppConfig {
    const defaultConfigRaw = this.loadYamlFile<Partial<AppConfig>>(this.defaultConfigPath) ?? {};
    const userConfigRaw = this.loadYamlFile<Partial<AppConfig>>(this.userConfigPath) ?? {};

    const merged = merge(defaultConfigRaw, userConfigRaw) as AppConfig;

    const parsed = AppConfigSchema.safeParse(merged);
    if (!parsed.success) {
      throw new Error(`Invalid configuration detected: ${parsed.error.message}`);
    }

    return parsed.data;
  }

  private initializeWatcher(): void {
    if (!existsSync(this.userConfigPath)) {
      return;
    }

    (async () => {
      try {
        this.fileWatcher = watch(this.userConfigPath, { persistent: false });
        for await (const event of this.fileWatcher) {
          if ((event as { eventType?: string }).eventType === 'change') {
            this.reload();
          }
        }
      } catch (error) {
        this.emit('error', error);
      }
    })().catch((error) => {
      this.emit('error', error);
    });
  }

  get config(): AppConfig {
    return this.currentConfig;
  }

  update(partial: Partial<AppConfig>): AppConfig {
    const merged = merge<AppConfig>(this.currentConfig, partial as AppConfig, {
      arrayMerge: (_destinationArray, sourceArray) => sourceArray
    });

    const parsed = AppConfigSchema.parse(merged);

    if (!existsSync(this.userConfigDir)) {
      mkdirSync(this.userConfigDir, { recursive: true });
    }

    writeFileSync(this.userConfigPath, YAML.stringify(parsed), 'utf-8');
    this.currentConfig = parsed;
    this.emit('updated', this.currentConfig);

    return this.currentConfig;
  }

  reload(): AppConfig {
    this.currentConfig = this.loadConfigFromDisk();
    this.emit('updated', this.currentConfig);
    return this.currentConfig;
  }
}
