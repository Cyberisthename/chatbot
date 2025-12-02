import { EventEmitter } from 'node:events';
import fs from 'node:fs';
import path from 'node:path';

export interface VoiceManagerOptions {
  modelDirectory: string;
  defaultModel: string;
  enabled: boolean;
}

export class VoiceManager extends EventEmitter {
  private enabled: boolean;
  private readonly modelDirectory: string;
  private readonly defaultModel: string;
  private wakeWordModel: Buffer | null = null;

  constructor(options: VoiceManagerOptions) {
    super();
    this.modelDirectory = options.modelDirectory;
    this.defaultModel = options.defaultModel;
    this.enabled = options.enabled;

    if (this.enabled) {
      this.loadWakeWordModel();
    }
  }

  private loadWakeWordModel(): void {
    try {
      const modelPath = path.join(this.modelDirectory, this.defaultModel);
      if (fs.existsSync(modelPath)) {
        this.wakeWordModel = fs.readFileSync(modelPath);
        console.log('[VoiceManager] Wake word model loaded:', this.defaultModel);
      } else {
        console.warn('[VoiceManager] Wake word model not found:', modelPath);
      }
    } catch (error) {
      console.error('[VoiceManager] Failed to load wake word model:', error);
    }
  }

  setEnabled(enabled: boolean): void {
    this.enabled = enabled;
    if (enabled && !this.wakeWordModel) {
      this.loadWakeWordModel();
    }
    console.log('[VoiceManager] Voice detection', enabled ? 'enabled' : 'disabled');
  }

  isEnabled(): boolean {
    return this.enabled;
  }

  onWake(callback: () => void): void {
    this.on('wake', callback);
  }

  triggerWake(): void {
    if (this.enabled) {
      this.emit('wake');
    }
  }

  dispose(): void {
    this.removeAllListeners();
    this.wakeWordModel = null;
  }
}
