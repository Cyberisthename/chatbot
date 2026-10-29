import { EventEmitter } from 'node:events';
import fs from 'node:fs';
import path from 'node:path';

export interface VoiceManagerOptions {
  /** Directory containing the wake asset file. */
  modelDirectory: string;
  /** Filename of the wake asset (original resonance chime WAV). */
  defaultModel: string;
  enabled: boolean;
}

export class VoiceManager extends EventEmitter {
  private enabled: boolean;
  private readonly modelDirectory: string;
  private readonly defaultModel: string;
  private wakeAsset: Buffer | null = null;

  constructor(options: VoiceManagerOptions) {
    super();
    this.modelDirectory = options.modelDirectory;
    this.defaultModel = options.defaultModel;
    this.enabled = options.enabled;

    if (this.enabled) {
      this.loadWakeAsset();
    }
  }

  /**
   * Loads the configured wake asset bytes (jarvis_wake_chime.wav).
   * NOTE: this shell performs NO wake-word recognition — there is no mic
   * capture path; `wake` is emitted only via the manual triggerWake() call.
   * The loaded bytes are the owned 41.02 Hz resonance chime (brand/status
   * sound asset), replacing the legacy third-party heycortana_*.table models.
   */
  private loadWakeAsset(): void {
    try {
      const assetPath = path.join(this.modelDirectory, this.defaultModel);
      if (fs.existsSync(assetPath)) {
        this.wakeAsset = fs.readFileSync(assetPath);
        console.log('[VoiceManager] Wake asset loaded:', this.defaultModel);
      } else {
        console.warn('[VoiceManager] Wake asset not found:', assetPath);
      }
    } catch (error) {
      console.error('[VoiceManager] Failed to load wake asset:', error);
    }
  }

  setEnabled(enabled: boolean): void {
    this.enabled = enabled;
    if (enabled && !this.wakeAsset) {
      this.loadWakeAsset();
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
    this.wakeAsset = null;
  }
}
