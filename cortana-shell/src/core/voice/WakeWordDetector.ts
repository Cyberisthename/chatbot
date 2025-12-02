import { EventEmitter } from 'eventemitter3';

export interface WakeWordDetectorOptions {
  keyword: string;
  threshold?: number;
}

export class WakeWordDetector extends EventEmitter {
  private keyword: string;
  private threshold: number;
  private isRunning = false;

  constructor(options: WakeWordDetectorOptions) {
    super();
    this.keyword = options.keyword.toLowerCase();
    this.threshold = options.threshold ?? 0.5;
  }

  async start(): Promise<void> {
    if (this.isRunning) {
      return;
    }

    this.isRunning = true;
    this.emit('started');

    console.log(`[WakeWordDetector] Started detecting "${this.keyword}"`);
    console.log('[WakeWordDetector] NOTE: Wake word detection requires additional setup.');
    console.log('[WakeWordDetector] Options:');
    console.log('  1. Porcupine (https://github.com/Picovoice/porcupine)');
    console.log('  2. Snowboy (https://github.com/Kitt-AI/snowboy)');
    console.log('  3. Custom model via WebRTC VAD + keyword spotting');
    console.log('[WakeWordDetector] For now, use the UI mic button or hotkey to activate.');
  }

  stop(): void {
    if (!this.isRunning) {
      return;
    }

    this.isRunning = false;
    this.emit('stopped');
    console.log('[WakeWordDetector] Stopped');
  }

  simulateDetection(): void {
    if (this.isRunning) {
      this.emit('detected');
    }
  }

  getKeyword(): string {
    return this.keyword;
  }

  isActive(): boolean {
    return this.isRunning;
  }
}
