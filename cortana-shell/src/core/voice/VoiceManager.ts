import { EventEmitter } from 'eventemitter3';
import type { VoiceConfig, VoiceState } from '../types/index.js';
import { WakeWordDetector } from './WakeWordDetector.js';
import { SpeechToText } from './SpeechToText.js';
import { TextToSpeech } from './TextToSpeech.js';

export interface VoiceManagerOptions {
  config: VoiceConfig;
}

export class VoiceManager extends EventEmitter {
  private config: VoiceConfig;
  private wakeWord?: WakeWordDetector;
  private stt?: SpeechToText;
  private tts?: TextToSpeech;
  private currentState: VoiceState;

  constructor(options: VoiceManagerOptions) {
    super();
    this.config = options.config;
    this.currentState = {
      listening: false,
      thinking: false,
      speaking: false
    };

    this.initialize();
  }

  private initialize(): void {
    this.disposeServices();

    if (this.config.wake_word_enabled) {
      this.wakeWord = new WakeWordDetector({
        keyword: this.config.wake_word
      });

      this.wakeWord.on('detected', () => {
        this.emit('wakeWord');
        void this.startListening();
      });
    }

    this.stt = new SpeechToText({
      mode: this.config.stt_mode,
      endpoint: this.config.stt_endpoint
    });

    this.stt.on('transcribed', (text: string) => {
      this.stopListening();
      this.emit('speechRecognized', text);
    });

    this.stt.on('error', (error: Error) => {
      this.stopListening();
      this.emit('error', error);
    });

    this.tts = new TextToSpeech({
      mode: this.config.tts_mode,
      voice: this.config.tts_voice
    });

    this.tts.on('started', () => {
      this.setState({ speaking: true });
      this.emit('speakingStarted');
    });

    this.tts.on('finished', () => {
      this.setState({ speaking: false });
      this.emit('speakingFinished');
    });
  }

  private disposeServices(): void {
    if (this.wakeWord) {
      this.wakeWord.removeAllListeners();
      this.wakeWord.stop();
      this.wakeWord = undefined;
    }

    if (this.stt) {
      this.stt.removeAllListeners();
      this.stt.stop();
      this.stt = undefined;
    }

    if (this.tts) {
      this.tts.removeAllListeners();
      this.tts.stop();
      this.tts = undefined;
    }
  }

  updateConfig(config: VoiceConfig): void {
    this.config = config;
    this.initialize();
  }

  async startWakeWordDetection(): Promise<void> {
    if (!this.wakeWord) {
      throw new Error('Wake word detection is not enabled');
    }

    await this.wakeWord.start();
    this.emit('wakeWordDetectionStarted');
  }

  stopWakeWordDetection(): void {
    if (this.wakeWord) {
      this.wakeWord.stop();
      this.emit('wakeWordDetectionStopped');
    }
  }

  async startListening(): Promise<void> {
    if (!this.stt) {
      throw new Error('Speech-to-text is not configured');
    }

    this.setState({ listening: true });
    this.emit('listeningStarted');
    await this.stt.start();
  }

  stopListening(): void {
    if (this.stt) {
      this.stt.stop();
    }
    this.setState({ listening: false });
    this.emit('listeningStopped');
  }

  async speak(text: string): Promise<void> {
    if (!this.tts) {
      throw new Error('Text-to-speech is not configured');
    }

    await this.tts.speak(text);
  }

  setThinking(thinking: boolean): void {
    this.setState({ thinking });
    this.emit(thinking ? 'thinkingStarted' : 'thinkingStopped');
  }

  getState(): VoiceState {
    return { ...this.currentState };
  }

  private setState(partial: Partial<VoiceState>): void {
    this.currentState = {
      ...this.currentState,
      ...partial
    };
    this.emit('stateChanged', this.currentState);
  }
}
