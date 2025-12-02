import { EventEmitter } from 'eventemitter3';
import { Readable } from 'node:stream';
import axios from 'axios';
import FormData from 'form-data';

export interface SpeechToTextOptions {
  mode: 'whisper_local' | 'whisper_http' | 'windows';
  endpoint?: string;
  modelPath?: string;
}

export class SpeechToText extends EventEmitter {
  private mode: string;
  private endpoint?: string;
  private modelPath?: string;
  private recordingStream?: Readable;
  private audioChunks: Buffer[] = [];
  private recorder?: typeof import('node-record-lpcm16');

  constructor(options: SpeechToTextOptions) {
    super();
    this.mode = options.mode;
    this.endpoint = options.endpoint;
    this.modelPath = options.modelPath ?? './models/ggml-base.en.bin';
  }

  async start(): Promise<void> {
    this.audioChunks = [];

    console.log(`[SpeechToText] Starting STT in ${this.mode} mode`);

    if (this.mode === 'windows') {
      console.log('[SpeechToText] Windows STT requires native integration');
      console.log('[SpeechToText] TODO: Implement Windows.Media.SpeechRecognition via C# bridge');

      setTimeout(() => {
        const mockTranscription = 'Hello Cortana, what can you do?';
        console.log(`[SpeechToText] Mock transcription: ${mockTranscription}`);
        this.emit('transcribed', mockTranscription);
      }, 2000);
      return;
    }

    try {
      const recordModule = await import('node-record-lpcm16');
      this.recorder = (recordModule.default ?? recordModule) as typeof import('node-record-lpcm16');

      const recordingStream = this.recorder.start({
        sampleRate: 16_000,
        channels: 1,
        audioType: 'wav',
        threshold: 0,
        silence: '2.0'
      }) as unknown as Readable;

      this.recordingStream = recordingStream;

      recordingStream.on('data', (chunk: Buffer) => {
        this.audioChunks.push(chunk);
      });

      recordingStream.on('end', async () => {
        await this.processAudio();
      });

      recordingStream.on('error', (error: Error) => {
        this.emit('error', error);
      });

      this.emit('started');
    } catch (error) {
      console.error('[SpeechToText] Error starting recording:', error);
      this.emit('error', error as Error);
    }
  }

  stop(): void {
    if (this.recorder) {
      try {
        this.recorder.stop();
      } catch (error) {
        console.error('[SpeechToText] Error stopping recorder:', error);
      }
    }

    if (this.recordingStream) {
      try {
        if (typeof (this.recordingStream as unknown as { destroy?: () => void }).destroy === 'function') {
          (this.recordingStream as unknown as { destroy: () => void }).destroy();
        }
      } catch (error) {
        console.error('[SpeechToText] Error destroying stream:', error);
      }

      this.recordingStream = undefined;
    }

    this.recorder = undefined;
  }

  private async processAudio(): Promise<void> {
    if (this.audioChunks.length === 0) {
      this.emit('error', new Error('No audio recorded'));
      return;
    }

    const audioBuffer = Buffer.concat(this.audioChunks);

    console.log(`[SpeechToText] Processing ${audioBuffer.length} bytes of audio`);

    if (this.mode === 'whisper_http' && this.endpoint) {
      await this.transcribeViaHTTP(audioBuffer);
    } else if (this.mode === 'whisper_local') {
      await this.transcribeLocal(audioBuffer);
    }
  }

  private async transcribeViaHTTP(audioBuffer: Buffer): Promise<void> {
    if (!this.endpoint) {
      this.emit('error', new Error('No HTTP endpoint configured'));
      return;
    }

    try {
      const formData = new FormData();
      formData.append('audio', audioBuffer, {
        filename: 'audio.wav',
        contentType: 'audio/wav'
      });

      const response = await axios.post<{ text?: string; transcription?: string }>(this.endpoint, formData, {
        headers: formData.getHeaders(),
        timeout: 30_000
      });

      const text = response.data.text ?? response.data.transcription ?? '';
      if (text) {
        this.emit('transcribed', text.trim());
      } else {
        this.emit('error', new Error('No transcription returned'));
      }
    } catch (error) {
      console.error('[SpeechToText] HTTP transcription error:', error);
      this.emit('error', error as Error);
    }
  }

  private async transcribeLocal(audioBuffer: Buffer): Promise<void> {
    console.log('[SpeechToText] Local Whisper transcription requires whisper.cpp');
    console.log('[SpeechToText] Options:');
    console.log('  1. Install whisper.cpp: https://github.com/ggerganov/whisper.cpp');
    console.log('  2. Download a model: ./models/ggml-base.en.bin');
    console.log(`  3. Run: ./whisper --model ${this.modelPath} --file audio.wav`);
    console.log('[SpeechToText] For now, using mock transcription');

    setTimeout(() => {
      const mockText = 'This is a mock transcription from local Whisper';
      this.emit('transcribed', mockText);
    }, 1000);
  }
}
