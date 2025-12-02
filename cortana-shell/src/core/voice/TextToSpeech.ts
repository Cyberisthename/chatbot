import { EventEmitter } from 'eventemitter3';
import { exec } from 'node:child_process';
import { promisify } from 'node:util';

const execAsync = promisify(exec);

export interface TextToSpeechOptions {
  mode: 'windows' | 'custom';
  voice?: string;
  rate?: number;
}

export class TextToSpeech extends EventEmitter {
  private mode: string;
  private voice: string;
  private rate: number;
  private isSpeaking = false;

  constructor(options: TextToSpeechOptions) {
    super();
    this.mode = options.mode;
    this.voice = options.voice ?? 'Microsoft David Desktop';
    this.rate = options.rate ?? 0;
  }

  async speak(text: string): Promise<void> {
    if (this.isSpeaking) {
      return;
    }

    this.isSpeaking = true;
    this.emit('started');

    try {
      if (this.mode === 'windows' && process.platform === 'win32') {
        await this.speakWindows(text);
      } else {
        console.log(`[TextToSpeech] Would speak: "${text}"`);
        await new Promise((resolve) => setTimeout(resolve, 1000));
      }
    } catch (error) {
      console.error('[TextToSpeech] Error:', error);
      this.emit('error', error);
    } finally {
      this.isSpeaking = false;
      this.emit('finished');
    }
  }

  private async speakWindows(text: string): Promise<void> {
    const escapedText = text.replace(/"/g, '""').replace(/'/g, "''");

    const psCommand = `
      Add-Type -AssemblyName System.Speech;
      $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer;
      $synth.SelectVoice('${this.voice}');
      $synth.Rate = ${this.rate};
      $synth.Speak('${escapedText}');
    `
      .split('\n')
      .map((line) => line.trim())
      .join(' ');

    await execAsync(`powershell -Command "${psCommand}"`, { timeout: 30_000 });
  }

  stop(): void {
    this.isSpeaking = false;
    this.emit('stopped');
  }

  getAvailableVoices(): Promise<string[]> {
    if (process.platform !== 'win32') {
      return Promise.resolve([]);
    }

    return (async () => {
      try {
        const { stdout } = await execAsync(
          `powershell -Command "Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).GetInstalledVoices() | ForEach-Object { $_.VoiceInfo.Name }"`
        );
        return stdout
          .split('\n')
          .map((v) => v.trim())
          .filter(Boolean);
      } catch {
        return [];
      }
    })();
  }
}
