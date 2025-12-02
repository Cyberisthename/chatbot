declare module 'node-record-lpcm16' {
  import { Readable } from 'node:stream';

  export interface RecordOptions {
    sampleRate?: number;
    channels?: number;
    audioType?: string;
    threshold?: number;
    silence?: string;
  }

  export function start(options?: RecordOptions): Readable;
  export function stop(): void;
}
