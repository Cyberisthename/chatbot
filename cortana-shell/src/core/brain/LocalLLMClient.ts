import axios, { AxiosInstance } from 'axios';
import type { BrainResponse, ChatMessage } from '../types/index.js';

export interface LocalLLMClientOptions {
  baseURL: string;
  model?: string;
  timeout?: number;
  retryAttempts?: number;
}

export class LocalLLMClient {
  private readonly client: AxiosInstance;
  private readonly model: string;
  private readonly retryAttempts: number;

  constructor(options: LocalLLMClientOptions) {
    this.model = options.model ?? 'llama2';
    this.retryAttempts = options.retryAttempts ?? 2;
    this.client = axios.create({
      baseURL: options.baseURL,
      timeout: options.timeout ?? 30_000,
      headers: {
        'Content-Type': 'application/json'
      }
    });
  }

  async chat(messages: ChatMessage[]): Promise<BrainResponse> {
    const prompt = messages
      .map((message) => `${message.role === 'assistant' ? 'Assistant' : 'User'}: ${message.content}`)
      .join('\n');

    const payload = {
      model: this.model,
      prompt,
      stream: false
    };

    for (let attempt = 1; attempt <= this.retryAttempts; attempt++) {
      try {
        const response = await this.client.post<{ response?: string; data?: { content?: string }[] }>('', payload);

        let reply = '';
        if (response.data.response) {
          reply = response.data.response.trim();
        } else if (Array.isArray(response.data.data)) {
          reply = response.data.data.map((item) => item.content ?? '').join('\n');
        }

        return {
          reply,
          backend_used: 'local'
        };
      } catch (error) {
        if (attempt === this.retryAttempts) {
          return {
            reply: '',
            error: axios.isAxiosError(error) ? error.message : String(error),
            backend_used: 'local'
          };
        }

        await new Promise((resolve) => setTimeout(resolve, 1000 * attempt));
      }
    }

    return {
      reply: '',
      error: 'Local LLM failed after retry attempts',
      backend_used: 'local'
    };
  }

  async isAvailable(): Promise<boolean> {
    try {
      await this.client.get('/');
      return true;
    } catch {
      return false;
    }
  }
}
