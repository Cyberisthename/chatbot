import axios, { AxiosInstance } from 'axios';
import type { ChatMessage, BrainResponse } from '../types/index.js';

export interface JarvisClientOptions {
  baseURL: string;
  timeout?: number;
  retryAttempts?: number;
}

export class JarvisClient {
  private readonly client: AxiosInstance;
  private readonly retryAttempts: number;
  private readonly chatPath: string;
  private readonly healthPath = '/api/health';

  constructor(options: JarvisClientOptions) {
    this.retryAttempts = options.retryAttempts ?? 3;

    const url = new URL(options.baseURL);
    this.chatPath = url.pathname === '/' ? '/api/chat' : url.pathname;

    this.client = axios.create({
      baseURL: `${url.protocol}//${url.host}`,
      timeout: options.timeout ?? 30_000,
      headers: {
        'Content-Type': 'application/json'
      }
    });
  }

  async chat(messages: ChatMessage[], context?: Record<string, unknown>): Promise<BrainResponse> {
    const payload = {
      messages: messages.map((msg) => ({
        role: msg.role,
        content: msg.content
      })),
      context,
      tools: []
    };

    for (let attempt = 1; attempt <= this.retryAttempts; attempt++) {
      try {
        const response = await this.client.post<{ text?: string; message?: { content: string }; tool_calls?: unknown[] }>(this.chatPath, payload);

        let replyText = '';
        if (response.data.text) {
          replyText = response.data.text;
        } else if (response.data.message && typeof response.data.message.content === 'string') {
          replyText = response.data.message.content;
        }

        return {
          reply: replyText,
          tool_calls: response.data.tool_calls as BrainResponse['tool_calls'],
          backend_used: 'jarvis'
        };
      } catch (error) {
        if (attempt === this.retryAttempts) {
          if (axios.isAxiosError(error)) {
            return {
              reply: '',
              error: error.response?.data?.error ?? error.message,
              backend_used: 'jarvis'
            };
          }

          return {
            reply: '',
            error: String(error),
            backend_used: 'jarvis'
          };
        }

        await new Promise((resolve) => setTimeout(resolve, 1000 * attempt));
      }
    }

    return {
      reply: '',
      error: 'Failed after maximum retry attempts',
      backend_used: 'jarvis'
    };
  }

  async healthCheck(): Promise<{ status: 'online' | 'offline'; latency?: number }> {
    const startTime = Date.now();

    try {
      await this.client.get(this.healthPath, { timeout: 5000 });
      return {
        status: 'online',
        latency: Date.now() - startTime
      };
    } catch {
      return {
        status: 'offline'
      };
    }
  }
}
