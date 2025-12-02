import axios, { AxiosInstance } from 'axios';
import type { BrainResponse } from '../types/index.js';

export interface InfiniteCapacityClientOptions {
  baseURL: string;
  timeout?: number;
}

export interface InfiniteCapacityTask {
  task_description: string;
  payload: Record<string, unknown>;
}

export class InfiniteCapacityClient {
  private readonly client: AxiosInstance;

  constructor(options: InfiniteCapacityClientOptions) {
    this.client = axios.create({
      baseURL: options.baseURL,
      timeout: options.timeout ?? 120_000,
      headers: {
        'Content-Type': 'application/json'
      }
    });
  }

  async runHeavyReasoning(taskDescription: string, payload: Record<string, unknown>): Promise<BrainResponse> {
    try {
      const response = await this.client.post<{ result?: string }>('/heavy', {
        task_description: taskDescription,
        payload
      });

      return {
        reply: response.data.result ?? 'Task completed',
        backend_used: 'infinite_capacity'
      };
    } catch (error) {
      return {
        reply: '',
        error: axios.isAxiosError(error) ? error.message : String(error),
        backend_used: 'infinite_capacity'
      };
    }
  }

  async runBatchSearch(query: string, options?: Record<string, unknown>): Promise<BrainResponse> {
    return {
      reply: '[Infinite Capacity] Batch search not yet implemented',
      backend_used: 'infinite_capacity',
      error: 'Not implemented'
    };
  }

  async isAvailable(): Promise<boolean> {
    try {
      await this.client.get('/health', { timeout: 5000 });
      return true;
    } catch {
      return false;
    }
  }
}
