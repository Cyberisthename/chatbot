import { EventEmitter } from 'eventemitter3';
import type { AppConfig, BrainResponse, ChatMessage } from '../types/index.js';
import { JarvisClient } from './JarvisClient.js';
import { LocalLLMClient } from './LocalLLMClient.js';
import { InfiniteCapacityClient } from './InfiniteCapacityClient.js';

export interface BrainRouterOptions {
  config: AppConfig;
}

export class BrainRouter extends EventEmitter {
  private readonly jarvis: JarvisClient;
  private readonly localLLM: LocalLLMClient;
  private readonly infiniteCapacity: InfiniteCapacityClient;
  private config: AppConfig;
  private jarvisStatus: 'online' | 'offline' | 'checking' = 'checking';
  private readonly conversationHistory: ChatMessage[] = [];

  constructor(options: BrainRouterOptions) {
    super();
    this.config = options.config;

    this.jarvis = new JarvisClient({
      baseURL: this.config.jarvis_api_url,
      timeout: this.config.advanced.request_timeout,
      retryAttempts: this.config.advanced.retry_attempts
    });

    this.localLLM = new LocalLLMClient({
      baseURL: this.config.local_llm_url,
      timeout: this.config.advanced.request_timeout,
      retryAttempts: 1
    });

    this.infiniteCapacity = new InfiniteCapacityClient({
      baseURL: this.config.infinite_capacity_url,
      timeout: this.config.advanced.request_timeout
    });

    this.startHealthChecks();
  }

  updateConfig(config: AppConfig): void {
    this.config = config;
  }

  async sendMessage(userMessage: string, options?: { forceBackend?: 'jarvis' | 'local' | 'infinite_capacity' }): Promise<BrainResponse> {
    const message: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: userMessage,
      timestamp: Date.now()
    };

    this.conversationHistory.push(message);
    if (this.conversationHistory.length > this.config.advanced.context_messages * 2) {
      const systemMessages = this.conversationHistory.filter((m) => m.role === 'system');
      const recentMessages = this.conversationHistory.slice(-this.config.advanced.context_messages * 2);
      this.conversationHistory.splice(0, this.conversationHistory.length, ...systemMessages, ...recentMessages);
    }

    const systemPrompt: ChatMessage = {
      id: 'system',
      role: 'system',
      content: this.config.personality.prompt_prefix,
      timestamp: Date.now()
    };

    const messagesToSend = [systemPrompt, ...this.conversationHistory.filter((m) => m.role !== 'system')];

    let response: BrainResponse;

    if (options?.forceBackend === 'infinite_capacity') {
      response = await this.infiniteCapacity.runHeavyReasoning(userMessage, { messages: messagesToSend });
    } else if (options?.forceBackend === 'local' || this.config.mode === 'local') {
      response = await this.localLLM.chat(messagesToSend);
    } else if (this.config.mode === 'jarvis' || (this.config.mode === 'hybrid' && this.jarvisStatus === 'online')) {
      response = await this.jarvis.chat(messagesToSend);

      if (response.error && this.config.mode === 'hybrid') {
        response = await this.localLLM.chat(messagesToSend);
      }
    } else {
      response = await this.localLLM.chat(messagesToSend);
    }

    if (response.reply && !response.error) {
      const assistantMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: response.reply,
        timestamp: Date.now(),
        tool_calls: response.tool_calls
      };
      this.conversationHistory.push(assistantMessage);
    }

    this.emit('message', response);
    return response;
  }

  clearHistory(): void {
    this.conversationHistory.splice(0, this.conversationHistory.length);
    this.emit('historyCleared');
  }

  getHistory(): ChatMessage[] {
    return [...this.conversationHistory];
  }

  getStatus(): {
    jarvis: 'online' | 'offline' | 'checking';
    mode: 'jarvis' | 'local' | 'hybrid';
  } {
    return {
      jarvis: this.jarvisStatus,
      mode: this.config.mode
    };
  }

  private startHealthChecks(): void {
    this.checkJarvisHealth();

    setInterval(() => {
      this.checkJarvisHealth();
    }, 30_000);
  }

  private checkJarvisHealth(): void {
    this.jarvisStatus = 'checking';

    this.jarvis
      .healthCheck()
      .then((health) => {
        this.jarvisStatus = health.status;
        this.emit('statusChange', this.getStatus());
      })
      .catch(() => {
        this.jarvisStatus = 'offline';
        this.emit('statusChange', this.getStatus());
      });
  }
}
