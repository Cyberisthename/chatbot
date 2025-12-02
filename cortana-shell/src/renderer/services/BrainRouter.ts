interface BackendConfig {
  jarvis?: {
    enabled?: boolean;
    url?: string;
    timeout?: number;
  };
  ollama?: {
    enabled?: boolean;
    url?: string;
    model?: string;
  };
  infiniteCapacity?: {
    enabled?: boolean;
    url?: string;
  };
}

type Route = 'jarvis' | 'ollama' | 'infiniteCapacity';

export class BrainRouter {
  private config: BackendConfig | null = null;

  private async ensureConfig(): Promise<void> {
    if (!this.config) {
      const cfg = await window.cortana.config.get();
      this.config = cfg?.backend ?? {};
    }
  }

  async testConnection(): Promise<boolean> {
    try {
      await this.ensureConfig();
      const jarvisUrl = this.config?.jarvis?.url;
      if (!jarvisUrl) return false;

      const response = await fetch(jarvisUrl, {
        method: 'POST',
        body: JSON.stringify({
          message: 'ping',
          system: 'status',
        }),
        headers: {
          'Content-Type': 'application/json',
        },
      });
      return response.ok;
    } catch (error) {
      console.warn('BrainRouter connectivity test failed:', error);
      return false;
    }
  }

  async sendMessage(prompt: string): Promise<string> {
    await this.ensureConfig();
    const route = this.pickRoute();

    switch (route) {
      case 'jarvis':
        return this.sendToJarvis(prompt);
      case 'ollama':
        return this.sendToOllama(prompt);
      case 'infiniteCapacity':
        return this.sendToInfiniteCapacity(prompt);
      default:
        throw new Error('No backend configured');
    }
  }

  private pickRoute(): Route {
    if (this.config?.jarvis?.enabled && this.config.jarvis.url) {
      return 'jarvis';
    }
    if (this.config?.ollama?.enabled && this.config.ollama.url) {
      return 'ollama';
    }
    if (this.config?.infiniteCapacity?.enabled && this.config.infiniteCapacity.url) {
      return 'infiniteCapacity';
    }
    throw new Error('No available backend');
  }

  private async sendToJarvis(prompt: string): Promise<string> {
    const url = this.config?.jarvis?.url;
    if (!url) throw new Error('Jarvis URL not configured');

    const response = await fetch(url, {
      method: 'POST',
      body: JSON.stringify({ message: prompt }),
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      throw new Error(`Jarvis responded with ${response.status}`);
    }

    const payload = await response.json();
    return payload?.reply ?? payload?.response ?? 'I encountered an empty reply.';
  }

  private async sendToOllama(prompt: string): Promise<string> {
    const url = this.config?.ollama?.url;
    const model = this.config?.ollama?.model ?? 'llama2';
    if (!url) throw new Error('Ollama URL not configured');

    const response = await fetch(`${url}/api/generate`, {
      method: 'POST',
      body: JSON.stringify({ model, prompt }),
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      throw new Error(`Ollama responded with ${response.status}`);
    }

    const payload = await response.json();
    return payload?.response ?? 'Ollama did not return a response.';
  }

  private async sendToInfiniteCapacity(prompt: string): Promise<string> {
    const url = this.config?.infiniteCapacity?.url;
    if (!url) throw new Error('Infinite Capacity URL not configured');

    const response = await fetch(url, {
      method: 'POST',
      body: JSON.stringify({ prompt }),
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      throw new Error(`Infinite Capacity responded with ${response.status}`);
    }

    const payload = await response.json();
    return payload?.result ?? 'No response from Infinite Capacity backend.';
  }
}
