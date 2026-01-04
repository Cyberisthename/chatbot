/**
 * JARVIS-2v API Client
 * Type-safe client for communicating with the FastAPI backend
 */

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// ============================================================================
// Type Definitions
// ============================================================================

export interface HealthStatus {
  status: string;
  version: string;
  mode: string;
  adapters_count: number;
  artifacts_count: number;
  timestamp: number;
}

export interface Adapter {
  id: string;
  task_tags: string[];
  y_bits: number[];
  z_bits: number[];
  x_bits: number[];
  status: string;
  total_calls: number;
  success_count: number;
  last_used: number;
  created_at: number;
  parent_ids: string[];
  success_rate?: number;
}

export interface AdaptersListResponse {
  adapters: Adapter[];
  total: number;
}

export interface QuantumArtifact {
  artifact_id: string;
  experiment_type: string;
  created_at: number;
  linked_adapters: string[];
  statistics?: Record<string, any>;
}

export interface ArtifactsListResponse {
  artifacts: QuantumArtifact[];
  total: number;
}

export interface InferRequest {
  query: string;
  context?: Record<string, any>;
  features?: string[];
}

export interface InferResponse {
  response: string;
  adapters_used: string[];
  bit_patterns: {
    y_bits: number[];
    z_bits: number[];
    x_bits: number[];
  };
  processing_time: number;
}

export interface ExperimentRequest {
  experiment_type: string;
  iterations?: number;
  noise_level?: number;
  seed?: number;
  parameters?: Record<string, any>;
}

export interface ExperimentResponse {
  artifact_id: string;
  experiment_type: string;
  created_at: number;
  linked_adapters: string[];
  results_summary: Record<string, any>;
}

export interface CreateAdapterRequest {
  task_tags: string[];
  parameters?: Record<string, any>;
  parent_ids?: string[];
  y_bits?: number[];
  z_bits?: number[];
  x_bits?: number[];
}

export interface CreateAdapterResponse {
  adapter_id: string;
  status: string;
  adapter: Adapter;
}

export interface ConfigUpdateRequest {
  mode?: string;
  quantum_enabled?: boolean;
  settings?: Record<string, any>;
}

// ============================================================================
// API Client Class
// ============================================================================

class APIClient {
  private baseURL: string;

  constructor(baseURL: string = API_URL) {
    this.baseURL = baseURL;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseURL}${endpoint}`;
    
    const defaultHeaders = {
      'Content-Type': 'application/json',
    };

    try {
      const response = await fetch(url, {
        ...options,
        headers: {
          ...defaultHeaders,
          ...options.headers,
        },
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(
          errorData.detail || `HTTP ${response.status}: ${response.statusText}`
        );
      }

      return await response.json();
    } catch (error: any) {
      if (error.message.includes('fetch')) {
        throw new Error(`Cannot connect to backend at ${this.baseURL}`);
      }
      throw error;
    }
  }

  // ============================================================================
  // System Endpoints
  // ============================================================================

  async getHealth(): Promise<HealthStatus> {
    return this.request<HealthStatus>('/health');
  }

  async getConfig(): Promise<Record<string, any>> {
    return this.request<Record<string, any>>('/api/config');
  }

  async updateConfig(data: ConfigUpdateRequest): Promise<any> {
    return this.request('/api/config', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  // ============================================================================
  // Inference Endpoints
  // ============================================================================

  async infer(data: InferRequest): Promise<InferResponse> {
    return this.request<InferResponse>('/api/infer', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  // ============================================================================
  // Adapter Endpoints
  // ============================================================================

  async listAdapters(status?: string): Promise<AdaptersListResponse> {
    const query = status ? `?status=${status}` : '';
    return this.request<AdaptersListResponse>(`/api/adapters${query}`);
  }

  async getAdapter(adapterId: string): Promise<Adapter> {
    return this.request<Adapter>(`/api/adapters/${adapterId}`);
  }

  async createAdapter(data: CreateAdapterRequest): Promise<CreateAdapterResponse> {
    return this.request<CreateAdapterResponse>('/api/adapters', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  // ============================================================================
  // Quantum Endpoints
  // ============================================================================

  async runExperiment(data: ExperimentRequest): Promise<ExperimentResponse> {
    return this.request<ExperimentResponse>('/api/quantum/experiment', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async listArtifacts(): Promise<ArtifactsListResponse> {
    return this.request<ArtifactsListResponse>('/api/artifacts');
  }

  async getArtifact(artifactId: string): Promise<QuantumArtifact> {
    return this.request<QuantumArtifact>(`/api/artifacts/${artifactId}`);
  }
}

// ============================================================================
// Export singleton instance
// ============================================================================

const apiClient = new APIClient();
export default apiClient;
