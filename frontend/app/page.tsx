"use client";

import { useEffect, useState } from "react";
import { Activity, Cpu, Atom, Clock, TrendingUp } from "lucide-react";
import apiClient, { HealthStatus, Adapter, QuantumArtifact } from "@/lib/api-client";

export default function Dashboard() {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [adapters, setAdapters] = useState<Adapter[]>([]);
  const [artifacts, setArtifacts] = useState<QuantumArtifact[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadDashboardData();
    const interval = setInterval(loadDashboardData, 5000); // Refresh every 5s
    return () => clearInterval(interval);
  }, []);

  const loadDashboardData = async () => {
    try {
      const [healthData, adaptersData, artifactsData] = await Promise.all([
        apiClient.getHealth(),
        apiClient.listAdapters(),
        apiClient.listArtifacts(),
      ]);
      setHealth(healthData);
      setAdapters(adaptersData.adapters.slice(0, 5)); // Top 5
      setArtifacts(artifactsData.artifacts.slice(0, 5)); // Top 5
      setError(null);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-jarvis-primary border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-gray-400">Loading JARVIS-2v Dashboard...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="max-w-2xl mx-auto mt-8">
        <div className="glass border-red-500 rounded-lg p-6">
          <h2 className="text-xl font-bold text-red-500 mb-2">Connection Error</h2>
          <p className="text-gray-300">Unable to connect to backend: {error}</p>
          <p className="text-sm text-gray-500 mt-4">
            Make sure the backend is running on {process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}
          </p>
          <button
            onClick={loadDashboardData}
            className="mt-4 px-4 py-2 bg-jarvis-primary text-black rounded-lg hover:bg-jarvis-secondary transition"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-4xl font-bold text-jarvis mb-2">Dashboard</h1>
        <p className="text-gray-400">System overview and metrics</p>
      </div>

      {/* Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatusCard
          title="System Status"
          value={health?.status === "ok" ? "ONLINE" : "OFFLINE"}
          icon={Activity}
          color={health?.status === "ok" ? "text-green-400" : "text-red-400"}
        />
        <StatusCard
          title="Active Mode"
          value={health?.mode.replace("_", " ").toUpperCase() || "N/A"}
          icon={TrendingUp}
          color="text-blue-400"
        />
        <StatusCard
          title="Adapters"
          value={health?.adapters_count || 0}
          icon={Cpu}
          color="text-jarvis-primary"
        />
        <StatusCard
          title="Quantum Artifacts"
          value={health?.artifacts_count || 0}
          icon={Atom}
          color="text-jarvis-accent"
        />
      </div>

      {/* System Info */}
      <div className="glass rounded-lg p-6">
        <h2 className="text-2xl font-bold text-jarvis mb-4 flex items-center">
          <Activity className="mr-2" />
          System Information
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <InfoRow label="Version" value={health?.version || "N/A"} />
          <InfoRow label="Mode" value={health?.mode || "N/A"} />
          <InfoRow
            label="Last Updated"
            value={
              health?.timestamp
                ? new Date(health.timestamp * 1000).toLocaleTimeString()
                : "N/A"
            }
          />
          <InfoRow label="API Status" value={health?.status || "N/A"} />
        </div>
      </div>

      {/* Recent Adapters */}
      <div className="glass rounded-lg p-6">
        <h2 className="text-2xl font-bold text-jarvis mb-4 flex items-center">
          <Cpu className="mr-2" />
          Recent Adapters
        </h2>
        {adapters.length === 0 ? (
          <p className="text-gray-500">No adapters created yet</p>
        ) : (
          <div className="space-y-2">
            {adapters.map((adapter) => (
              <div
                key={adapter.id}
                className="bg-jarvis-darker p-4 rounded-lg border border-jarvis/30 hover:border-jarvis/60 transition"
              >
                <div className="flex justify-between items-start">
                  <div>
                    <p className="font-mono text-jarvis-primary">{adapter.id}</p>
                    <p className="text-sm text-gray-400 mt-1">
                      {adapter.task_tags.join(", ")}
                    </p>
                  </div>
                  <div className="text-right">
                    <p className="text-sm text-gray-500">
                      {adapter.success_rate?.toFixed(1)}% success
                    </p>
                    <p className="text-xs text-gray-600">
                      {adapter.total_calls} calls
                    </p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Recent Artifacts */}
      <div className="glass rounded-lg p-6">
        <h2 className="text-2xl font-bold text-jarvis mb-4 flex items-center">
          <Atom className="mr-2" />
          Recent Quantum Artifacts
        </h2>
        {artifacts.length === 0 ? (
          <p className="text-gray-500">No artifacts generated yet</p>
        ) : (
          <div className="space-y-2">
            {artifacts.map((artifact) => (
              <div
                key={artifact.artifact_id}
                className="bg-jarvis-darker p-4 rounded-lg border border-jarvis/30 hover:border-jarvis/60 transition"
              >
                <div className="flex justify-between items-start">
                  <div>
                    <p className="font-mono text-jarvis-accent">
                      {artifact.artifact_id}
                    </p>
                    <p className="text-sm text-gray-400 mt-1">
                      {artifact.experiment_type.replace(/_/g, " ")}
                    </p>
                  </div>
                  <div className="text-right">
                    <p className="text-xs text-gray-600 flex items-center">
                      <Clock size={12} className="mr-1" />
                      {new Date(artifact.created_at * 1000).toLocaleString()}
                    </p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function StatusCard({
  title,
  value,
  icon: Icon,
  color,
}: {
  title: string;
  value: string | number;
  icon: any;
  color: string;
}) {
  return (
    <div className="glass rounded-lg p-6 hover:glow-hover transition">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-gray-400 text-sm mb-1">{title}</p>
          <p className={`text-2xl font-bold ${color}`}>{value}</p>
        </div>
        <Icon className={`${color} opacity-50`} size={32} />
      </div>
    </div>
  );
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between py-2 border-b border-gray-800">
      <span className="text-gray-400">{label}:</span>
      <span className="text-jarvis-primary font-mono">{value}</span>
    </div>
  );
}
