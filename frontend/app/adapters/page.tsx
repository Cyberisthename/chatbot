"use client";

import { useEffect, useState } from "react";
import { Cpu, Plus, Search, Filter } from "lucide-react";
import apiClient, { Adapter } from "@/lib/api-client";

export default function AdaptersPage() {
  const [adapters, setAdapters] = useState<Adapter[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<string>("");
  const [showCreateModal, setShowCreateModal] = useState(false);

  useEffect(() => {
    loadAdapters();
  }, [filter]);

  const loadAdapters = async () => {
    try {
      const data = await apiClient.listAdapters(filter || undefined);
      setAdapters(data.adapters);
      setError(null);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-4xl font-bold text-jarvis mb-2">Adapter Graph</h1>
          <p className="text-gray-400">Manage modular AI adapters</p>
        </div>
        <button
          onClick={() => setShowCreateModal(true)}
          className="flex items-center space-x-2 px-4 py-2 bg-jarvis-primary text-black rounded-lg hover:bg-jarvis-secondary transition glow-hover"
        >
          <Plus size={20} />
          <span>Create Adapter</span>
        </button>
      </div>

      {/* Filters */}
      <div className="glass rounded-lg p-4 flex items-center space-x-4">
        <Filter className="text-gray-400" size={20} />
        <select
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          className="bg-jarvis-darker border border-jarvis/30 rounded px-4 py-2 text-jarvis-primary focus:outline-none focus:border-jarvis"
        >
          <option value="">All Status</option>
          <option value="active">Active</option>
          <option value="frozen">Frozen</option>
          <option value="deprecated">Deprecated</option>
        </select>
        <div className="flex-1"></div>
        <span className="text-gray-400 text-sm">
          {adapters.length} adapter{adapters.length !== 1 ? "s" : ""}
        </span>
      </div>

      {/* Error */}
      {error && (
        <div className="glass border-red-500 rounded-lg p-4 text-red-400">
          Error: {error}
        </div>
      )}

      {/* Loading */}
      {loading && (
        <div className="text-center py-12">
          <div className="w-12 h-12 border-4 border-jarvis-primary border-t-transparent rounded-full animate-spin mx-auto"></div>
        </div>
      )}

      {/* Adapters Grid */}
      {!loading && !error && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {adapters.length === 0 ? (
            <div className="col-span-2 text-center py-12 text-gray-500">
              <Cpu size={48} className="mx-auto mb-4 opacity-50" />
              <p>No adapters found</p>
              <button
                onClick={() => setShowCreateModal(true)}
                className="mt-4 text-jarvis-primary hover:underline"
              >
                Create your first adapter
              </button>
            </div>
          ) : (
            adapters.map((adapter) => (
              <AdapterCard key={adapter.id} adapter={adapter} />
            ))
          )}
        </div>
      )}

      {/* Create Modal */}
      {showCreateModal && (
        <CreateAdapterModal
          onClose={() => setShowCreateModal(false)}
          onSuccess={() => {
            setShowCreateModal(false);
            loadAdapters();
          }}
        />
      )}
    </div>
  );
}

function AdapterCard({ adapter }: { adapter: Adapter }) {
  const statusColors = {
    active: "text-green-400 border-green-400/50",
    frozen: "text-blue-400 border-blue-400/50",
    deprecated: "text-gray-400 border-gray-400/50",
  };

  const statusClass =
    statusColors[adapter.status as keyof typeof statusColors] ||
    "text-gray-400 border-gray-400/50";

  return (
    <div className="glass rounded-lg p-6 hover:glow-hover transition">
      {/* Header */}
      <div className="flex justify-between items-start mb-4">
        <div>
          <h3 className="text-xl font-mono text-jarvis-primary">{adapter.id}</h3>
          <div className="flex items-center space-x-2 mt-2">
            {adapter.task_tags.map((tag) => (
              <span
                key={tag}
                className="px-2 py-1 bg-jarvis-primary/20 text-jarvis-primary text-xs rounded"
              >
                {tag}
              </span>
            ))}
          </div>
        </div>
        <span
          className={`px-3 py-1 text-sm rounded-full border ${statusClass}`}
        >
          {adapter.status}
        </span>
      </div>

      {/* Bit Patterns */}
      <div className="grid grid-cols-3 gap-2 mb-4">
        <BitPattern label="Y" bits={adapter.y_bits} color="text-blue-400" />
        <BitPattern label="Z" bits={adapter.z_bits} color="text-purple-400" />
        <BitPattern label="X" bits={adapter.x_bits} color="text-pink-400" />
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-3 gap-4 pt-4 border-t border-gray-800">
        <Metric label="Success Rate" value={`${adapter.success_rate?.toFixed(1)}%`} />
        <Metric label="Total Calls" value={adapter.total_calls} />
        <Metric
          label="Last Used"
          value={
            adapter.last_used
              ? new Date(adapter.last_used * 1000).toLocaleDateString()
              : "Never"
          }
        />
      </div>
    </div>
  );
}

function BitPattern({
  label,
  bits,
  color,
}: {
  label: string;
  bits: number[];
  color: string;
}) {
  const activeCount = bits.filter((b) => b === 1).length;
  return (
    <div className="text-center">
      <p className={`text-xs ${color} mb-1`}>{label}-bits</p>
      <p className="text-sm font-mono">
        {activeCount}/{bits.length}
      </p>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="text-center">
      <p className="text-xs text-gray-500 mb-1">{label}</p>
      <p className="text-sm font-bold text-jarvis-primary">{value}</p>
    </div>
  );
}

function CreateAdapterModal({
  onClose,
  onSuccess,
}: {
  onClose: () => void;
  onSuccess: () => void;
}) {
  const [tags, setTags] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const task_tags = tags
        .split(",")
        .map((t) => t.trim())
        .filter((t) => t);
      await apiClient.createAdapter({ task_tags });
      onSuccess();
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/80 flex items-center justify-center p-4 z-50">
      <div className="glass rounded-lg p-6 max-w-md w-full glow">
        <h2 className="text-2xl font-bold text-jarvis mb-4">Create Adapter</h2>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm text-gray-400 mb-2">
              Task Tags (comma-separated)
            </label>
            <input
              type="text"
              value={tags}
              onChange={(e) => setTags(e.target.value)}
              placeholder="coding, python, debugging"
              className="w-full px-4 py-2 bg-jarvis-darker border border-jarvis/30 rounded text-jarvis-primary focus:outline-none focus:border-jarvis"
              required
            />
          </div>

          {error && <p className="text-red-400 text-sm">{error}</p>}

          <div className="flex space-x-2">
            <button
              type="button"
              onClick={onClose}
              className="flex-1 px-4 py-2 bg-gray-700 text-white rounded hover:bg-gray-600 transition"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={loading}
              className="flex-1 px-4 py-2 bg-jarvis-primary text-black rounded hover:bg-jarvis-secondary transition disabled:opacity-50"
            >
              {loading ? "Creating..." : "Create"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
