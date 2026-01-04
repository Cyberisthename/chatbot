"use client";

import { useEffect, useState } from "react";
import { Settings as SettingsIcon, Save, RefreshCw } from "lucide-react";
import apiClient from "@/lib/api-client";

export default function SettingsPage() {
  const [config, setConfig] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [mode, setMode] = useState("standard");
  const [quantumEnabled, setQuantumEnabled] = useState(true);

  useEffect(() => {
    loadConfig();
  }, []);

  const loadConfig = async () => {
    try {
      const data = await apiClient.getConfig();
      setConfig(data);
      setMode(data.engine?.mode || "standard");
      setQuantumEnabled(data.quantum?.simulation_mode !== false);
    } catch (err) {
      console.error("Failed to load config:", err);
    } finally {
      setLoading(false);
    }
  };

  const saveSettings = async () => {
    setSaving(true);
    try {
      await apiClient.updateConfig({
        mode,
        quantum_enabled: quantumEnabled,
      });
      alert("Settings saved successfully!");
      await loadConfig();
    } catch (err: any) {
      alert(`Failed to save settings: ${err.message}`);
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="w-12 h-12 border-4 border-jarvis-primary border-t-transparent rounded-full animate-spin"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6 max-w-4xl">
      {/* Header */}
      <div>
        <h1 className="text-4xl font-bold text-jarvis mb-2">Settings</h1>
        <p className="text-gray-400">Configure JARVIS-2v system settings</p>
      </div>

      {/* Deployment Profile */}
      <div className="glass rounded-lg p-6">
        <h2 className="text-2xl font-bold text-jarvis mb-4 flex items-center">
          <SettingsIcon className="mr-2" />
          Deployment Profile
        </h2>
        <div className="space-y-4">
          <div>
            <label className="block text-sm text-gray-400 mb-2">
              Select Mode
            </label>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              {[
                {
                  id: "low_power",
                  name: "Low Power",
                  desc: "Minimal resources (FeatherEdge)",
                },
                {
                  id: "standard",
                  name: "Standard",
                  desc: "Balanced performance",
                },
                {
                  id: "jetson_orin",
                  name: "Jetson Orin",
                  desc: "GPU-accelerated edge",
                },
              ].map((modeOption) => (
                <button
                  key={modeOption.id}
                  onClick={() => setMode(modeOption.id)}
                  className={`p-4 rounded-lg border transition text-left ${
                    mode === modeOption.id
                      ? "border-jarvis-primary bg-jarvis-primary/20"
                      : "border-gray-700 bg-jarvis-darker hover:border-jarvis/50"
                  }`}
                >
                  <p className="font-bold text-jarvis-primary">
                    {modeOption.name}
                  </p>
                  <p className="text-xs text-gray-500 mt-1">{modeOption.desc}</p>
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Features */}
      <div className="glass rounded-lg p-6">
        <h2 className="text-2xl font-bold text-jarvis mb-4">Features</h2>
        <div className="space-y-4">
          <ToggleSetting
            label="Synthetic Quantum Simulation"
            description="Enable quantum experiment generation and artifacts"
            value={quantumEnabled}
            onChange={setQuantumEnabled}
          />
        </div>
      </div>

      {/* System Info */}
      {config && (
        <div className="glass rounded-lg p-6">
          <h2 className="text-2xl font-bold text-jarvis mb-4">System Information</h2>
          <div className="space-y-2">
            <InfoRow label="Version" value={config.engine?.version || "N/A"} />
            <InfoRow label="Current Mode" value={config.engine?.mode || "N/A"} />
            <InfoRow
              label="Adapter Storage"
              value={config.adapters?.storage_path || "N/A"}
            />
            <InfoRow
              label="Quantum Artifacts"
              value={config.quantum?.artifacts_path || "N/A"}
            />
            <InfoRow
              label="Y/Z/X Bits"
              value={`${config.bits?.y_bits || 0}/${config.bits?.z_bits || 0}/${
                config.bits?.x_bits || 0
              }`}
            />
            <InfoRow
              label="API Port"
              value={config.api?.port?.toString() || "N/A"}
            />
          </div>
        </div>
      )}

      {/* Actions */}
      <div className="flex space-x-3">
        <button
          onClick={saveSettings}
          disabled={saving}
          className="flex items-center space-x-2 px-6 py-3 bg-jarvis-primary text-black rounded-lg hover:bg-jarvis-secondary transition disabled:opacity-50 glow-hover"
        >
          <Save size={20} />
          <span>{saving ? "Saving..." : "Save Settings"}</span>
        </button>
        <button
          onClick={loadConfig}
          className="flex items-center space-x-2 px-6 py-3 bg-gray-700 text-white rounded-lg hover:bg-gray-600 transition"
        >
          <RefreshCw size={20} />
          <span>Reload</span>
        </button>
      </div>
    </div>
  );
}

function ToggleSetting({
  label,
  description,
  value,
  onChange,
}: {
  label: string;
  description: string;
  value: boolean;
  onChange: (value: boolean) => void;
}) {
  return (
    <div className="flex items-center justify-between p-4 bg-jarvis-darker rounded-lg">
      <div>
        <p className="font-bold text-jarvis-primary">{label}</p>
        <p className="text-sm text-gray-500 mt-1">{description}</p>
      </div>
      <button
        onClick={() => onChange(!value)}
        className={`relative inline-flex h-6 w-11 items-center rounded-full transition ${
          value ? "bg-jarvis-primary" : "bg-gray-600"
        }`}
      >
        <span
          className={`inline-block h-4 w-4 transform rounded-full bg-white transition ${
            value ? "translate-x-6" : "translate-x-1"
          }`}
        />
      </button>
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
