"use client";

import { useEffect, useState } from "react";
import { Atom, Play, Beaker, Clock } from "lucide-react";
import apiClient, { QuantumArtifact } from "@/lib/api-client";

const experimentTypes = [
  { id: "interference_experiment", name: "Interference Experiment", icon: "🌊" },
  { id: "bell_pair_simulation", name: "Bell Pair Simulation", icon: "🔗" },
  { id: "chsh_test", name: "CHSH Test", icon: "📊" },
  { id: "noise_field_scan", name: "Noise Field Scan", icon: "📡" },
];

export default function QuantumLabPage() {
  const [artifacts, setArtifacts] = useState<QuantumArtifact[]>([]);
  const [loading, setLoading] = useState(true);
  const [runningExperiment, setRunningExperiment] = useState(false);
  const [selectedType, setSelectedType] = useState(experimentTypes[0].id);
  const [iterations, setIterations] = useState(1000);
  const [noiseLevel, setNoiseLevel] = useState(0.1);

  useEffect(() => {
    loadArtifacts();
  }, []);

  const loadArtifacts = async () => {
    try {
      const data = await apiClient.listArtifacts();
      setArtifacts(data.artifacts);
    } catch (err) {
      console.error("Failed to load artifacts:", err);
    } finally {
      setLoading(false);
    }
  };

  const runExperiment = async () => {
    setRunningExperiment(true);
    try {
      await apiClient.runExperiment({
        experiment_type: selectedType,
        iterations,
        noise_level: noiseLevel,
      });
      await loadArtifacts();
    } catch (err: any) {
      alert(`Experiment failed: ${err.message}`);
    } finally {
      setRunningExperiment(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-4xl font-bold text-jarvis mb-2">Quantum Lab</h1>
        <p className="text-gray-400">
          Run synthetic quantum experiments and generate artifacts
        </p>
      </div>

      {/* Experiment Config */}
      <div className="glass rounded-lg p-6">
        <h2 className="text-2xl font-bold text-jarvis mb-4 flex items-center">
          <Beaker className="mr-2" />
          Run Experiment
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Experiment Type */}
          <div>
            <label className="block text-sm text-gray-400 mb-2">
              Experiment Type
            </label>
            <div className="space-y-2">
              {experimentTypes.map((exp) => (
                <button
                  key={exp.id}
                  onClick={() => setSelectedType(exp.id)}
                  className={`w-full text-left px-4 py-3 rounded-lg border transition ${
                    selectedType === exp.id
                      ? "border-jarvis-primary bg-jarvis-primary/20 text-jarvis-primary"
                      : "border-gray-700 bg-jarvis-darker text-gray-400 hover:border-jarvis/50"
                  }`}
                >
                  <span className="mr-2">{exp.icon}</span>
                  {exp.name}
                </button>
              ))}
            </div>
          </div>

          {/* Parameters */}
          <div className="space-y-4">
            <div>
              <label className="block text-sm text-gray-400 mb-2">
                Iterations: {iterations}
              </label>
              <input
                type="range"
                min="100"
                max="5000"
                step="100"
                value={iterations}
                onChange={(e) => setIterations(Number(e.target.value))}
                className="w-full"
              />
            </div>

            <div>
              <label className="block text-sm text-gray-400 mb-2">
                Noise Level: {noiseLevel.toFixed(2)}
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={noiseLevel}
                onChange={(e) => setNoiseLevel(Number(e.target.value))}
                className="w-full"
              />
            </div>

            <button
              onClick={runExperiment}
              disabled={runningExperiment}
              className="w-full flex items-center justify-center space-x-2 px-6 py-3 bg-jarvis-primary text-black rounded-lg hover:bg-jarvis-secondary transition disabled:opacity-50 glow-hover"
            >
              {runningExperiment ? (
                <>
                  <div className="w-5 h-5 border-2 border-black border-t-transparent rounded-full animate-spin"></div>
                  <span>Running Experiment...</span>
                </>
              ) : (
                <>
                  <Play size={20} />
                  <span>Run Experiment</span>
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Artifacts */}
      <div className="glass rounded-lg p-6">
        <h2 className="text-2xl font-bold text-jarvis mb-4 flex items-center">
          <Atom className="mr-2" />
          Quantum Artifacts ({artifacts.length})
        </h2>

        {loading ? (
          <div className="text-center py-8">
            <div className="w-12 h-12 border-4 border-jarvis-primary border-t-transparent rounded-full animate-spin mx-auto"></div>
          </div>
        ) : artifacts.length === 0 ? (
          <div className="text-center py-8 text-gray-500">
            <Atom size={48} className="mx-auto mb-4 opacity-50" />
            <p>No artifacts yet. Run an experiment to generate one!</p>
          </div>
        ) : (
          <div className="space-y-3">
            {artifacts.map((artifact) => (
              <ArtifactCard key={artifact.artifact_id} artifact={artifact} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function ArtifactCard({ artifact }: { artifact: QuantumArtifact }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div
      className="bg-jarvis-darker rounded-lg border border-jarvis/30 hover:border-jarvis/60 transition cursor-pointer"
      onClick={() => setExpanded(!expanded)}
    >
      <div className="p-4">
        <div className="flex justify-between items-start">
          <div>
            <p className="font-mono text-jarvis-accent text-lg">
              {artifact.artifact_id}
            </p>
            <p className="text-sm text-gray-400 mt-1">
              {artifact.experiment_type.replace(/_/g, " ")}
            </p>
          </div>
          <div className="text-right">
            <p className="text-xs text-gray-500 flex items-center">
              <Clock size={12} className="mr-1" />
              {new Date(artifact.created_at * 1000).toLocaleString()}
            </p>
            <p className="text-xs text-gray-600 mt-1">
              {artifact.linked_adapters.length} linked adapter(s)
            </p>
          </div>
        </div>

        {/* Statistics */}
        {expanded && artifact.statistics && (
          <div className="mt-4 pt-4 border-t border-gray-800">
            <h4 className="text-sm font-bold text-jarvis-primary mb-2">
              Statistics
            </h4>
            <div className="grid grid-cols-2 gap-2">
              {Object.entries(artifact.statistics).map(([key, value]) => (
                <div key={key} className="text-sm">
                  <span className="text-gray-500">
                    {key.replace(/_/g, " ")}:{" "}
                  </span>
                  <span className="text-jarvis-primary font-mono">
                    {typeof value === "number" ? value.toFixed(4) : String(value)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
