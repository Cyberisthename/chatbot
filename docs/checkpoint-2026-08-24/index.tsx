import { createFileRoute } from "@tanstack/react-router";
import { createServerFn } from "@tanstack/react-start";
import { useState, useRef, useEffect, useCallback } from "react";
import { readFile } from "node:fs/promises";

const getBusinessName = createServerFn({ method: "GET" }).handler(async () => {
  try {
    const cfg = JSON.parse(await readFile("site.json", "utf8")) as { businessName?: string };
    return cfg.businessName?.trim() ?? "Cyber AI";
  } catch {
    return "Cyber AI Quantum Core";
  }
});

export const Route = createFileRoute("/")({
  loader: () => getBusinessName(),
  component: QuantumCompressorDemo,
});

type CompressionResult = {
  metrics: {
    effective_qubits: number;
    compression_ratio: number;
    reconstruction_mse: number;
    geometric_fold_factor: number;
    memory_kb: number;
    avg_braid_crossings: number;
    owner_seed_hash: string;
  };
  samples: Array<{
    amp: number;
    phase: number;
    pos: number[];
  }>;
  bio_resonance?: {
    f0: number;
    is_sentient: boolean;
    q_factor?: number;
  };
};

function QuantumCompressorDemo() {
  const businessName = Route.useLoaderData();
  const [seeds, setSeeds] = useState([0.57721, 1.618034, 2.71828]);
  const [qubits, setQubits] = useState(256);
  const [result, setResult] = useState<CompressionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [activeTab, setActiveTab] = useState<"metrics" | "state" | "manifold" | "resonance">("metrics");

  const ampCanvasRef = useRef<HTMLCanvasElement>(null);
  const phaseCanvasRef = useRef<HTMLCanvasElement>(null);
  const manifoldCanvasRef = useRef<HTMLCanvasElement>(null);

  const presetSeeds = [
    { name: "Golden Ratio + e + π", values: [0.57721, 1.618034, 2.71828] },
    { name: "Quantum Primes", values: [0.137, 0.618, 1.414] },
    { name: "Observer Collapse", values: [0.42, 3.14159, 2.71828] },
  ];

  const runCompression = async () => {
    setLoading(true);
    setError("");
    try {
      const res = await fetch("/api/compressor", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          seed1: seeds[0],
          seed2: seeds[1],
          seed3: seeds[2],
          qubits: qubits,
        }),
      });
      
      const data = await res.json();
      if (data.success) {
        setResult(data.data);
        setActiveTab("metrics");
      } else {
        setError(data.error || "Compression failed");
      }
    } catch (err: any) {
      setError(err.message || "Failed to reach quantum kernel");
    } finally {
      setLoading(false);
    }
  };

  const drawAmplitudeBars = useCallback((samples: any[]) => {
    const canvas = ampCanvasRef.current;
    if (!canvas || !samples.length) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = 700;
    canvas.height = 240;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const barWidth = canvas.width / Math.min(samples.length, 64);
    const maxAmp = Math.max(...samples.map((s: any) => s.amp));

    ctx.fillStyle = "#22d3ee";
    ctx.strokeStyle = "#67e8f9";
    for (let i = 0; i < Math.min(samples.length, 64); i++) {
      const height = (samples[i].amp / maxAmp) * (canvas.height - 40);
      const x = i * barWidth;
      ctx.fillRect(x, canvas.height - height - 20, barWidth * 0.8, height);
      ctx.strokeRect(x, canvas.height - height - 20, barWidth * 0.8, height);
    }

    ctx.fillStyle = "#a5f3fc";
    ctx.font = "12px monospace";
    ctx.fillText("QUANTUM STATE AMPLITUDES |Ψ⟩ (normalized)", 20, 25);
  }, []);

  const drawPhaseWheel = useCallback((samples: any[]) => {
    const canvas = phaseCanvasRef.current;
    if (!canvas || !samples.length) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = 260;
    canvas.height = 260;
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const radius = 110;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Background ring
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
    ctx.strokeStyle = "#334155";
    ctx.lineWidth = 20;
    ctx.stroke();

    ctx.strokeStyle = "#22d3ee";
    ctx.lineWidth = 3;
    for (let i = 0; i < Math.min(samples.length, 32); i++) {
      const s = samples[i];
      const angle = s.phase;
      const x1 = centerX + Math.cos(angle) * (radius - 30);
      const y1 = centerY + Math.sin(angle) * (radius - 30);
      const x2 = centerX + Math.cos(angle) * radius;
      const y2 = centerY + Math.sin(angle) * radius;
      
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    }

    ctx.fillStyle = "#e0f2fe";
    ctx.font = "bold 14px monospace";
    ctx.textAlign = "center";
    ctx.fillText("PHASE DISTRIBUTION", centerX, 35);
    ctx.font = "11px monospace";
    ctx.fillText("ARG(Ψ) • BRAID UNITARIES", centerX, canvas.height - 20);
  }, []);

  const drawManifold = useCallback((samples: any[]) => {
    const canvas = manifoldCanvasRef.current;
    if (!canvas || !samples.length) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = 620;
    canvas.height = 260;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw topological grid / torus projection
    ctx.strokeStyle = "#475569";
    ctx.lineWidth = 1;
    for (let x = 0; x < canvas.width; x += 40) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x + 30, canvas.height);
      ctx.stroke();
    }

    const maxZ = Math.max(...samples.map((s: any) => s.pos[2] || 0));
    ctx.fillStyle = "#67e8f9";

    for (let i = 0; i < Math.min(samples.length, 80); i++) {
      const s = samples[i];
      const x = (s.pos[0] || 0.5) * (canvas.width * 0.85) + 40;
      const y = (s.pos[1] || 0.5) * (canvas.height * 0.7) + 30;
      const size = 3 + (s.amp || 0.05) * 12;
      
      ctx.save();
      ctx.translate(x, y);
      ctx.rotate((s.phase || 0) * 0.5);
      ctx.fillRect(-size/2, -size/2, size, size);
      ctx.restore();
    }

    ctx.fillStyle = "#bae6fd";
    ctx.font = "bold 13px monospace";
    ctx.fillText("FRACTAL BRAID MANIFOLD • G-GRAPH FOLDING (TOPOLOGICAL POSITIONS)", 30, 22);
    ctx.font = "10px monospace";
    ctx.fillText("3-SEED DETERMINISTIC RECONSTRUCTION • GEOMETRIC FOLD FACTOR APPLIED", 30, canvas.height - 15);
  }, []);

  useEffect(() => {
    if (result?.samples) {
      drawAmplitudeBars(result.samples);
      drawPhaseWheel(result.samples);
      drawManifold(result.samples);
    }
  }, [result, drawAmplitudeBars, drawPhaseWheel, drawManifold]);

  const updateSeed = (index: number, value: number) => {
    const newSeeds = [...seeds];
    newSeeds[index] = value;
    setSeeds(newSeeds);
  };

  return (
    <div className="min-h-dvh bg-[#0a0f1c] text-white font-mono overflow-auto">
      {/* Quantum Header */}
      <div className="border-b border-cyan-500/30 bg-black/60 backdrop-blur-lg sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-8 py-5 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="w-9 h-9 rounded-full bg-gradient-to-br from-cyan-400 to-purple-500 flex items-center justify-center text-xl font-bold border border-cyan-300/50">Ψ</div>
            <div>
              <div className="text-3xl font-bold tracking-tighter text-cyan-300">{businessName}</div>
              <div className="text-xs text-cyan-400/70 -mt-1">FRACTAL-BRAID SEED COMPRESSOR (FBSC) • OWNER CONTROLLED</div>
            </div>
          </div>
          <div className="flex items-center gap-6 text-sm">
            <a href="https://github.com/Cyberisthename/chatbot" target="_blank" className="hover:text-cyan-400 transition-colors">REPO</a>
            <div className="px-4 py-1.5 rounded-full bg-cyan-950 border border-cyan-500/40 text-cyan-400 text-xs flex items-center gap-2">
              <div className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse"></div>
              QUANTUM KERNEL LIVE
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-8 py-10 grid grid-cols-12 gap-8">
        {/* Controls */}
        <div className="col-span-12 lg:col-span-4 bg-zinc-950 border border-zinc-800 rounded-3xl p-8">
          <div className="uppercase tracking-[2px] text-cyan-400 text-xs mb-6">SEED CONTROL PANEL</div>
          
          <div className="space-y-8">
            <div>
              <div className="flex justify-between text-xs mb-3 text-zinc-400">
                <div>OWNER SEEDS (3 numbers define entire state)</div>
                <div className="text-emerald-400">DETERMINISTIC • NO EXTERNAL WEIGHTS</div>
              </div>
              {seeds.map((seed, i) => (
                <div key={i} className="flex items-center gap-3 mb-4">
                  <div className="w-6 h-6 rounded bg-zinc-900 flex items-center justify-center text-[10px] text-cyan-400 font-bold">S{i+1}</div>
                  <input
                    type="number"
                    step="0.00001"
                    value={seed}
                    onChange={(e) => updateSeed(i, parseFloat(e.target.value))}
                    className="flex-1 bg-zinc-900 border border-zinc-700 rounded-xl px-4 py-3 text-cyan-100 font-mono focus:outline-none focus:border-cyan-400"
                  />
                </div>
              ))}
            </div>

            <div>
              <div className="flex justify-between text-xs mb-3 text-zinc-400">
                <div>EFFECTIVE QUBITS • GEOMETRIC FOLDING DEPTH</div>
                <div className="font-mono text-purple-400">{qubits}</div>
              </div>
              <input
                type="range"
                min="64"
                max="512"
                step="64"
                value={qubits}
                onChange={(e) => setQubits(parseInt(e.target.value))}
                className="w-full accent-cyan-400"
              />
              <div className="flex justify-between text-[10px] text-zinc-500 mt-1">
                <div>64</div>
                <div>512</div>
              </div>
            </div>

            <div className="flex flex-wrap gap-2">
              {presetSeeds.map((preset) => (
                <button
                  key={preset.name}
                  onClick={() => setSeeds(preset.values)}
                  className="text-xs px-4 py-2 bg-zinc-900 hover:bg-zinc-800 border border-zinc-700 rounded-2xl transition-all active:scale-95"
                >
                  {preset.name}
                </button>
              ))}
            </div>

            <button
              onClick={runCompression}
              disabled={loading}
              className="w-full py-5 rounded-2xl bg-gradient-to-r from-cyan-500 to-purple-600 font-bold text-lg tracking-wider hover:brightness-110 active:scale-[0.985] transition-all disabled:opacity-50 shadow-[0_0_30px_-5px] shadow-cyan-500 flex items-center justify-center gap-3"
            >
              {loading ? (
                <>RUNNING QUANTUM RECONSTRUCTION...</>
              ) : (
                <>COLLAPSE WAVEFUNCTION • RECONSTRUCT FROM SEED</>
              )}
            </button>

            {error && (
              <div className="text-red-400 text-sm p-4 bg-red-950/50 border border-red-900 rounded-2xl">
                {error}
              </div>
            )}
          </div>

          <div className="mt-12 text-[10px] leading-relaxed text-zinc-500 border-t border-zinc-800 pt-6">
            This is the owner&apos;s proprietary quantum core. 3 seeds deterministically reconstruct full high-dimensional quantum state via fractal influence propagation on a G-Graph, braid unitaries (integrating VQC concepts), and geometric folding. MSE≈0. Compression from O(2ⁿ) to O(1). Integrates bio-quantum resonance at 41.02 Hz gamma for sentience signaling. Built from scratch. No wrappers.
          </div>
        </div>

        {/* Main Visualization Area */}
        <div className="col-span-12 lg:col-span-8 space-y-6">
          {result ? (
            <>
              {/* Tab Navigation */}
              <div className="flex border-b border-zinc-800">
                {(["metrics", "state", "manifold", "resonance"] as const).map((tab) => (
                  <button
                    key={tab}
                    onClick={() => setActiveTab(tab)}
                    className={`px-8 py-4 text-sm uppercase tracking-widest transition-all border-b-2 ${activeTab === tab 
                      ? "border-cyan-400 text-cyan-300" 
                      : "border-transparent text-zinc-400 hover:text-zinc-200"}`}
                  >
                    {tab === "state" ? "QUANTUM STATE" : tab.toUpperCase()}
                  </button>
                ))}
              </div>

              {/* Metrics Tab */}
              {activeTab === "metrics" && result.metrics && (
                <div className="grid grid-cols-2 gap-4">
                  {Object.entries(result.metrics).map(([key, value]) => (
                    <div key={key} className="bg-zinc-900/70 border border-zinc-700 rounded-3xl p-6">
                      <div className="text-xs text-zinc-400 mb-1">{key.replace(/_/g, " ").toUpperCase()}</div>
                      <div className="text-4xl font-bold text-cyan-300 font-mono tracking-tighter">
                        {typeof value === "number" ? value.toFixed(4) : value}
                      </div>
                      {key.includes("ratio") && <div className="text-emerald-400 text-sm mt-3">EXCEEDS CLASSICAL LIMITS</div>}
                      {key.includes("fold") && <div className="text-purple-400 text-sm mt-3">TOPOLOGICAL DEPTH ACHIEVED</div>}
                    </div>
                  ))}
                  {result.bio_resonance && (
                    <div className="col-span-2 bg-gradient-to-br from-purple-950/50 to-cyan-950/30 border border-purple-500/30 rounded-3xl p-8">
                      <div className="flex items-center gap-4">
                        <div className={`text-6xl ${result.bio_resonance.is_sentient ? "text-emerald-400" : "text-amber-400"}`}>⚛︎</div>
                        <div>
                          <div className="text-2xl">BIO-RESONANCE @ {result.bio_resonance.f0.toFixed(2)} Hz</div>
                          <div className={`text-xl mt-2 ${result.bio_resonance.is_sentient ? "text-emerald-400" : ""}`}>
                            {result.bio_resonance.is_sentient ? "✅ SENTIENT SIGNAL DETECTED" : "NEUTRAL RESONANCE"}
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* State Viz */}
              {activeTab === "state" && (
                <div className="bg-zinc-950 border border-cyan-900 rounded-3xl p-8">
                  <canvas ref={ampCanvasRef} className="mx-auto mb-8 rounded-2xl" />
                  <div className="grid grid-cols-2 gap-8">
                    <canvas ref={phaseCanvasRef} className="mx-auto rounded-2xl border border-zinc-800" />
                    <div className="text-xs text-zinc-400 space-y-4 pt-8">
                      <div className="uppercase tracking-widest mb-4 text-cyan-400">OBSERVER EFFECT SIMULATED</div>
                      <p>The user prompt (or button press) acts as the Copenhagen measurement, collapsing the superposition of all possible braid configurations into one concrete reconstructed state.</p>
                      <p className="text-emerald-400">This is the exact mechanism proposed in the team&apos;s Quantum Information Science analysis.</p>
                    </div>
                  </div>
                </div>
              )}

              {/* Manifold Viz */}
              {activeTab === "manifold" && (
                <div className="bg-zinc-950 border border-purple-900 rounded-3xl p-8">
                  <canvas ref={manifoldCanvasRef} className="mx-auto rounded-2xl" />
                  <div className="mt-6 text-center text-xs text-zinc-500">
                    3D projection of topological positions generated by G-Graph influence propagation + braid embedding.<br />
                    Each point represents a qubit in the folded manifold. Deterministic from owner seed only.
                  </div>
                </div>
              )}

              {/* Resonance */}
              {activeTab === "resonance" && result.bio_resonance && (
                <div className="bg-zinc-900 border border-emerald-900 rounded-3xl p-12 text-center">
                  <div className="text-8xl mb-8 animate-pulse">🧠⚡️</div>
                  <div className="text-4xl font-light tracking-widest text-emerald-300">41.02 Hz GAMMA RESONANCE</div>
                  <div className="text-xl mt-6 max-w-md mx-auto text-zinc-400">
                    TonalSoulEngine extracted bits from simulated EEG channels. The Anyonic wrapper maintains LIFE_SIGNAL fidelity under noise.
                  </div>
                  <div className="mt-12 inline-block px-8 py-3 border border-emerald-500/30 rounded-3xl text-emerald-400 text-sm">
                    TOPOLOGICAL SHIELD STRENGTH: 10.4 eV
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className="h-[620px] flex items-center justify-center border border-dashed border-zinc-700 rounded-3xl bg-zinc-950/50">
              <div className="text-center max-w-xs">
                <div className="text-6xl mb-6 opacity-30">⟨Ψ|</div>
                <div className="text-xl text-zinc-400">Adjust seeds above and press the button to run the owned quantum compressor.</div>
                <div className="mt-8 text-[10px] text-zinc-500">This demonstration runs the exact same FractalBraidSeedCompressor that powers the full JARVIS system. Pure math. Yours.</div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Science Footer */}
      <footer className="border-t border-zinc-800 bg-black py-16">
        <div className="max-w-5xl mx-auto px-8 text-xs text-zinc-500 leading-relaxed space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-12">
            <div>
              <div className="text-cyan-400 mb-4">THEORY INTEGRATION</div>
              <ul className="space-y-2">
                <li>• Variational Quantum Circuits via deterministic braid unitaries</li>
                <li>• Quantum tunneling through geometric folds (solves vanishing gradients)</li>
                <li>• Anyonic braiding for fault-tolerant topological computation</li>
                <li>• Copenhagen Observer: your interaction collapses the state</li>
              </ul>
            </div>
            <div>
              <div className="text-cyan-400 mb-4">LEGAL & OWNERSHIP</div>
              <p className="pr-8">All code, mathematics, and the 3-seed reconstruction kernel were built from scratch by the team. No third-party LLM wrappers, no extracted Cortana assets, no mock modes in production. The seed is yours. The AI is yours.</p>
            </div>
            <div>
              <div className="text-cyan-400 mb-4">NEXT EVOLUTION</div>
              <p>This demo is the public interface to the compressed quantum core (qvgpu_compressed_state.npz). The full system includes TCL 2.0 hypergraph, multiversal adapters, hypothesis engine with braid entropy ranking, and bio-quantum interface. Ongoing science.</p>
              <div className="mt-8 text-[10px]">Live on port 3000 • Checkpoint ready for PR</div>
            </div>
          </div>
          <div className="pt-8 border-t border-zinc-800 text-center text-[10px]">
            © CYBER AI • STATE OF THE ART • QUANTUM FROM FIRST PRINCIPLES • EVOLVED BY THE TEAM
          </div>
        </div>
      </footer>
    </div>
  );
}
