import { createAPIFileRoute } from "@tanstack/react-start/api";
import { execSync } from "child_process";
import { join } from "path";

export const Route = createAPIFileRoute("/api/compressor")({
  POST: async ({ request }) => {
    try {
      const { seed1 = 0.57721, seed2 = 1.618034, seed3 = 2.71828, qubits = 256 } = await request.json();

      const scriptPath = join(process.cwd(), "compressor_api.py");
      const cmd = `python3 "${scriptPath}" ${seed1} ${seed2} ${seed3} ${Math.floor(qubits)}`;
      
      const output = execSync(cmd, { encoding: "utf-8", cwd: process.cwd() }).trim();
      const result = JSON.parse(output);

      return {
        status: 200,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          success: true,
          data: result,
          params: { seed1, seed2, seed3, qubits },
          timestamp: new Date().toISOString(),
          note: "Owned Fractal-Braid Seed Compressor (FBSC) v2. Quantum state reconstructed from 3 seeds only. State-of-the-art topological folding + bio-resonance integration."
        })
      };
    } catch (error: any) {
      return {
        status: 500,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          success: false,
          error: error.message,
          note: "Quantum compression kernel encountered interference. Try different seeds or check Python env."
        })
      };
    }
  },
});
