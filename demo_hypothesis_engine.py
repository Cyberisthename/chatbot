"""Demo for the refined autonomous hypothesis engine.

Run:
    python demo_hypothesis_engine.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.research import AutonomousHypothesisEngine


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / "demos" / "modern_research_observations.json"

    engine = AutonomousHypothesisEngine(inference_cycles=5)
    observations = engine.load_observations_from_json(data_path)
    proposals = engine.propose_hypotheses(observations, top_k=3)

    top_feedback = proposals[0].evaluation.quantum_metrics if proposals else None
    payload = {
        "data_path": str(data_path),
        "proposal_count": len(proposals),
        "top_feedback_summary": top_feedback.to_dict() if top_feedback else None,
        "proposals": [proposal.to_dict() for proposal in proposals],
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
