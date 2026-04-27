"""Run the computational simulation for the CRISPR falsification experiment."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.research.crispr_falsification_sim import CrisprFalsificationSimulator


def main() -> None:
    simulator = CrisprFalsificationSimulator()
    report = simulator.run_trials(n_trials=24)
    print(json.dumps(report.to_dict(), indent=2))


if __name__ == "__main__":
    main()
