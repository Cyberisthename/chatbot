"""Research-oriented modules for autonomous scientific discovery."""

from .crispr_falsification_sim import (
    AggregateSimulationReport,
    ArmSummary,
    CrisprFalsificationSimulator,
    QuantumConditionSummary,
    TrialResult,
)
from .hypothesis_engine import (
    AutonomousHypothesisEngine,
    CandidateHypothesis,
    FalsificationPlan,
    HypothesisEvaluation,
    ProposedHypothesis,
    QuantumMetricCycle,
    QuantumMetricsTrace,
    ResearchObservation,
    StableInterferenceDetector,
    StableInterferenceReport,
)

__all__ = [
    "AggregateSimulationReport",
    "ArmSummary",
    "AutonomousHypothesisEngine",
    "CandidateHypothesis",
    "CrisprFalsificationSimulator",
    "FalsificationPlan",
    "HypothesisEvaluation",
    "ProposedHypothesis",
    "QuantumConditionSummary",
    "QuantumMetricCycle",
    "QuantumMetricsTrace",
    "ResearchObservation",
    "StableInterferenceDetector",
    "StableInterferenceReport",
    "TrialResult",
]
