"""Prototype autonomous hypothesis engine for scientific discovery.

This module bridges the existing Thought-Compression Language (TCL) system with
simulated quantum metrics inspired by the repository's Quantum Transformer.
It is designed to:

1. ingest structured scientific observations,
2. compress them into TCL-compatible concepts,
3. generate direct and cross-domain candidate hypotheses,
4. score those candidates using novelty, plausibility, and interference,
5. propose only the candidates that exhibit stable constructive interference,
6. attach a falsification plan so every accepted hypothesis is testable.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from ..thought_compression import ThoughtCompressionEngine


def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _normalized_name(text: str) -> str:
    return text.strip().lower().replace("-", "_").replace(" ", "_")


def _deterministic_unit(key: str) -> float:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    integer = int.from_bytes(digest[:8], "big")
    return integer / float(2**64 - 1)


SCIENTIFIC_STOPWORDS = {
    "and",
    "are",
    "against",
    "before",
    "can",
    "during",
    "from",
    "into",
    "its",
    "may",
    "that",
    "the",
    "their",
    "they",
    "this",
    "through",
    "under",
    "uses",
    "using",
    "with",
    "improve",
    "improved",
    "improves",
}


def _tokenize_scientific_text(*parts: str) -> List[str]:
    tokens: List[str] = []
    for part in parts:
        cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in part)
        tokens.extend(
            token
            for token in cleaned.split()
            if len(token) > 2 and token not in SCIENTIFIC_STOPWORDS
        )
    return tokens


THEME_ALIASES: Dict[str, str] = {
    "errors": "error",
    "repair": "repair",
    "repairs": "repair",
    "editing": "edit",
    "edits": "edit",
    "fidelity": "fidelity",
    "noise": "noise",
    "noisy": "noise",
    "stability": "stability",
    "stable": "stability",
    "screening": "screening",
    "screen": "screening",
    "feedback": "feedback",
    "checkpoint": "checkpoint",
    "checkpoints": "checkpoint",
    "coherence": "coherence",
    "decoder": "decode",
    "decoding": "decode",
    "correlated": "correlation",
    "redundant": "redundancy",
    "redundancy": "redundancy",
    "detection": "detect",
    "detects": "detect",
    "detected": "detect",
    "guide": "guide",
    "guides": "guide",
    "mutation": "mutation",
    "mutations": "mutation",
}


@dataclass
class ResearchObservation:
    id: str
    title: str
    domain: str
    summary: str
    entities: List[str]
    mechanism: str
    outcome: str
    evidence_weight: float = 0.6
    year: Optional[int] = None
    proposed_experiment: Optional[str] = None

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ResearchObservation":
        return cls(
            id=str(payload["id"]),
            title=str(payload["title"]),
            domain=str(payload["domain"]),
            summary=str(payload.get("summary", "")),
            entities=[str(entity) for entity in payload.get("entities", [])],
            mechanism=str(payload["mechanism"]),
            outcome=str(payload["outcome"]),
            evidence_weight=float(payload.get("evidence_weight", 0.6)),
            year=int(payload["year"]) if payload.get("year") is not None else None,
            proposed_experiment=payload.get("proposed_experiment"),
        )

    @property
    def themes(self) -> List[str]:
        normalized = []
        for token in _tokenize_scientific_text(
            self.title,
            self.summary,
            self.mechanism,
            self.outcome,
            *self.entities,
        ):
            normalized.append(THEME_ALIASES.get(token, token))
        return sorted(set(normalized))


@dataclass
class CandidateHypothesis:
    id: str
    candidate_type: str
    premise_entities: List[str]
    mechanism: str
    predicted_outcome: str
    claim: str
    tcl_expression: str
    supporting_observation_ids: List[str]
    source_domains: List[str]
    shared_themes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SimulatedQuantumMetrics:
    coherence: float
    entanglement: float
    interference: float
    quantum_fidelity: float
    interference_samples: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HypothesisEvaluation:
    novelty: float
    plausibility: float
    interference_gain: float
    stability: float
    testability: float
    overall_score: float
    stable_constructive_interference: bool
    quantum_metrics: SimulatedQuantumMetrics

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["quantum_metrics"] = self.quantum_metrics.to_dict()
        return payload


@dataclass
class FalsificationPlan:
    experiment_name: str
    required_data: str
    positive_signal: str
    falsifier: str
    controls: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProposedHypothesis:
    candidate: CandidateHypothesis
    evaluation: HypothesisEvaluation
    falsification_plan: FalsificationPlan

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(),
            "evaluation": self.evaluation.to_dict(),
            "falsification_plan": self.falsification_plan.to_dict(),
        }


class AutonomousHypothesisEngine:
    """Prototype engine for autonomous scientific hypothesis discovery."""

    def __init__(
        self,
        tcl_engine: Optional[ThoughtCompressionEngine] = None,
        session_id: Optional[str] = None,
        interference_samples: int = 7,
        proposal_threshold: float = 0.64,
    ):
        self.tcl_engine = tcl_engine or ThoughtCompressionEngine(enable_quantum_mode=True)
        self.session_id = session_id or self.tcl_engine.create_session(
            "autonomous_hypothesis_engine",
            cognitive_level=0.85,
        )
        self.interference_samples = max(3, interference_samples)
        self.proposal_threshold = proposal_threshold
        self.observations: List[ResearchObservation] = []
        self.observation_index: Dict[str, ResearchObservation] = {}

    @property
    def context(self):
        return self.tcl_engine.sessions[self.session_id]

    def load_observations_from_json(self, path: Union[Path, str]) -> List[ResearchObservation]:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list):
            raise ValueError("Observation JSON must contain a list of observations")
        return [ResearchObservation.from_dict(item) for item in payload]

    def ingest_observations(self, observations: Sequence[ResearchObservation]) -> None:
        self.observations = list(observations)
        self.observation_index = {observation.id: observation for observation in self.observations}

        for observation in self.observations:
            entity_ids = [self._ensure_symbol(entity) for entity in observation.entities]
            mechanism_id = self._ensure_symbol(observation.mechanism)
            outcome_id = self._ensure_symbol(observation.outcome)
            self._ensure_symbol(observation.title)
            self._ensure_symbol(observation.summary)

            for entity_id in entity_ids:
                self.context.causality.add_causal_link(entity_id, mechanism_id, observation.evidence_weight)
            self.context.causality.add_causal_link(
                mechanism_id,
                outcome_id,
                _clip(observation.evidence_weight + 0.05),
            )

    def generate_candidate_hypotheses(self) -> List[CandidateHypothesis]:
        if not self.observations:
            return []

        candidates: List[CandidateHypothesis] = []
        for observation in self.observations:
            primary_entity = observation.entities[0] if observation.entities else observation.title
            tcl_expression = (
                f"{_normalized_name(primary_entity)} → "
                f"{_normalized_name(observation.mechanism)} ⟹ "
                f"{_normalized_name(observation.outcome)}"
            )
            claim = (
                f"Strengthening '{observation.mechanism}' around {primary_entity} should increase "
                f"the probability of {observation.outcome.lower()}."
            )
            candidates.append(
                CandidateHypothesis(
                    id=f"direct::{observation.id}",
                    candidate_type="direct_extension",
                    premise_entities=observation.entities or [observation.title],
                    mechanism=observation.mechanism,
                    predicted_outcome=observation.outcome,
                    claim=claim,
                    tcl_expression=tcl_expression,
                    supporting_observation_ids=[observation.id],
                    source_domains=[observation.domain],
                    shared_themes=observation.themes[:6],
                )
            )

        for left, right in combinations(self.observations, 2):
            left_themes = set(left.themes)
            right_themes = set(right.themes)
            shared_themes = sorted(left_themes.intersection(right_themes))
            cross_domain = left.domain != right.domain
            if not shared_themes and not cross_domain:
                continue

            if cross_domain and len(shared_themes) < 2:
                # Cross-domain proposals need at least two aligned scientific themes
                # to count as a stable mechanistic bridge rather than a loose analogy.
                continue

            theme_phrase = ", ".join(shared_themes[:3]) if shared_themes else "error-control"
            target_entity = right.entities[0] if right.entities else right.title
            claim = (
                f"Transplanting the '{left.mechanism}' control pattern from {left.domain} into "
                f"{right.domain} systems may yield {right.outcome.lower()} through shared "
                f"{theme_phrase} dynamics."
            )
            tcl_expression = (
                f"{{{_normalized_name(left.mechanism)}, {_normalized_name(target_entity)}}} ⟹ "
                f"{_normalized_name(right.outcome)}"
            )
            candidate_type = "cross_domain_analogy" if cross_domain else "mechanism_composition"
            candidates.append(
                CandidateHypothesis(
                    id=f"pair::{left.id}::{right.id}",
                    candidate_type=candidate_type,
                    premise_entities=(left.entities[:1] + right.entities[:1]) or [left.title, right.title],
                    mechanism=f"{left.mechanism} adapted to {right.domain}",
                    predicted_outcome=right.outcome,
                    claim=claim,
                    tcl_expression=tcl_expression,
                    supporting_observation_ids=[left.id, right.id],
                    source_domains=[left.domain, right.domain],
                    shared_themes=shared_themes[:6],
                )
            )

        return candidates

    def score_candidate(self, candidate: CandidateHypothesis) -> HypothesisEvaluation:
        quantum_metrics = self._simulate_quantum_metrics(candidate)
        evidence_strength = self._average_evidence(candidate)
        causal_support = self._causal_support(candidate)
        theme_overlap = self._theme_overlap(candidate)
        cross_domain_bonus = 1.0 if len(set(candidate.source_domains)) > 1 else 0.25

        novelty = _clip(
            0.26
            + 0.35 * cross_domain_bonus
            + 0.20 * (1.0 - causal_support)
            + 0.19 * min(theme_overlap * 1.5, 1.0)
        )

        plausibility = _clip(
            0.45 * evidence_strength
            + 0.25 * causal_support
            + 0.15 * quantum_metrics.coherence
            + 0.15 * quantum_metrics.quantum_fidelity
        )

        sample_spread = statistics.pstdev(quantum_metrics.interference_samples)
        stability = _clip(1.0 - sample_spread / 0.12)
        interference_gain = _clip(quantum_metrics.interference * stability)
        testability = self._estimate_testability(candidate)

        overall_score = _clip(
            0.26 * novelty
            + 0.30 * plausibility
            + 0.26 * interference_gain
            + 0.18 * testability
        )

        stable_constructive_interference = (
            len(candidate.supporting_observation_ids) > 1
            and quantum_metrics.interference >= 0.60
            and stability >= 0.70
            and quantum_metrics.coherence >= 0.55
            and overall_score >= self.proposal_threshold
        )

        return HypothesisEvaluation(
            novelty=novelty,
            plausibility=plausibility,
            interference_gain=interference_gain,
            stability=stability,
            testability=testability,
            overall_score=overall_score,
            stable_constructive_interference=stable_constructive_interference,
            quantum_metrics=quantum_metrics,
        )

    def propose_hypotheses(
        self,
        observations: Optional[Sequence[ResearchObservation]] = None,
        top_k: int = 5,
    ) -> List[ProposedHypothesis]:
        if observations is not None:
            self.ingest_observations(observations)

        candidates = self.generate_candidate_hypotheses()
        proposals: List[ProposedHypothesis] = []
        fallback: List[ProposedHypothesis] = []

        for candidate in candidates:
            evaluation = self.score_candidate(candidate)
            proposal = ProposedHypothesis(
                candidate=candidate,
                evaluation=evaluation,
                falsification_plan=self.build_falsification_plan(candidate, evaluation),
            )
            fallback.append(proposal)
            if evaluation.stable_constructive_interference:
                proposals.append(proposal)

        proposals.sort(key=lambda item: item.evaluation.overall_score, reverse=True)
        if proposals:
            return proposals[:top_k]

        fallback.sort(key=lambda item: item.evaluation.overall_score, reverse=True)
        return fallback[:top_k]

    def build_falsification_plan(
        self,
        candidate: CandidateHypothesis,
        evaluation: HypothesisEvaluation,
    ) -> FalsificationPlan:
        observation_notes = [
            self.observation_index[observation_id].proposed_experiment
            for observation_id in candidate.supporting_observation_ids
            if observation_id in self.observation_index
            and self.observation_index[observation_id].proposed_experiment
        ]
        control_notes = [
            "Matched baseline with the proposed mechanism disabled",
            "Negative-control perturbation that preserves measurement noise but removes the causal intervention",
        ]
        if observation_notes:
            control_notes.append(observation_notes[0])

        target_domain = candidate.source_domains[-1]
        focus_entity = candidate.premise_entities[-1]
        experiment_name = (
            f"Perturb-and-measure validation of {candidate.mechanism} in {target_domain} "
            f"systems centered on {focus_entity}"
        )
        required_data = (
            f"Intervention/control readouts for {focus_entity}, outcome measurements for "
            f"'{candidate.predicted_outcome}', and replication across at least two matched contexts."
        )
        positive_signal = (
            f"A reproducible shift toward '{candidate.predicted_outcome}' under the intervention, "
            f"with effect size scaling alongside the candidate mechanism."
        )
        falsifier = (
            f"No improvement in '{candidate.predicted_outcome}', or an effect that disappears once "
            f"controls remove the proposed {', '.join(candidate.shared_themes[:2]) or 'shared-theme'} pathway."
        )

        if evaluation.quantum_metrics.interference < 0.60:
            falsifier += " Low interference also predicts that the cross-domain analogy should fail to generalize."

        return FalsificationPlan(
            experiment_name=experiment_name,
            required_data=required_data,
            positive_signal=positive_signal,
            falsifier=falsifier,
            controls=control_notes,
        )

    def _simulate_quantum_metrics(self, candidate: CandidateHypothesis) -> SimulatedQuantumMetrics:
        evidence_strength = self._average_evidence(candidate)
        theme_overlap = self._theme_overlap(candidate)
        cross_domain_bonus = 1.0 if len(set(candidate.source_domains)) > 1 else 0.2

        coherence_samples: List[float] = []
        entanglement_samples: List[float] = []
        interference_samples: List[float] = []
        fidelity_samples: List[float] = []

        for trial in range(self.interference_samples):
            jitter = (_deterministic_unit(f"{candidate.id}:{trial}") - 0.5) * 0.10
            jitter_secondary = (_deterministic_unit(f"{candidate.id}:secondary:{trial}") - 0.5) * 0.08

            coherence = _clip(
                0.44
                + 0.18 * evidence_strength
                + 0.16 * theme_overlap
                + 0.10 * cross_domain_bonus
                + jitter
            )
            entanglement = _clip(
                0.28
                + 0.24 * cross_domain_bonus
                + 0.22 * theme_overlap
                + 0.08 * evidence_strength
                + jitter_secondary
            )
            interference = _clip(
                0.40
                + 0.22 * theme_overlap
                + 0.20 * evidence_strength
                + 0.12 * cross_domain_bonus
                + 0.08 * coherence
                + 0.05 * entanglement
                + jitter
            )
            fidelity = _clip(
                0.46
                + 0.24 * evidence_strength
                + 0.12 * theme_overlap
                + 0.08 * coherence
                - abs(jitter_secondary) * 0.4
            )

            coherence_samples.append(coherence)
            entanglement_samples.append(entanglement)
            interference_samples.append(interference)
            fidelity_samples.append(fidelity)

        return SimulatedQuantumMetrics(
            coherence=statistics.mean(coherence_samples),
            entanglement=statistics.mean(entanglement_samples),
            interference=statistics.mean(interference_samples),
            quantum_fidelity=statistics.mean(fidelity_samples),
            interference_samples=interference_samples,
        )

    def _average_evidence(self, candidate: CandidateHypothesis) -> float:
        weights = [
            self.observation_index[observation_id].evidence_weight
            for observation_id in candidate.supporting_observation_ids
            if observation_id in self.observation_index
        ]
        if not weights:
            return 0.5
        return _clip(statistics.mean(weights))

    def _theme_overlap(self, candidate: CandidateHypothesis) -> float:
        if not candidate.supporting_observation_ids:
            return 0.0

        theme_sets = [
            set(self.observation_index[observation_id].themes)
            for observation_id in candidate.supporting_observation_ids
            if observation_id in self.observation_index
        ]
        if not theme_sets:
            return 0.0
        if len(theme_sets) == 1:
            return _clip(min(1.0, len(theme_sets[0]) / 8.0))

        overlap = set.intersection(*theme_sets)
        union = set.union(*theme_sets)
        if not union:
            return 0.0
        return _clip(len(overlap) / len(union) + 0.15 * bool(overlap))

    def _causal_support(self, candidate: CandidateHypothesis) -> float:
        mechanism_id = self._find_symbol_id(candidate.mechanism)
        outcome_id = self._find_symbol_id(candidate.predicted_outcome)
        if not mechanism_id or not outcome_id:
            return 0.0

        support_values: List[float] = []
        direct_mechanism_support = self.context.causality.causal_edges.get(mechanism_id, {}).get(outcome_id, 0.0)
        support_values.append(direct_mechanism_support)

        for entity in candidate.premise_entities:
            entity_id = self._find_symbol_id(entity)
            if entity_id:
                support_values.append(
                    self.context.causality.causal_edges.get(entity_id, {}).get(mechanism_id, 0.0)
                )

        if not support_values:
            return 0.0
        return _clip(sum(support_values) / len(support_values))

    def _estimate_testability(self, candidate: CandidateHypothesis) -> float:
        experiment_defined = any(
            self.observation_index[observation_id].proposed_experiment
            for observation_id in candidate.supporting_observation_ids
            if observation_id in self.observation_index
        )
        measurable_outcome = 1.0 if any(
            keyword in candidate.predicted_outcome.lower()
            for keyword in ["fidelity", "load", "stability", "noise", "mutation", "accuracy"]
        ) else 0.65
        entity_grounding = 1.0 if candidate.premise_entities else 0.5
        return _clip(0.35 * measurable_outcome + 0.35 * entity_grounding + 0.30 * float(experiment_defined))

    def _ensure_symbol(self, text: str) -> str:
        normalized = _normalized_name(text)
        symbol_id = self._find_symbol_id(text)
        if symbol_id:
            return symbol_id

        self.tcl_engine.compress_concept(self.session_id, text)
        symbol_id = self._find_symbol_id(text)
        if symbol_id:
            return symbol_id

        # Fallback: concept symbols created from `compress_concept` use underscores.
        symbol_id = self._find_symbol_id(normalized)
        if symbol_id:
            return symbol_id

        raise ValueError(f"Unable to register TCL symbol for concept: {text}")

    def _find_symbol_id(self, text: str) -> Optional[str]:
        normalized = _normalized_name(text)
        for symbol_id, symbol in self.context.symbols.symbols.items():
            if _normalized_name(symbol.name) == normalized:
                return symbol_id
        return None
