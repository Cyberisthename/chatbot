"""Refined autonomous hypothesis engine with live quantum-metric feedback.

This module connects the Thought-Compression Language (TCL) layer to the
repository's actual ``QuantumTransformer`` implementation instead of relying on
purely simulated scores. Candidate hypotheses are scored from repeated forward
passes through the transformer, allowing the engine to identify stable
constructive interference patterns and attach falsification plans.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import statistics
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from ..quantum_llm.braid_math import get_braid_metrics
from ..quantum_llm.quantum_transformer import QuantumTransformer, SimpleTokenizer
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
    "guided": "guided",
    "guides": "guide",
    "mutation": "mutation",
    "mutations": "mutation",
    "syndrome": "syndrome",
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
    ensemble_groups: List[List[str]] = field(default_factory=list) # Tier-2 Ensembles

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QuantumMetricCycle:
    cycle_index: int
    prompt: str
    coherence: float
    entanglement: float
    interference: float
    quantum_fidelity: float
    braid_entropy: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StableInterferenceReport:
    cycles_observed: int
    mean_interference: float
    std_interference: float
    min_interference: float
    max_interference: float
    constructive_fraction: float
    stability_score: float
    stable: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QuantumMetricsTrace:
    source: str
    coherence: float
    entanglement: float
    interference: float
    quantum_fidelity: float
    braid_entropy: float
    information_density: float
    novelty_regime: str
    high_value_discovery: bool
    braid_word: List[int] = field(default_factory=list)
    cycle_metrics: List[QuantumMetricCycle] = field(default_factory=list)
    stable_interference: Optional[StableInterferenceReport] = None

    @property
    def interference_samples(self) -> List[float]:
        return [cycle.interference for cycle in self.cycle_metrics]

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["cycle_metrics"] = [cycle.to_dict() for cycle in self.cycle_metrics]
        payload["stable_interference"] = (
            self.stable_interference.to_dict() if self.stable_interference else None
        )
        payload["interference_samples"] = self.interference_samples
        return payload


@dataclass
class HypothesisEvaluation:
    novelty: float
    plausibility: float
    interference_gain: float
    stability: float
    testability: float
    braid_novelty: float
    overall_score: float
    stable_constructive_interference: bool
    quantum_metrics: QuantumMetricsTrace

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


class StableInterferenceDetector:
    """Detect stable constructive interference across repeated inference cycles."""

    def __init__(self, constructive_threshold: float = 0.62, spread_tolerance: float = 0.10):
        self.constructive_threshold = constructive_threshold
        self.spread_tolerance = spread_tolerance

    def analyze(self, interference_samples: Sequence[float]) -> StableInterferenceReport:
        if not interference_samples:
            return StableInterferenceReport(
                cycles_observed=0,
                mean_interference=0.0,
                std_interference=1.0,
                min_interference=0.0,
                max_interference=0.0,
                constructive_fraction=0.0,
                stability_score=0.0,
                stable=False,
            )

        mean_interference = float(statistics.mean(interference_samples))
        std_interference = float(statistics.pstdev(interference_samples))
        constructive_fraction = float(
            sum(sample >= self.constructive_threshold for sample in interference_samples)
            / len(interference_samples)
        )
        stability_score = _clip(1.0 - std_interference / max(self.spread_tolerance, 1e-6))
        stable = (
            mean_interference >= self.constructive_threshold
            and constructive_fraction >= 0.75
            and stability_score >= 0.65
        )
        return StableInterferenceReport(
            cycles_observed=len(interference_samples),
            mean_interference=mean_interference,
            std_interference=std_interference,
            min_interference=float(min(interference_samples)),
            max_interference=float(max(interference_samples)),
            constructive_fraction=constructive_fraction,
            stability_score=stability_score,
            stable=stable,
        )


class AutonomousHypothesisEngine:
    """Hypothesis engine driven by live metrics from the QuantumTransformer."""

    def __init__(
        self,
        tcl_engine: Optional[ThoughtCompressionEngine] = None,
        session_id: Optional[str] = None,
        quantum_model: Optional[QuantumTransformer] = None,
        tokenizer: Optional[SimpleTokenizer] = None,
        transformer_config: Optional[Dict[str, Any]] = None,
        inference_cycles: int = 5,
        proposal_threshold: float = 0.64,
        constructive_threshold: float = 0.62,
        random_seed: int = 7,
    ):
        self.tcl_engine = tcl_engine or ThoughtCompressionEngine(enable_quantum_mode=True)
        self.session_id = session_id or self.tcl_engine.create_session(
            "autonomous_hypothesis_engine",
            cognitive_level=0.85,
        )
        self.inference_cycles = max(3, inference_cycles)
        self.proposal_threshold = proposal_threshold
        self.random_seed = random_seed
        self.observations: List[ResearchObservation] = []
        self.observation_index: Dict[str, ResearchObservation] = {}
        self.detector = StableInterferenceDetector(constructive_threshold=constructive_threshold)

        if quantum_model is None or tokenizer is None:
            quantum_model, tokenizer = self._build_quantum_stack(transformer_config or {})
        self.quantum_model = quantum_model
        self.tokenizer = tokenizer

    @property
    def context(self):
        return self.tcl_engine.sessions[self.session_id]

    def _build_quantum_stack(
        self,
        transformer_config: Dict[str, Any],
    ) -> tuple[QuantumTransformer, SimpleTokenizer]:
        config = {
            "vocab_size": 2048,
            "d_model": 128,
            "n_layers": 4,
            "n_heads": 4,
            "d_ff": 512,
            "max_seq_len": 128,
            "dropout": 0.0,
        }
        config.update(transformer_config)

        previous_state = np.random.get_state()
        np.random.seed(self.random_seed)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                model = QuantumTransformer(**config)
        finally:
            np.random.set_state(previous_state)
        tokenizer = SimpleTokenizer(vocab_size=config["vocab_size"])
        return model, tokenizer

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
                continue

            theme_phrase = ", ".join(shared_themes[:3]) if shared_themes else "error-control"
            target_entity = right.entities[0] if right.entities else right.title
            outcome_phrase = right.outcome.lower()
            claim = (
                f"Transplanting the '{left.mechanism}' control pattern from {left.domain} into "
                f"{right.domain} systems may yield {outcome_phrase} through shared "
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

        # Tier-2: Redundant-guide ensembles for high-weight mechanisms
        for cand in candidates:
            # Create ensemble of 3 redundant guides (represented as perturbed mechanisms)
            # This simulates fault-tolerant targeting in biology (Milestone A)
            redundant_guides = [f"{cand.mechanism}_α", f"{cand.mechanism}_β", f"{cand.mechanism}_γ"]
            cand.ensemble_groups.append(redundant_guides)

        return candidates

    def collect_quantum_metrics(self, candidate: CandidateHypothesis) -> QuantumMetricsTrace:
        prompt_cycles = self._build_prompt_cycles(candidate)
        cycle_metrics: List[QuantumMetricCycle] = []
        metric_vectors: List[np.ndarray] = []

        # Tier-2: Biological repair pathway conditioning (Milestone C)
        repair_bias_str = self.context.metrics.repair_pathway_bias
        repair_bias_val = 1.1 if repair_bias_str == "HDR" else 1.0

        for cycle_index in range(self.inference_cycles):
            prompt = prompt_cycles[cycle_index % len(prompt_cycles)]
            encoded = self.tokenizer.encode(prompt)
            encoded = encoded[-self.quantum_model.max_seq_len :]
            input_ids = np.array(encoded, dtype=np.int64).reshape(1, -1)
            _, live_metrics = self.quantum_model.forward(input_ids, use_cache=True, repair_bias=repair_bias_val)

            current_vector = np.array(
                [
                    float(live_metrics.get("avg_coherence", 0.0)),
                    float(live_metrics.get("avg_entanglement", 0.0)),
                    float(live_metrics.get("avg_interference", 0.0)),
                    float(live_metrics.get("avg_fidelity", 0.0)),
                ]
            )
            metric_vectors.append(current_vector)

        braid_word, n_strands = self._construct_reasoning_braid(candidate, metric_vectors)
        braid_metrics = get_braid_metrics(braid_word=braid_word, n_strands=n_strands)
        braid_entropy = float(braid_metrics["braid_entropy"])

        for cycle_index, (prompt, current_vector) in enumerate(zip(prompt_cycles, metric_vectors)):
            cycle_entropy = self._cycle_braid_entropy(braid_entropy, cycle_index, len(metric_vectors))
            cycle_metrics.append(
                QuantumMetricCycle(
                    cycle_index=cycle_index,
                    prompt=prompt,
                    coherence=current_vector[0],
                    entanglement=current_vector[1],
                    interference=current_vector[2],
                    quantum_fidelity=current_vector[3],
                    braid_entropy=cycle_entropy,
                )
            )

        detector_report = self.detector.analyze([cycle.interference for cycle in cycle_metrics])
        novelty_regime = str(braid_metrics["novelty_regime"])
        high_value_discovery = novelty_regime.startswith("III")
        return QuantumMetricsTrace(
            source="transformer_forward",
            coherence=float(statistics.mean(cycle.coherence for cycle in cycle_metrics)),
            entanglement=float(statistics.mean(cycle.entanglement for cycle in cycle_metrics)),
            interference=float(statistics.mean(cycle.interference for cycle in cycle_metrics)),
            quantum_fidelity=float(statistics.mean(cycle.quantum_fidelity for cycle in cycle_metrics)),
            braid_entropy=braid_entropy,
            information_density=float(braid_metrics["information_density"]),
            novelty_regime=novelty_regime,
            high_value_discovery=high_value_discovery,
            braid_word=braid_word,
            cycle_metrics=cycle_metrics,
            stable_interference=detector_report,
        )

    def _syndrome_decoder_score(self, candidate: CandidateHypothesis) -> float:
        """Aggregate signals from redundant-guide ensembles using batched inference."""
        if not candidate.ensemble_groups:
            return 0.0
            
        # Tier-2: Biological repair pathway conditioning (Milestone C)
        repair_bias_str = self.context.metrics.repair_pathway_bias
        repair_bias_val = 1.1 if repair_bias_str == "HDR" else 1.0

        all_group_scores = []
        for group in candidate.ensemble_groups:
            # Batch inference for the whole group (Milestone B)
            batch_prompts = [f"Validation of guide {g} for mechanism {candidate.mechanism}" for g in group]
            encoded_batch = []
            for p in batch_prompts:
                ids = self.tokenizer.encode(p)
                # Ensure correct sequence length for the model
                if len(ids) > self.quantum_model.max_seq_len:
                    ids = ids[-self.quantum_model.max_seq_len:]
                elif len(ids) < self.quantum_model.max_seq_len:
                    ids = [0]*(self.quantum_model.max_seq_len - len(ids)) + ids
                encoded_batch.append(ids)
                
            input_ids = np.array(encoded_batch, dtype=np.int64)
            _, metrics = self.quantum_model.forward(input_ids, use_cache=True, repair_bias=repair_bias_val)
            
            # Syndrome Decoding: Aggregate signals across the redundant ensemble
            # We take the top 3 consensus signals (ignoring low-performing outliers/noise)
            batch_interferences = metrics.get("batch_interference", [0.0]*len(group))
            
            # Convert to list if it's a numpy array
            if isinstance(batch_interferences, np.ndarray):
                batch_interferences = batch_interferences.tolist()
                
            batch_interferences.sort(reverse=True)
            # Use statistics.mean for robust aggregation
            consensus_signal = statistics.mean(batch_interferences[:3]) 
            all_group_scores.append(consensus_signal)
            
        return float(statistics.mean(all_group_scores))

    def score_candidate(self, candidate: CandidateHypothesis) -> HypothesisEvaluation:
        quantum_metrics = self.collect_quantum_metrics(candidate)
        evidence_strength = self._average_evidence(candidate)
        causal_support = self._causal_support(candidate)
        theme_overlap = self._theme_overlap(candidate)
        cross_domain_bonus = 1.0 if len(set(candidate.source_domains)) > 1 else 0.25
        stability = quantum_metrics.stable_interference.stability_score if quantum_metrics.stable_interference else 0.0
        braid_novelty = quantum_metrics.braid_entropy
        information_density = quantum_metrics.information_density
        regime_bonus = self._novelty_regime_bonus(quantum_metrics.novelty_regime)
        syndrome_score = self._syndrome_decoder_score(candidate)
        repair_bias = self.context.metrics.repair_pathway_bias
        repair_adjustment = 1.1 if repair_bias == "HDR" else 1.0 # HDR is more precise

        novelty = _clip(
            (0.12
            + 0.18 * cross_domain_bonus
            + 0.12 * (1.0 - causal_support)
            + 0.12 * min(theme_overlap * 1.5, 1.0)
            + 0.20 * _clip(braid_novelty / 3.0)
            + 0.20 * information_density
            + 0.18 * regime_bonus) * repair_adjustment
        )

        plausibility = _clip(
            0.35 * evidence_strength
            + 0.18 * causal_support
            + 0.12 * quantum_metrics.coherence
            + 0.08 * quantum_metrics.quantum_fidelity
            + 0.12 * stability
            + 0.08 * quantum_metrics.stable_interference.constructive_fraction
            + 0.07 * regime_bonus
            + 0.10 * syndrome_score
        )

        interference_gain = _clip(
            quantum_metrics.interference
            * (0.50 + 0.50 * stability)
            * (0.55 + 0.45 * information_density)
        )
        testability = self._estimate_testability(candidate, quantum_metrics)

        overall_score = _clip(
            0.22 * novelty
            + 0.24 * plausibility
            + 0.24 * interference_gain
            + 0.14 * testability
            + 0.16 * regime_bonus
        )

        stable_constructive_interference = (
            len(candidate.supporting_observation_ids) > 1
            and quantum_metrics.stable_interference.stable
            and quantum_metrics.high_value_discovery
            and overall_score >= self.proposal_threshold
        )

        return HypothesisEvaluation(
            novelty=novelty,
            plausibility=plausibility,
            interference_gain=interference_gain,
            stability=stability,
            testability=testability,
            braid_novelty=braid_novelty,
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
        controls = [
            "Matched baseline with the proposed mechanism disabled",
            "Negative-control perturbation that preserves measurement noise but removes the causal intervention",
            "Scramble entity-to-mechanism pairings and verify that constructive interference and braid entropy collapse",
        ]
        if observation_notes:
            controls.append(observation_notes[0])

        focus_entity = candidate.premise_entities[-1] if candidate.premise_entities else candidate.mechanism
        experiment_name = (
            f"Perturb-and-measure validation of {candidate.mechanism} centered on {focus_entity}"
        )
        required_data = (
            f"Intervention/control readouts for {focus_entity}, downstream outcome measurements for "
            f"'{candidate.predicted_outcome}', and cycle-by-cycle quantum metrics "
            f"(coherence, entanglement, interference, braid entropy)."
        )
        positive_signal = (
            f"A reproducible shift toward '{candidate.predicted_outcome}' together with sustained "
            f"constructive interference (mean interference {evaluation.quantum_metrics.stable_interference.mean_interference:.2f}), "
            f"a retained novelty regime of {evaluation.quantum_metrics.novelty_regime}, and braid entropy remaining elevated under matched repeats."
        )
        falsifier = (
            f"No improvement in '{candidate.predicted_outcome}', or a drop from {evaluation.quantum_metrics.novelty_regime} to a lower regime "
            f"with braid entropy/interference collapsing once the proposed causal bridge is perturbed or entity-mechanism pairings are scrambled."
        )

        return FalsificationPlan(
            experiment_name=experiment_name,
            required_data=required_data,
            positive_signal=positive_signal,
            falsifier=falsifier,
            controls=controls,
        )

    def _build_prompt_cycles(self, candidate: CandidateHypothesis) -> List[str]:
        supporting = [
            self.observation_index[observation_id]
            for observation_id in candidate.supporting_observation_ids
            if observation_id in self.observation_index
        ]
        summaries = " ".join(observation.summary for observation in supporting)
        titles = "; ".join(observation.title for observation in supporting)
        domains = ", ".join(candidate.source_domains)
        entities = ", ".join(candidate.premise_entities)
        themes = ", ".join(candidate.shared_themes) or "causal coupling"

        prompts = [
            f"Hypothesis cycle 1: {candidate.claim}",
            f"Hypothesis cycle 2: TCL {candidate.tcl_expression}. Entities: {entities}.",
            f"Hypothesis cycle 3: Mechanism {candidate.mechanism}. Outcome {candidate.predicted_outcome}. Domains {domains}.",
            f"Hypothesis cycle 4: Supporting observations {titles}. Shared themes {themes}.",
            f"Hypothesis cycle 5: Evidence {summaries}",
        ]
        return prompts[: self.inference_cycles]

    def _construct_reasoning_braid(
        self,
        candidate: CandidateHypothesis,
        metric_vectors: Sequence[np.ndarray],
    ) -> tuple[List[int], int]:
        concept_count = len(set(candidate.premise_entities + candidate.shared_themes))
        n_strands = max(2, min(6, concept_count if concept_count else 2))
        braid_word: List[int] = []
        prior_rank: Optional[np.ndarray] = None

        for vector in metric_vectors:
            current_rank = np.argsort(vector)
            deltas = np.diff(vector)
            for idx, delta in enumerate(deltas, start=1):
                generator = min(idx, n_strands - 1)
                braid_word.append(generator if delta >= 0 else -generator)

            if prior_rank is not None:
                rank_shift = current_rank - prior_rank
                for idx, shift in enumerate(rank_shift[1:], start=1):
                    generator = min(idx, n_strands - 1)
                    braid_word.append(generator if shift >= 0 else -generator)
            prior_rank = current_rank

        if len(set(candidate.source_domains)) > 1:
            braid_word.extend([1, min(2, n_strands - 1), 1] * 3)
        elif len(candidate.supporting_observation_ids) > 1:
            braid_word.extend([1, min(2, n_strands - 1), -1] * 2)
        else:
            braid_word.extend([1])

        return [generator for generator in braid_word if generator != 0], n_strands

    def _cycle_braid_entropy(self, base_entropy: float, cycle_index: int, total_cycles: int) -> float:
        if total_cycles <= 1:
            return base_entropy
        fraction = (cycle_index + 1) / total_cycles
        return float(base_entropy * (0.85 + 0.15 * fraction))

    def _novelty_regime_bonus(self, novelty_regime: str) -> float:
        if novelty_regime.startswith("III"):
            return 1.0
        if novelty_regime.startswith("II"):
            return 0.55
        return 0.15

    def _average_evidence(self, candidate: CandidateHypothesis) -> float:
        weights = [
            self.observation_index[observation_id].evidence_weight
            for observation_id in candidate.supporting_observation_ids
            if observation_id in self.observation_index
        ]
        if not weights:
            return 0.5
        return _clip(float(statistics.mean(weights)))

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
        direct_support = self.context.causality.causal_edges.get(mechanism_id, {}).get(outcome_id, 0.0)
        support_values.append(direct_support)

        for entity in candidate.premise_entities:
            entity_id = self._find_symbol_id(entity)
            if entity_id:
                support_values.append(
                    self.context.causality.causal_edges.get(entity_id, {}).get(mechanism_id, 0.0)
                )

        return _clip(sum(support_values) / len(support_values)) if support_values else 0.0

    def _estimate_testability(
        self,
        candidate: CandidateHypothesis,
        quantum_metrics: QuantumMetricsTrace,
    ) -> float:
        experiment_defined = any(
            self.observation_index[observation_id].proposed_experiment
            for observation_id in candidate.supporting_observation_ids
            if observation_id in self.observation_index
        )
        measurable_outcome = 1.0 if any(
            keyword in candidate.predicted_outcome.lower()
            for keyword in ["fidelity", "load", "stability", "noise", "mutation", "accuracy"]
        ) else 0.65
        braid_ready = 1.0 if quantum_metrics.braid_entropy >= 0.45 else 0.6
        return _clip(
            0.35 * measurable_outcome
            + 0.30 * float(experiment_defined)
            + 0.20 * braid_ready
            + 0.15 * bool(candidate.premise_entities)
        )

    def _ensure_symbol(self, text: str) -> str:
        symbol_id = self._find_symbol_id(text)
        if symbol_id:
            return symbol_id

        self.tcl_engine.compress_concept(self.session_id, text)
        symbol_id = self._find_symbol_id(text)
        if symbol_id:
            return symbol_id

        normalized = _normalized_name(text)
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
