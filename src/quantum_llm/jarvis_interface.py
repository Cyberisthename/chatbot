"""
JARVIS Interface for Quantum LLM
Integrates Quantum LLM with JARVIS ecosystem
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from .quantum_transformer import QuantumTransformer, SimpleTokenizer
from .quantum_attention import QuantumSuperposition, QuantumAttention
from .training_engine import TrainingConfig, QuantumTrainingEngine


class JarvisQuantumLLM:
    """
    Main interface for Quantum LLM integrated with JARVIS
    Connects to quantum engines, adapter system, and TCL
    """
    
    def __init__(
        self,
        model: Optional[QuantumTransformer] = None,
        config: Optional[TrainingConfig] = None,
        jarvis_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize JARVIS Quantum LLM
        
        Args:
            model: Pre-trained QuantumTransformer (optional)
            config: Training configuration (optional)
            jarvis_config: JARVIS integration configuration
        """
        self.config = config or TrainingConfig()
        self.jarvis_config = jarvis_config or {}
        
        # Initialize or load model
        if model is None:
            self.model = QuantumTransformer(
                vocab_size=self.config.vocab_size,
                d_model=self.config.d_model,
                n_layers=self.config.n_layers,
                n_heads=self.config.n_heads,
                d_ff=self.config.d_ff,
                max_seq_len=self.config.max_seq_len,
                dropout=self.config.dropout
            )
        else:
            self.model = model
        
        # Initialize tokenizer
        self.tokenizer = SimpleTokenizer(vocab_size=self.config.vocab_size)
        
        # Initialize training engine
        self.training_engine = QuantumTrainingEngine(self.config, self.model)
        
        # JARVIS integration
        self.adapter_engine = None
        self.multiverse_engine = None
        self.tcl_engine = None
        
        # Initialize JARVIS components if configured
        self._init_jarvis_integration()
        
        # Knowledge base
        self.knowledge_base = []
        self.conversation_history = []
        
        # Metrics
        self.interaction_count = 0
        self.quantum_states = []
        
        print("🤖 JARVIS Quantum LLM initialized")
        print(f"   Model: {self.config.d_model}d, {self.config.n_layers} layers")
        print(f"   Parameters: {self._count_parameters():,}")
    
    def _init_jarvis_integration(self):
        """Initialize JARVIS ecosystem integration"""
        try:
            # Import JARVIS components
            from src.core.adapter_engine import AdapterEngine
            from src.core.multiversal_compute_system import MultiversalComputeSystem
            from src.thought_compression.tcl_engine import ThoughtCompressionEngine
            
            # Initialize adapter engine
            adapter_config = {
                "adapters": {
                    "storage_path": "./adapters",
                    "graph_path": "./adapters_graph.json"
                },
                "bits": {
                    "y_bits": 16,
                    "z_bits": 8,
                    "x_bits": 8
                }
            }
            self.adapter_engine = AdapterEngine(adapter_config)
            
            # Initialize multiverse engine
            multiverse_config = {
                "multiverse": {"storage_path": "./multiverse"},
                "artifacts": {"storage_path": "./artifacts"}
            }
            self.multiverse_engine = MultiversalComputeSystem(multiverse_config)
            
            # Initialize TCL engine
            self.tcl_engine = ThoughtCompressionEngine(enable_quantum_mode=True)
            
            print("✅ JARVIS ecosystem integrated")
            print("   - Adapter Engine")
            print("   - Multiverse Engine")
            print("   - TCL Engine")
            
        except ImportError as e:
            print(f"⚠️  JARVIS components not available: {e}")
            print("   Running in standalone mode")
    
    def _count_parameters(self) -> int:
        """Count model parameters"""
        total = 0
        for layer in self.model.layers:
            total += layer.query_proj.size + layer.key_proj.size + layer.value_proj.size
            total += layer.ffn1.size + layer.ffn2.size
        total += self.model.embedding.size + self.model.output_projection.size
        return total
    
    def chat(
        self,
        user_input: str,
        max_tokens: int = 100,
        temperature: float = 0.8,
        use_quantum_enhancement: bool = True
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Chat with the Quantum LLM
        
        Args:
            user_input: User's input text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            use_quantum_enhancement: Use quantum enhancement from JARVIS
            
        Returns:
            Tuple of (response, metrics)
        """
        start_time = time.time()
        self.interaction_count += 1
        
        # Store in conversation history
        self.conversation_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": time.time()
        })
        
        # Generate response
        response, generation_metrics = self.model.generate(
            prompt=user_input,
            tokenizer=self.tokenizer,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=50
        )
        
        # Store assistant response
        self.conversation_history.append({
            "role": "assistant",
            "content": response,
            "timestamp": time.time()
        })
        
        # Get quantum metrics
        quantum_metrics = generation_metrics.get("quantum_metrics", {})
        
        # Apply quantum enhancement if available
        if use_quantum_enhancement:
            response, enhancement_metrics = self._apply_quantum_enhancement(
                user_input, response, quantum_metrics
            )
        else:
            enhancement_metrics = {}
        
        # Compute interaction metrics
        interaction_time = time.time() - start_time
        metrics = {
            "interaction_id": self.interaction_count,
            "interaction_time": interaction_time,
            "quantum_coherence": float(quantum_metrics.get("avg_coherence", 0)),
            "quantum_entanglement": float(quantum_metrics.get("avg_entanglement", 0)),
            "quantum_interference": float(quantum_metrics.get("avg_interference", 0)),
            "tokens_generated": generation_metrics["generated_tokens"],
            "generation_metrics": generation_metrics,
            "enhancement_metrics": enhancement_metrics,
        }
        
        # Store quantum state snapshot
        self.quantum_states.append({
            "interaction_id": self.interaction_count,
            "metrics": quantum_metrics,
            "timestamp": time.time()
        })
        
        return response, metrics
    
    def _apply_quantum_enhancement(
        self,
        user_input: str,
        response: str,
        quantum_metrics: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Apply quantum enhancement using JARVIS components
        
        Args:
            user_input: User's input
            response: Generated response
            quantum_metrics: Quantum metrics from generation
            
        Returns:
            Tuple of (enhanced_response, enhancement_metrics)
        """
        enhancement_metrics = {}
        
        # Use multiverse engine for alternative responses
        if self.multiverse_engine:
            try:
                from src.core.multiversal_compute_system import MultiversalQuery
                
                query = MultiversalQuery(
                    query_id=f"chat_{self.interaction_count}",
                    problem_description=user_input,
                    problem_domain="conversation",
                    complexity=quantum_metrics.get("avg_coherence", 0.5),
                    urgency="medium",
                    max_universes=3
                )
                
                # Process through multiverse (simplified)
                # In production, would get actual multiverse insights
                enhancement_metrics["multiverse_accessed"] = True
            except Exception as e:
                enhancement_metrics["multiverse_error"] = str(e)
        
        # Use TCL for thought compression
        if self.tcl_engine:
            try:
                session_id = self.tcl_engine.create_session(f"chat_{self.interaction_count}")
                
                # Compress user input and response
                compressed_input = self.tcl_engine.compress_concept(session_id, user_input)
                compressed_response = self.tcl_engine.compress_concept(session_id, response)
                
                enhancement_metrics["tcl_compression"] = {
                    "input_ratio": compressed_input.get("compression_ratio", 0),
                    "response_ratio": compressed_response.get("compression_ratio", 0),
                }
            except Exception as e:
                enhancement_metrics["tcl_error"] = str(e)
        
        # Create adapter for this interaction
        if self.adapter_engine:
            try:
                # Infer Y/Z/X bits from input
                y_bits = [0] * 16
                if "quantum" in user_input.lower():
                    y_bits[0] = 1
                if "scientific" in user_input.lower():
                    y_bits[1] = 1
                
                z_bits = [0] * 8
                if len(user_input) > 100:
                    z_bits[0] = 1
                
                x_bits = [0] * 8
                if quantum_metrics.get("avg_coherence", 0) > 0.7:
                    x_bits[0] = 1
                
                adapter = self.adapter_engine.create_adapter(
                    task_tags=["chat", "quantum_llm"],
                    y_bits=y_bits,
                    z_bits=z_bits,
                    x_bits=x_bits,
                    parameters={
                        "interaction_id": self.interaction_count,
                        "quantum_coherence": quantum_metrics.get("avg_coherence", 0),
                        "response_length": len(response),
                    }
                )
                
                enhancement_metrics["adapter_created"] = adapter.id
            except Exception as e:
                enhancement_metrics["adapter_error"] = str(e)
        
        return response, enhancement_metrics
    
    def train(
        self,
        dataset_type: str = "wikitext",
        epochs: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Train the Quantum LLM
        
        Args:
            dataset_type: Type of dataset to use
            epochs: Number of epochs (overrides config)
            
        Returns:
            Training metrics
        """
        print("\n🎓 Starting Quantum LLM training...")
        
        # Override epochs if specified
        if epochs is not None:
            self.config.epochs = epochs
        
        # Load dataset
        self.training_engine.load_dataset(dataset_type)
        
        # Train
        self.training_engine.train()
        
        # Get final metrics
        training_metrics = {
            "total_epochs": self.config.epochs,
            "final_train_loss": float(self.training_engine.train_losses[-1]),
            "best_val_loss": float(self.training_engine.best_val_loss),
            "total_steps": self.training_engine.global_step,
            "quantum_metrics_aggregated": {
                "avg_coherence": float(np.mean([m.get("avg_coherence", 0) for m in self.training_engine.quantum_metrics_history])),
                "avg_entanglement": float(np.mean([m.get("avg_entanglement", 0) for m in self.training_engine.quantum_metrics_history])),
                "avg_interference": float(np.mean([m.get("avg_interference", 0) for m in self.training_engine.quantum_metrics_history])),
                "avg_fidelity": float(np.mean([m.get("avg_fidelity", 0) for m in self.training_engine.quantum_metrics_history])),
            }
        }
        
        print("✅ Training complete!")
        return training_metrics
    
    def run_quantum_experiment(
        self,
        experiment_type: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run a quantum experiment using the LLM
        
        Args:
            experiment_type: Type of experiment
            parameters: Experiment parameters
            
        Returns:
            Experiment results
        """
        print(f"\n🔬 Running quantum experiment: {experiment_type}")
        
        results = {
            "experiment_type": experiment_type,
            "parameters": parameters or {},
            "timestamp": time.time(),
        }
        
        if experiment_type == "coherence_analysis":
            results.update(self._analyze_quantum_coherence())
        
        elif experiment_type == "entanglement_test":
            results.update(self._test_quantum_entanglement())
        
        elif experiment_type == "interference_pattern":
            results.update(self._analyze_interference_patterns())
        
        elif experiment_type == "fidelity_measurement":
            results.update(self._measure_quantum_fidelity())
        
        else:
            results["error"] = f"Unknown experiment type: {experiment_type}"
        
        # Store in knowledge base
        self.knowledge_base.append(results)
        
        return results
    
    def _analyze_quantum_coherence(self) -> Dict[str, Any]:
        """Analyze quantum coherence of the model"""
        coherence_values = []
        
        # Generate several responses and measure coherence
        test_prompts = [
            "What is quantum mechanics?",
            "Explain machine learning",
            "Describe scientific research",
        ]
        
        for prompt in test_prompts:
            _, metrics = self.chat(prompt, max_tokens=50, temperature=0.7)
            coherence_values.append(metrics["quantum_coherence"])
        
        avg_coherence = np.mean(coherence_values)
        std_coherence = np.std(coherence_values)
        
        return {
            "avg_coherence": float(avg_coherence),
            "std_coherence": float(std_coherence),
            "coherence_values": coherence_values,
            "coherence_stable": std_coherence < 0.1,
        }
    
    def _test_quantum_entanglement(self) -> Dict[str, Any]:
        """Test quantum entanglement between attention heads"""
        entanglement_values = []
        
        # Generate responses and measure entanglement
        test_prompts = ["test"] * 5
        
        for prompt in test_prompts:
            _, metrics = self.chat(prompt, max_tokens=20)
            entanglement_values.append(metrics["quantum_entanglement"])
        
        avg_entanglement = np.mean(entanglement_values)
        
        return {
            "avg_entanglement": float(avg_entanglement),
            "entanglement_values": entanglement_values,
            "entanglement_present": avg_entanglement > 0.1,
        }
    
    def _analyze_interference_patterns(self) -> Dict[str, Any]:
        """Analyze quantum interference in attention"""
        interference_values = []
        
        test_prompts = ["test"] * 5
        
        for prompt in test_prompts:
            _, metrics = self.chat(prompt, max_tokens=20)
            interference_values.append(metrics["quantum_interference"])
        
        avg_interference = np.mean(interference_values)
        
        return {
            "avg_interference": float(avg_interference),
            "interference_values": interference_values,
            "interference_detected": avg_interference > 0.3,
        }
    
    def _measure_quantum_fidelity(self) -> Dict[str, Any]:
        """Measure quantum fidelity of states"""
        fidelity_values = []
        
        test_prompts = ["test"] * 5
        
        for prompt in test_prompts:
            _, metrics = self.chat(prompt, max_tokens=20)
            fidelity_values.append(metrics["quantum_fidelity"])
        
        avg_fidelity = np.mean(fidelity_values)
        
        return {
            "avg_fidelity": float(avg_fidelity),
            "fidelity_values": fidelity_values,
            "fidelity_high": avg_fidelity > 0.7,
        }
    
    def save_state(self, path: str):
        """Save complete JARVIS Quantum LLM state"""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = save_path / "model.json"
        self.model.save(model_path)
        
        # Save conversation history
        history_path = save_path / "conversation_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.conversation_history, f, indent=2)
        
        # Save knowledge base
        knowledge_path = save_path / "knowledge_base.json"
        with open(knowledge_path, 'w') as f:
            json.dump(self.knowledge_base, f, indent=2)
        
        # Save quantum states
        quantum_path = save_path / "quantum_states.json"
        with open(quantum_path, 'w') as f:
            json.dump(self.quantum_states, f, indent=2)
        
        print(f"✅ Saved JARVIS Quantum LLM state to {save_path}")
    
    def load_state(self, path: str):
        """Load JARVIS Quantum LLM state"""
        load_path = Path(path)
        
        # Load model
        model_path = load_path / "model.json"
        if model_path.exists():
            self.model.load(model_path)
        
        # Load conversation history
        history_path = load_path / "conversation_history.json"
        if history_path.exists():
            with open(history_path, 'r') as f:
                self.conversation_history = json.load(f)
        
        # Load knowledge base
        knowledge_path = load_path / "knowledge_base.json"
        if knowledge_path.exists():
            with open(knowledge_path, 'r') as f:
                self.knowledge_base = json.load(f)
        
        # Load quantum states
        quantum_path = load_path / "quantum_states.json"
        if quantum_path.exists():
            with open(quantum_path, 'r') as f:
                self.quantum_states = json.load(f)
        
        print(f"✅ Loaded JARVIS Quantum LLM state from {load_path}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of JARVIS Quantum LLM"""
        return {
            "model_parameters": self._count_parameters(),
            "config": {
                "vocab_size": self.config.vocab_size,
                "d_model": self.config.d_model,
                "n_layers": self.config.n_layers,
                "n_heads": self.config.n_heads,
            },
            "interaction_count": self.interaction_count,
            "knowledge_base_size": len(self.knowledge_base),
            "conversation_length": len(self.conversation_history),
            "jarvis_integration": {
                "adapter_engine": self.adapter_engine is not None,
                "multiverse_engine": self.multiverse_engine is not None,
                "tcl_engine": self.tcl_engine is not None,
            }
        }


__all__ = ["JarvisQuantumLLM"]
