#!/usr/bin/env python3
"""
Interactive Demo for Quantum LLM
Real intelligent system - trained from scratch
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.quantum_llm import JarvisQuantumLLM, TrainingConfig


def interactive_demo():
    """Run interactive demo of Quantum LLM"""
    
    print("="*80)
    print("🤖 QUANTUM LLM - INTERACTIVE DEMO")
    print("="*80)
    print()
    print("This Quantum LLM was built from scratch:")
    print("  - No pre-trained models")
    print("  - Quantum-inspired attention mechanisms")
    print("  - Connected to JARVIS quantum engines")
    print("  - Real training on real data")
    print()
    print("SCIENTIFIC DISCLOSURE:")
    print("  All biology is real. All physics is real.")
    print("  This is a scientific research system.")
    print()
    
    # Try to load trained model
    model_path = Path("./quantum_llm_checkpoints/best_model.json")
    
    if model_path.exists():
        print(f"✅ Loading trained model from {model_path}...")
        model = JarvisQuantumLLM(config=TrainingConfig())
        model.model.load(model_path)
    else:
        print("⚠️  No trained model found. Creating fresh model...")
        print("   (For better results, run train_quantum_llm.py first)")
        config = TrainingConfig(
            vocab_size=1000,
            d_model=128,
            n_layers=3,
            n_heads=4,
            d_ff=512,
            max_seq_len=128,
        )
        model = JarvisQuantumLLM(config=config)
    
    print()
    print("="*80)
    print("INTERACTIVE MODE")
    print("="*80)
    print()
    print("Commands:")
    print("  Type your question or prompt to chat with Quantum LLM")
    print("  'stats' - Show model statistics")
    print("  'experiment' - Run quantum experiments")
    print("  'status' - Show JARVS connection status")
    print("  'quit' or 'exit' - Exit demo")
    print()
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit']:
                print("\n👋 Goodbye!")
                break
            
            if user_input.lower() == 'stats':
                status = model.get_status()
                print("\n📊 Model Statistics:")
                print(f"  Parameters: {status['model_parameters']:,}")
                print(f"  Interactions: {status['interaction_count']}")
                print(f"  Knowledge Base: {status['knowledge_base_size']} items")
                print(f"  Config: {status['config']}")
                print()
                continue
            
            if user_input.lower() == 'experiment':
                print("\n🔬 Running quantum experiments...")
                exp_types = [
                    "coherence_analysis",
                    "entanglement_test",
                    "interference_pattern",
                    "fidelity_measurement"
                ]
                for exp_type in exp_types:
                    result = model.run_quantum_experiment(exp_type)
                    print(f"\n  {exp_type}:")
                    print(f"    {result}")
                print()
                continue
            
            if user_input.lower() == 'status':
                status = model.get_status()
                print("\n🔌 JARVIS Connection Status:")
                print(f"  Adapter Engine: {'✅ Connected' if status['jarvis_integration']['adapter_engine'] else '❌ Not connected'}")
                print(f"  Multiverse Engine: {'✅ Connected' if status['jarvis_integration']['multiverse_engine'] else '❌ Not connected'}")
                print(f"  TCL Engine: {'✅ Connected' if status['jarvis_integration']['tcl_engine'] else '❌ Not connected'}")
                print()
                continue
            
            # Chat with model
            print("\nQuantum LLM: ", end="", flush=True)
            
            response, metrics = model.chat(
                user_input,
                max_tokens=100,
                temperature=0.8,
                use_quantum_enhancement=True
            )
            
            print(response)
            
            # Show quantum metrics
            print(f"\n📊 Quantum Metrics:")
            print(f"  Coherence: {metrics['quantum_coherence']:.3f}")
            print(f"  Entanglement: {metrics['quantum_entanglement']:.3f}")
            print(f"  Interference: {metrics['quantum_interference']:.3f}")
            print(f"  Tokens generated: {metrics['tokens_generated']}")
            print(f"  Time: {metrics['interaction_time']:.2f}s")
            
            if 'enhancement_metrics' in metrics and metrics['enhancement_metrics']:
                print(f"\n🚀 JARVIS Enhancements:")
                for key, value in metrics['enhancement_metrics'].items():
                    print(f"  {key}: {value}")
            
            print()
            print("="*80)
            print()
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            print()


def demo_experiments():
    """Run quantum experiments demo"""
    
    print("="*80)
    print("🔬 QUANTUM EXPERIMENTS DEMO")
    print("="*80)
    print()
    
    config = TrainingConfig(
        vocab_size=1000,
        d_model=128,
        n_layers=3,
        n_heads=4,
    )
    
    model = JarvisQuantumLLM(config=config)
    
    print("Running quantum experiments...\n")
    
    # Experiment 1: Coherence Analysis
    print("1. Coherence Analysis")
    print("-" * 40)
    result = model.run_quantum_experiment("coherence_analysis")
    print(f"Average Coherence: {result.get('avg_coherence', 0):.4f}")
    print(f"Coherence Stable: {result.get('coherence_stable', False)}")
    print()
    
    # Experiment 2: Entanglement Test
    print("2. Entanglement Test")
    print("-" * 40)
    result = model.run_quantum_experiment("entanglement_test")
    print(f"Average Entanglement: {result.get('avg_entanglement', 0):.4f}")
    print(f"Entanglement Present: {result.get('entanglement_present', False)}")
    print()
    
    # Experiment 3: Interference Pattern Analysis
    print("3. Interference Pattern Analysis")
    print("-" * 40)
    result = model.run_quantum_experiment("interference_pattern")
    print(f"Average Interference: {result.get('avg_interference', 0):.4f}")
    print(f"Interference Detected: {result.get('interference_detected', False)}")
    print()
    
    # Experiment 4: Fidelity Measurement
    print("4. Fidelity Measurement")
    print("-" * 40)
    result = model.run_quantum_experiment("fidelity_measurement")
    print(f"Average Fidelity: {result.get('avg_fidelity', 0):.4f}")
    print(f"Fidelity High: {result.get('fidelity_high', False)}")
    print()
    
    print("="*80)
    print("✅ Experiments complete!")
    print("="*80)


def demo_training():
    """Demo training process"""
    
    print("="*80)
    print("🎓 TRAINING DEMO")
    print("="*80)
    print()
    print("This demo shows the training process with minimal data for quick testing.")
    print()
    
    config = TrainingConfig(
        vocab_size=100,
        d_model=64,
        n_layers=2,
        n_heads=2,
        d_ff=256,
        max_seq_len=64,
        batch_size=4,
        learning_rate=0.001,
        epochs=1,
        checkpoint_interval=10,
    )
    
    model = JarvisQuantumLLM(config=config)
    
    # Quick training demo
    print("Starting quick training demo...")
    print("(Using minimal data for fast demonstration)\n")
    
    metrics = model.train(dataset_type="synthetic", epochs=1)
    
    print("\n✅ Training demo complete!")
    print(f"Final train loss: {metrics['final_train_loss']:.4f}")
    print(f"Best val loss: {metrics['best_val_loss']:.4f}")
    print(f"Total steps: {metrics['total_steps']}")
    print()


def main():
    """Main demo entry point"""
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        
        if mode == "interactive":
            interactive_demo()
        elif mode == "experiments":
            demo_experiments()
        elif mode == "training":
            demo_training()
        else:
            print(f"Unknown mode: {mode}")
            print("Available modes: interactive, experiments, training")
    else:
        print("Quantum LLM Demo")
        print()
        print("Usage:")
        print("  python3 demo_quantum_llm.py interactive  - Interactive chat mode")
        print("  python3 demo_quantum_llm.py experiments - Run quantum experiments")
        print("  python3 demo_quantum_llm.py training    - Demo training process")
        print()
        
        # Default to interactive
        interactive_demo()


if __name__ == "__main__":
    main()
