#!/usr/bin/env python3
"""
Generate JSONL training data from Ben Lab documentation.

This script reads markdown files from lab_corpus/ and generates
instruction/output pairs for fine-tuning a lab-aware LLM.
"""
import json
import os
import re
from pathlib import Path
from typing import List, Dict, Tuple


def extract_sections(md_content: str, filename: str) -> List[Dict[str, str]]:
    """Extract meaningful sections from markdown content."""
    sections = []
    
    # Split by headers (## and ### level)
    lines = md_content.split('\n')
    current_header = filename.replace('.md', '').replace('_', ' ')
    current_content = []
    
    for line in lines:
        # Check for headers
        header_match = re.match(r'^(#{1,3})\s+(.+)$', line)
        if header_match:
            # Save previous section if it has content
            if current_content:
                content_text = '\n'.join(current_content).strip()
                if content_text:
                    sections.append({
                        'header': current_header,
                        'content': content_text
                    })
            
            # Start new section
            current_header = header_match.group(2).strip()
            current_content = []
        else:
            # Add line to current section
            current_content.append(line)
    
    # Add final section
    if current_content:
        content_text = '\n'.join(current_content).strip()
        if content_text:
            sections.append({
                'header': current_header,
                'content': content_text
            })
    
    return sections


def generate_qa_pairs(sections: List[Dict[str, str]], doc_name: str) -> List[Dict[str, str]]:
    """Generate instruction/output pairs from document sections."""
    pairs = []
    
    for section in sections:
        header = section['header']
        content = section['content']
        
        if not content or len(content) < 50:
            continue
        
        # Generate multiple question types for each section
        
        # 1. "Explain X" format
        pairs.append({
            'instruction': f"Explain {header} in the Ben Lab system.",
            'output': content[:800]  # Truncate very long content
        })
        
        # 2. "What is X?" format
        if len(content) > 100:
            pairs.append({
                'instruction': f"What is {header}?",
                'output': content[:600]
            })
        
        # 3. "Summarize X" format
        if len(content) > 300:
            pairs.append({
                'instruction': f"Summarize the {header} concept from the lab.",
                'output': content[:400]
            })
        
        # 4. Context-specific questions based on keywords
        if 'TRI' in content or 'Time-Reversal Instability' in content:
            pairs.append({
                'instruction': "How do I measure time-reversal instability in a phase?",
                'output': f"To measure TRI (Time-Reversal Instability):\n{content[:500]}"
            })
        
        if 'RSI' in content or 'Replay Sensitivity' in content:
            pairs.append({
                'instruction': "What is RSI and how does it characterize phases?",
                'output': f"RSI (Replay Sensitivity Index):\n{content[:500]}"
            })
        
        if 'clustering' in content.lower() or 'unsupervised' in content.lower():
            pairs.append({
                'instruction': "How do I discover phases without labels?",
                'output': f"Unsupervised phase discovery:\n{content[:500]}"
            })
        
        if 'PhaseDetector' in content:
            pairs.append({
                'instruction': "How do I use PhaseDetector to run experiments?",
                'output': content[:600]
            })
        
        if 'Ising' in content and 'symmetry' in content.lower():
            pairs.append({
                'instruction': "What is an Ising symmetry-breaking phase?",
                'output': content[:500]
            })
        
        if 'SPT' in content or 'topological' in content.lower():
            pairs.append({
                'instruction': "Explain SPT phases in the lab.",
                'output': content[:500]
            })
        
        if 'Jarvis-5090X' in content or 'orchestrator' in content.lower():
            pairs.append({
                'instruction': "What is Jarvis-5090X?",
                'output': content[:600]
            })
        
        if 'cache' in content.lower() and 'recompute' in content.lower():
            pairs.append({
                'instruction': "How does the infinite cache work?",
                'output': content[:500]
            })
        
        if 'compression' in content.lower() and 'FLOP' in content:
            pairs.append({
                'instruction': "Explain FLOP compression in Jarvis-5090X.",
                'output': content[:500]
            })
        
        if 'Y-bit' in content or 'Z-bit' in content or 'G-graph' in content:
            pairs.append({
                'instruction': "What are the bit systems used in Ben Lab?",
                'output': content[:600]
            })
    
    return pairs


def generate_practical_qa() -> List[Dict[str, str]]:
    """Generate practical how-to questions."""
    return [
        {
            'instruction': "How do I run a phase experiment with PhaseDetector?",
            'output': """To run a phase experiment:

1. Create Jarvis5090X orchestrator with devices
2. Initialize PhaseDetector with the orchestrator
3. Call run_phase_experiment() with parameters:
   - phase_type: 'ising_symmetry_breaking', 'spt_cluster', 'trivial_product', or 'pseudorandom'
   - system_size: number of qubits (e.g., 32)
   - depth: circuit depth (e.g., 8)
   - seed: random seed for reproducibility

Example:
```python
detector = PhaseDetector(orchestrator)
result = detector.run_phase_experiment(
    phase_type="ising_symmetry_breaking",
    system_size=32,
    depth=8,
    seed=42
)
```"""
        },
        {
            'instruction': "Design an experiment to maximize TRI.",
            'output': """To maximize TRI (Time-Reversal Instability):

1. Choose a directional phase: Ising symmetry-breaking works best
2. Use high bias (~0.7-0.8) to amplify directionality
3. Use sufficient depth (12-16 layers) for features to diverge
4. Run forward (bias) and reverse (1-bias) experiments
5. Compare feature vectors with L2 distance

Example parameters:
- phase_type: 'ising_symmetry_breaking'
- bias: 0.7
- depth: 12
- system_size: 32

Expected: TRI > 0.05 indicates strong directional sensitivity."""
        },
        {
            'instruction': "What depth should I use for phase experiments?",
            'output': """Depth recommendations by goal:

**Quick testing**: depth = 4-8
- Fast execution
- Good for debugging

**Standard experiments**: depth = 8-12
- Balanced runtime vs signal
- Recommended for TRI, classification

**Deep scaling studies**: depth = 12-20+
- Used for RSI measurements
- Reveals thermalization behavior
- Higher computational cost

**Phase-specific**:
- Trivial product: depth = 4-8 (saturates quickly)
- Ising/SPT: depth = 8-12 (moderate complexity)
- Pseudorandom: depth = 12+ (high scrambling)"""
        },
        {
            'instruction': "How do I interpret TRI values?",
            'output': """TRI (Time-Reversal Instability) interpretation:

**Low TRI (0.0001 - 0.001)**:
- Phase is time-symmetric
- Features barely change under bias reversal
- Example: Trivial product states

**Medium TRI (0.01 - 0.05)**:
- Moderate directional sensitivity
- Some asymmetry in phase structure

**High TRI (0.05+)**:
- Strong directional sensitivity
- Phase has built-in arrow of time
- Example: Ising symmetry-breaking with bias=0.7

Physical meaning: High TRI phases have asymmetric correlation structure that depends on bias direction."""
        },
        {
            'instruction': "What are the four phase types in Ben Lab?",
            'output': """The four synthetic phase types:

1. **Ising Symmetry Breaking** (ising_symmetry_breaking)
   - High magnetization
   - Broken symmetry (symmetry_indicator = 1.0)
   - Low entropy, medium correlation
   - Use case: Directional TRI studies

2. **SPT Cluster** (spt_cluster)
   - High string order (~0.85)
   - Edge mode imbalance
   - Topological indicators
   - Use case: Protected edge studies

3. **Trivial Product** (trivial_product)
   - Near-zero string order
   - Low correlation length
   - Minimal entanglement
   - Use case: Baseline/control experiments

4. **Pseudorandom** (pseudorandom)
   - High entropy, high scrambling
   - Uniform probability distribution
   - Maximally scrambled
   - Use case: Complexity scaling studies"""
        },
        {
            'instruction': "Explain the feature vector dimensions.",
            'output': """The 16-dimensional phase feature vector:

**Entropy metrics (4)**:
1. entropy_mean - Average entropy across layers
2. entropy_max - Peak entropy observed
3. entropy_min - Minimum entropy
4. entropy_final - Final layer entropy

**Branch count metrics (4)**:
5. branch_count_mean - Average branches
6. branch_count_max - Peak branch count
7. branch_count_min - Minimum branches
8. branch_count_final - Final branches

**Correlation metrics (4)**:
9. scrambling_score - Probability uniformity
10. correlation_mean - Average correlation
11. correlation_max - Peak correlation
12. correlation_min - Minimum correlation

**System parameters (4)**:
13. layer_count - Total logged layers
14. execution_time - Runtime
15. system_size - Number of qubits
16. depth - Circuit depth"""
        },
        {
            'instruction': "How do I run the full Discovery Suite?",
            'output': """Run all three discovery experiments:

```bash
python experiments/discovery_suite.py
```

This runs:

**Experiment A: Time-Reversal Instability (TRI)**
- Tests bias reversal sensitivity
- Quantifies directional fragility

**Experiment B: Unsupervised Clustering**
- K-means on raw features (no labels)
- Discovers emergent phase structure

**Experiment C: Replay Drift Scaling (RSI)**
- Measures feature drift vs depth
- Characterizes complexity growth

Output: Console reports + potential artifacts in ./artifacts/

Customization: Edit discovery_suite.py to adjust phases, sample counts, depth ranges."""
        },
        {
            'instruction': "What is QPR-R?",
            'output': """QPR-R: Quantum Phase Recognition with Replay

**Traditional Model (Hard)**:
- Only measurement access to quantum states
- Phase recognition is exponentially hard
- Limited to output statistics

**QPR-R Model (Efficient)**:
- Full logging of internal state evolution
- Deterministic replay capability
- Polynomial-time phase recognition
- Feature extraction scales as O(layers × features)

**Key Insight**: By allowing synthetic logging and replay, we bypass hardness assumptions from real quantum hardware constraints.

**Practical Meaning**: In Ben Lab, we can log branch probabilities, correlations, and scrambling at every layer - impossible with real quantum computers - enabling efficient phase classification."""
        },
        {
            'instruction': "How does the Quantum Approximation Layer work?",
            'output': """Quantum Approximation Layer workflow:

**1. Spawn**: Create branches from base state
   - Generates N variations with equal amplitudes
   - Each branch has unique phase
   - Example: protein folding explores conformations

**2. Interfere**: Adjust amplitudes via scoring
   - Evaluate each branch with scoring function
   - Update amplitudes based on scores
   - Normalizes to maintain probability = 1

**3. Collapse**: Select top-k branches
   - Weighted selection by probability
   - Blend states of top branches
   - Returns final collapsed state

**Determinism**: Uses fixed seeds for reproducibility

**Use in PhaseDetector**: Simulates quantum phase evolution with branch/interference dynamics."""
        }
    ]


def main():
    corpus_dir = Path("lab_corpus")
    output_file = Path("lab_training_data.jsonl")
    
    if not corpus_dir.exists():
        print(f"Error: {corpus_dir} not found")
        print("Please create lab_corpus/ and add documentation files.")
        return
    
    all_pairs = []
    
    # Process each markdown file
    for md_file in corpus_dir.glob("*.md"):
        print(f"Processing {md_file.name}...")
        content = md_file.read_text()
        sections = extract_sections(content, md_file.stem)
        pairs = generate_qa_pairs(sections, md_file.stem)
        all_pairs.extend(pairs)
        print(f"  Generated {len(pairs)} Q&A pairs from {len(sections)} sections")
    
    # Add practical questions
    practical = generate_practical_qa()
    all_pairs.extend(practical)
    print(f"Added {len(practical)} practical Q&A pairs")
    
    # Write JSONL
    with open(output_file, 'w') as f:
        for pair in all_pairs:
            json.dump(pair, f)
            f.write('\n')
    
    print(f"\n✅ Generated {len(all_pairs)} training pairs")
    print(f"📄 Output: {output_file}")
    print(f"\nNext steps:")
    print(f"1. Review {output_file} for quality")
    print(f"2. Use this dataset to fine-tune a base model (qwen2.5:1.5b, llama3.2, phi3)")
    print(f"3. Convert fine-tuned model to GGUF")
    print(f"4. Create Ollama Modelfile (see ollama/Modelfile.example)")
    print(f"5. Run: ollama create ben-lab -f ollama/Modelfile")


if __name__ == "__main__":
    main()
