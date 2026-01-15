#!/usr/bin/env python3
"""
🌌 Hugging Face Spaces App: Quantum Time-Entangled Cancer Cure Demo

Deploy this to Hugging Face Spaces for public access.

Run locally:
    python app.py

Deploy to Spaces:
    1. Create a new Space with "Gradio" SDK
    2. Upload these files
    3. Add to requirements.txt: gradio, numpy, matplotlib
    4. Space will auto-deploy
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from gradio_quantum_cancer_demo import create_interface

if __name__ == "__main__":
    print("\n🌌 Launching Quantum Time-Entangled Cancer Cure Demo")
    print("   Hugging Face Spaces Edition\n")
    
    demo = create_interface()
    
    # For Hugging Face Spaces
    demo.launch()
