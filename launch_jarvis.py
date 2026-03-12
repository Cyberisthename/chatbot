#!/usr/bin/env python3
"""
JARVIS Launcher Script
Simple entry point to start the JARVIS Assistant
"""

import sys
import os
from pathlib import Path

# Ensure we're in the project directory
project_dir = Path(__file__).parent
os.chdir(project_dir)

# Add src to Python path
sys.path.insert(0, str(project_dir / "src"))

def check_dependencies():
    """Check if required dependencies are installed"""
    missing = []
    
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
    
    try:
        import gradio
    except ImportError:
        missing.append("gradio")
    
    if missing:
        print("❌ Missing dependencies:")
        for dep in missing:
            print(f"   - {dep}")
        print("\nInstall with: pip install " + " ".join(missing))
        return False
    
    return True

def main():
    """Main launcher"""
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║                                                                ║
    ║          🚀 J.A.R.V.I.S. LAUNCH SEQUENCE                       ║
    ║                                                                ║
    ╚════════════════════════════════════════════════════════════════╝
    """)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    print("✅ Dependencies verified")
    print("📦 Loading JARVIS modules...")
    
    try:
        # Import and launch
        from jarvis_assistant import create_jarvis_ui
        
        print("🎨 Initializing user interface...")
        demo = create_jarvis_ui()
        
        print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║                                                                ║
    ║          ✅ J.A.R.V.I.S. IS READY                              ║
    ║                                                                ║
    ║          🌐 Open your browser to:                              ║
    ║             http://localhost:7860                              ║
    ║                                                                ║
    ║          💡 Features Available:                                ║
    ║             • Neural Chat with Quantum LLM                     ║
    ║             • Thought-Compression Language                     ║
    ║             • Cancer Research Tools                            ║
    ║             • Multiversal Protein Folding                      ║
    ║             • Quantum Experiments                              ║
    ║             • Adapter Management                               ║
    ║                                                                ║
    ╚════════════════════════════════════════════════════════════════╝
        """)
        
        # Launch the app
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True,
            quiet=False,
            inbrowser=True
        )
        
    except KeyboardInterrupt:
        print("\n\n👋 JARVIS shutting down. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error launching JARVIS: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
