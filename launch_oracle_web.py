import os
import sys
from pathlib import Path

# Add chatbot to path
sys.path.insert(0, str(Path(__file__).parent))

# Install requirements
print("Checking requirements...")
os.system(f"{sys.executable} -m pip install -q --user gradio requests tqdm numpy matplotlib --break-system-packages")

from jarvis_v1_gradio_space import create_gradio_app

if __name__ == "__main__":
    print("🚀 Launching Jarvis v1 Quantum Oracle for Owner Testing...")
    
    app = create_gradio_app()
    
    # Launch on port 7861 as requested
    app.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=True # Enable sharing so the owner can access it via the public URL
    )
