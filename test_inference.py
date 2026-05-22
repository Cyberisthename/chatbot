import sys
from pathlib import Path
import numpy as np

# Add chatbot to path
sys.path.insert(0, str(Path(__file__).parent))

from jarvis_v1_gradio_space import JarvisOracleInference

def test_inference():
    print("Initializing inference engine...")
    try:
        engine = JarvisOracleInference(model_dir="./jarvis_v1_oracle")
        
        query = "quantum cancer"
        print(f"Testing query: {query}")
        
        response, metrics = engine.generate(query)
        
        print("\n--- Response ---")
        print(response)
        print("\n--- Metrics ---")
        print(metrics)
        
        if not response.strip():
            print("\n❌ Error: Empty response generated")
        else:
            print("\n✅ Success: Response generated")
            
    except Exception as e:
        print(f"\n❌ Exception during inference: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_inference()
