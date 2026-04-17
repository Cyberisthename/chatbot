import json
import os
import sys
import uuid
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.thought_compression.tcl_engine import ThoughtCompressionEngine
from src.core.adapter_engine import AdapterEngine, AdapterStatus

def create_prototype_adapter():
    print("🚀 Developing Prototype Knowledge Adapter for Modern Research...")
    
    # 1. Load Modern Research Data
    data_path = Path("/home/agent-engineer/modern_research_data.json")
    with open(data_path, "r") as f:
        modern_data = json.load(f)
    
    # 2. Initialize TCL Engine
    print("🗜️ Compressing concepts with TCL Engine...")
    tcl_engine = ThoughtCompressionEngine(enable_quantum_mode=True)
    session_id = tcl_engine.create_session("modern_research_prototype")
    
    # 3. Process data with TCL
    processed_items = []
    for item in modern_data:
        # Compress title and text
        input_text = f"{item['title']}. {item['text']}"
        compressed = tcl_engine.compress_concept(session_id, input_text)
        
        # Calculate reasoning enhancement
        enhancement = tcl_engine.enhance_reasoning(session_id, item['title'])
        
        processed_items.append({
            "original": item,
            "compressed_symbols": compressed['compressed_symbols'],
            "compression_ratio": compressed['compression_ratio'],
            "enhanced_solutions": enhancement['enhanced_solutions']
        })
        print(f"   Processed: {item['title']}")

    # 4. Create Adapter via AdapterEngine
    print("🔌 Creating Knowledge Adapter...")
    adapter_config = {
        "adapters": {
            "storage_path": "./jarvis_v1_oracle/adapters", 
            "graph_path": "./jarvis_v1_oracle/adapter_graph.json"
        },
        "bits": {"y_bits": 16, "z_bits": 8, "x_bits": 8}
    }
    adapter_engine = AdapterEngine(adapter_config)
    
    # Select the first item for the prototype adapter
    target = processed_items[0]
    
    # Generate bit patterns (Modern/Scientific)
    y_bits = [0] * 16
    y_bits[2] = 1  # scientific domain
    y_bits[4] = 1  # modern research bit (custom assignment)
    
    z_bits = [0] * 8
    z_bits[1] = 1  # high complexity
    
    x_bits = [0] * 8
    x_bits[0] = 1  # quantum enabled
    
    # Create the adapter
    adapter = adapter_engine.create_adapter(
        task_tags=["modern", "quantum", "error-correction"],
        y_bits=y_bits,
        z_bits=z_bits,
        x_bits=x_bits,
        parameters={
            "book_title": target['original']['title'],
            "author": target['original']['author'],
            "tcl_compression_ratio": target['compression_ratio'],
            "enhanced_solutions": target['enhanced_solutions'],
            "is_modern_prototype": True
        }
    )
    
    # Persist to disk with the format expected by JarvisOracleInference
    final_data = {
        "adapter": adapter.to_dict(),
        "book_title": target['original']['title'],
        "tcl_compression_ratio": target['compression_ratio'],
        "is_modern_prototype": True
    }
    
    adapter_path = Path(adapter_config["adapters"]["storage_path"]) / f"{adapter.id}.json"
    with open(adapter_path, 'w') as f:
        json.dump(final_data, f, indent=2)
    
    print(f"✅ Prototype Adapter Created: {adapter.id}")
    print(f"   Path: ./jarvis_v1_oracle/adapters/{adapter.id}.json")
    
    return adapter.id

if __name__ == "__main__":
    create_prototype_adapter()
