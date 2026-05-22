import json
import os
import sys
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from ingestion.research_fetcher import ResearchFetcher
from thought_compression.tcl_engine import ThoughtCompressionEngine
from thought_compression.tcl_types import Evidence, TCLMetadata, HyperEdge
from core.adapter_engine import AdapterEngine

class IngestionPipeline:
    """
    Automated pipeline for fetching research and creating Knowledge Adapters
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        self.fetcher = ResearchFetcher()
        self.tcl_engine = ThoughtCompressionEngine(enable_quantum_mode=True)
        
        # Determine paths relative to repository root
        self.repo_root = Path(__file__).parent.parent.parent
        self.tcl_state_path = self.repo_root / "jarvis_v1_oracle" / "tcl_graph.json"
        
        # Load existing state if available
        if self.tcl_state_path.exists():
            print(f"📂 Loading TCL state from {self.tcl_state_path}")
            self.tcl_engine.load_state(str(self.tcl_state_path))
        
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = self.repo_root / "jarvis_v1_oracle" / "adapters"
            
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.adapter_config = {
            "adapters": {
                "storage_path": str(self.output_dir), 
                "graph_path": str(self.repo_root / "jarvis_v1_oracle" / "adapter_graph.json")
            },
            "bits": {"y_bits": 16, "z_bits": 8, "x_bits": 8}
        }
        self.adapter_engine = AdapterEngine(self.adapter_config)
        
    def run(self, query: str, source: str = "both", max_results: int = 5):
        """
        Run the ingestion pipeline
        """
        print(f"🚀 Starting Ingestion Pipeline for: '{query}'")
        
        papers = []
        if source in ["arxiv", "both"]:
            print(f"📡 Fetching from arXiv...")
            papers.extend(self.fetcher.fetch_arxiv(query, max_results=max_results))
            
        if source in ["pubmed", "both"]:
            print(f"📡 Fetching from PubMed...")
            papers.extend(self.fetcher.fetch_pubmed(query, max_results=max_results))
            
        print(f"📚 Found {len(papers)} papers to process")
        
        session_id = self.tcl_engine.create_session(f"ingestion_{uuid.uuid4().hex[:8]}")
        
        created_adapters = []
        for paper in papers:
            try:
                adapter_id = self._process_paper(paper, session_id)
                created_adapters.append(adapter_id)
            except Exception as e:
                print(f"❌ Failed to process paper {paper['id']}: {e}")
                
        # Finalize
        self.tcl_engine.refresh_metrics(session_id)
        self.tcl_engine.save_state(str(self.tcl_state_path))
        self._update_adapter_links()

        print(f"✨ Ingestion complete. Created {len(created_adapters)} adapters.")
        return created_adapters

    def _update_adapter_links(self):
        """Link adapters in the graph based on shared TCL concepts"""
        print("🔗 Updating adapter graph links based on TCL hypergraph...")
        # Map symbol IDs to adapter IDs
        symbol_to_adapters = {}
        for adapter_file in self.output_dir.glob("*.json"):
            with open(adapter_file, 'r') as f:
                data = json.load(f)
                adapter_id = data['adapter'].get('id')
                # In our pipeline, we store the source title in book_title
                # We can find symbols that were created for this adapter if we tracked them
                # For now, let's use a simple name matching between symbols and adapter titles
                title = data.get('book_title', '').lower()
                for sym_id, symbol in self.tcl_engine.global_symbols.symbols.items():
                    if symbol.name.lower() in title or title in symbol.definition.lower():
                        if sym_id not in symbol_to_adapters:
                            symbol_to_adapters[sym_id] = []
                        symbol_to_adapters[sym_id].append(adapter_id)

        # Use HyperEdges to connect adapters
        for edge in self.tcl_engine.global_symbols.hyperedges.values():
            participating_adapters = set()
            for node_id in edge.nodes:
                if node_id in symbol_to_adapters:
                    participating_adapters.update(symbol_to_adapters[node_id])
            
            # Connect all participating adapters
            adapter_list = list(participating_adapters)
            for i, a1 in enumerate(adapter_list):
                for a2 in adapter_list[i+1:]:
                    self.adapter_engine.adapter_graph.add_dependency(a1, a2, weight=edge.weight)

    def _process_paper(self, paper: Dict[str, Any], session_id: str) -> str:
        """
        Process a single paper: TCL compression -> Adapter creation -> Storage
        """
        print(f"  Processing: {paper['title'][:50]}...")
        context = self.tcl_engine.sessions[session_id]
        
        # 1. TCL Compression
        input_text = f"{paper['title']}. {paper.get('summary', 'No summary')}"
        # We need to know which symbols were newly created/referenced
        existing_symbol_ids = set(context.symbols.symbols.keys())
        compressed = self.tcl_engine.compress_concept(session_id, input_text)
        new_symbol_ids = set(context.symbols.symbols.keys()) - existing_symbol_ids
        
        # 1b. Enrich with TCL 2.0 Evidence and Metadata
        paper_evidence = Evidence(
            source=paper['source'],
            confidence=0.9,
            timestamp=paper['published'],
            metadata={"id": paper['id'], "link": paper['link']}
        )
        
        domain = "general"
        if "quant" in paper['title'].lower() or "quantum" in paper.get('summary', '').lower():
            domain = "quantum_physics"
        elif "bio" in paper['title'].lower() or "cell" in paper.get('summary', '').lower():
            domain = "biology"
        elif "ai" in paper['title'].lower() or "neural" in paper.get('summary', '').lower():
            domain = "ai"

        paper_metadata = TCLMetadata(
            domain=domain,
            tags=set(paper.get('authors', [])[:2])
        )

        for sym_id in new_symbol_ids:
            symbol = context.symbols.symbols[sym_id]
            symbol.evidence.append(paper_evidence)
            symbol.metadata = paper_metadata

        # 1c. Create HyperEdge for the paper (linking all its concepts)
        if len(new_symbol_ids) >= 2:
            hyper_edge = HyperEdge(
                id=f"paper_{paper['id'].replace('/', '_')}",
                nodes=list(new_symbol_ids),
                weight=0.8,
                edge_type="co-occurrence",
                evidence=[paper_evidence],
                metadata=paper_metadata
            )
            context.symbols.add_hyperedge(hyper_edge)

        # 2. Define Bit Patterns
        enhancement = self.tcl_engine.enhance_reasoning(session_id, paper['title'])
        # ... rest of the bit logic remains same
        y_bits = [0] * 16
        y_bits[2] = 1  # scientific domain
        if paper['source'] == "arXiv":
            y_bits[5] = 1 # arXiv bit
        else:
            y_bits[6] = 1 # PubMed bit
            
        z_bits = [0] * 8
        z_bits[1] = 1  # high complexity
        
        x_bits = [0] * 8
        x_bits[0] = 1  # quantum enabled
        
        # 3. Create Adapter
        adapter = self.adapter_engine.create_adapter(
            task_tags=["modern", paper['source'].lower(), "automated-ingestion"],
            y_bits=y_bits,
            z_bits=z_bits,
            x_bits=x_bits,
            parameters={
                "book_title": paper['title'],
                "author": ", ".join(paper['authors'][:3]),
                "source": paper['source'],
                "source_id": paper['id'],
                "tcl_compression_ratio": compressed['compression_ratio'],
                "enhanced_solutions": enhancement['enhanced_solutions'],
                "published_date": paper['published'],
                "link": paper['link'],
                "is_automated": True
            }
        )
        
        # 4. Save to disk in the format expected by the Oracle
        final_data = {
            "adapter": adapter.to_dict(),
            "book_title": paper['title'],
            "tcl_compression_ratio": compressed['compression_ratio'],
            "is_automated": True,
            "source": paper['source']
        }
        
        adapter_path = self.output_dir / f"{adapter.id}.json"
        with open(adapter_path, 'w') as f:
            json.dump(final_data, f, indent=2)
            
        # Update Oracle config
        self._update_oracle_config()
            
        return adapter.id

    def _update_oracle_config(self):
        """Update num_adapters in Oracle config"""
        # Try to find the oracle directory relative to the repository root
        repo_root = Path(__file__).parent.parent.parent
        config_path = repo_root / "jarvis_v1_oracle" / "huggingface_export" / "config.json"
        
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # Count actual adapters
                adapter_count = len(list(self.output_dir.glob("*.json")))
                config["num_adapters"] = adapter_count
                
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                print(f"  Updated Oracle config: num_adapters={adapter_count}")
            except Exception as e:
                print(f"  Warning: Could not update Oracle config: {e}")
        else:
            print(f"  Warning: Config not found at {config_path}")

if __name__ == "__main__":
    pipeline = IngestionPipeline()
    # Test with a few modern AI and Quantum topics
    pipeline.run("cat:cs.AI", source="arxiv", max_results=2)
    pipeline.run("Quantum Computing", source="pubmed", max_results=2)
