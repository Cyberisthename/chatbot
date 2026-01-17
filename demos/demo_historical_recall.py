#!/usr/bin/env python3
"""
JARVIS HISTORICAL RECALL DEMO
=============================
Demonstrates Jarvis's infinite historical memory system.

Shows how to:
1. Query historical scientific knowledge (1800-1950)
2. Route questions to appropriate adapters
3. Load TCL-compressed knowledge seeds
4. Synthesize answers from multiple historical sources
"""

import json
from pathlib import Path
from typing import List, Dict, Any
import sys

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.adapter_engine import AdapterEngine, Adapter


class JarvisHistoricalRecallDemo:
    """Demo of Jarvis's trained historical knowledge system"""
    
    def __init__(self):
        # Load config
        config = self._load_config()
        
        # Initialize adapter engine
        self.engine = AdapterEngine(config)
        
        # Load training report
        self.training_info = self._load_training_report()
        
        print("=" * 80)
        print("🧠 JARVIS HISTORICAL RECALL SYSTEM")
        print("=" * 80)
        print(f"📚 Knowledge Base Loaded:")
        print(f"   • {self.training_info['statistics']['books_processed']} historical books (1800-1950)")
        print(f"   • {self.training_info['statistics']['adapters_created']} persistent adapters")
        print(f"   • {self.training_info['statistics']['tcl_seeds_generated']} TCL knowledge seeds")
        print(f"   • Training: {self.training_info['epochs']} epochs complete")
        print("=" * 80)
        print()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load JARVIS config"""
        config_path = Path("config.json")
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        
        # Default config
        return {
            "adapters": {
                "storage_path": "./adapters",
                "graph_path": "./adapters_graph.json",
                "auto_create": True,
                "freeze_after_creation": False
            },
            "bits": {"y_bits": 16, "z_bits": 8, "x_bits": 8}
        }
    
    def _load_training_report(self) -> Dict[str, Any]:
        """Load training report with statistics"""
        report_path = Path("jarvis_historical_knowledge/TRAINING_REPORT.json")
        if report_path.exists():
            with open(report_path, 'r') as f:
                return json.load(f)
        return {'statistics': {}, 'epochs': 0}
    
    def ask(self, question: str) -> Dict[str, Any]:
        """
        Ask Jarvis a historical question and get results from trained adapters
        """
        print(f"❓ Question: {question}")
        print()
        
        # Route to adapters
        print("🔍 Searching historical knowledge base...")
        adapters = self.engine.route_task(
            question,
            {'features': ['recall_only', 'historical_knowledge']}
        )
        
        if not adapters:
            print("⚠️  No relevant historical knowledge found.")
            return {'adapters': [], 'sources': []}
        
        print(f"✅ Found {len(adapters)} relevant knowledge adapter(s)")
        print()
        
        # Gather sources from adapters
        sources = []
        for i, adapter in enumerate(adapters, 1):
            print(f"📚 Adapter {i}: {adapter.id}")
            print(f"   Topics: {', '.join(adapter.task_tags)}")
            
            # Get era info
            era_range = adapter.parameters.get('era_range', (0, 0))
            if era_range[0] > 0:
                print(f"   Era: {era_range[0]}-{era_range[1]}")
            
            # Count books
            book_params = [k for k in adapter.parameters if k.startswith('book_')]
            print(f"   Historical Sources: {len(book_params)} books")
            
            # Show sample sources
            print(f"   Sample Knowledge:")
            for rule in adapter.rules[:3]:
                print(f"      • {rule[:90]}...")
            
            # Collect book info
            for book_param in book_params[:5]:  # Show first 5
                book_info = adapter.parameters[book_param]
                sources.append({
                    'title': book_info['title'],
                    'author': book_info['author'],
                    'year': book_info['year'],
                    'tcl_seed': book_info['tcl_seed_path'],
                    'symbols': book_info['symbol_count']
                })
            
            print()
        
        # Show unique sources
        print("📖 Historical Sources Referenced:")
        unique_sources = {}
        for src in sources:
            key = f"{src['title']}_{src['author']}_{src['year']}"
            if key not in unique_sources:
                unique_sources[key] = src
        
        for i, src in enumerate(sorted(unique_sources.values(), key=lambda x: x['year']), 1):
            print(f"   {i}. \"{src['title']}\" by {src['author']} ({src['year']})")
            print(f"      └─ {src['symbols']} TCL symbols available from: {src['tcl_seed']}")
        
        print()
        
        return {
            'question': question,
            'adapters': [a.id for a in adapters],
            'sources': list(unique_sources.values())
        }
    
    def inspect_tcl_seed(self, seed_path: str):
        """Load and display a TCL knowledge seed"""
        path = Path(seed_path)
        if not path.exists():
            print(f"❌ TCL seed not found: {seed_path}")
            return
        
        with open(path, 'r') as f:
            seed = json.load(f)
        
        print(f"🧬 TCL Knowledge Seed: {path.name}")
        print(f"   Book: \"{seed['title']}\"")
        print(f"   Author: {seed['author']}")
        print(f"   Year: {seed['year']}")
        print(f"   Subjects: {', '.join(seed['subjects'])}")
        print(f"   Compressed Symbols: {seed['symbol_count']}")
        print(f"   Compression Ratio: {seed['compression_ratio']:.8f}")
        print(f"   (Original size ÷ {int(1/seed['compression_ratio'])} = {seed['symbol_count']} symbols)")
        print()
        print(f"   Symbol Preview: {', '.join(seed['symbols'][:15])}")
        if len(seed['symbols']) > 15:
            print(f"                   ... and {len(seed['symbols']) - 15} more symbols")
        print()
    
    def run_interactive_demo(self):
        """Run interactive demo with pre-set questions"""
        questions = [
            "What did 19th century doctors think about cancer cures?",
            "How did early quantum physicists explain radiation?",
            "What were the key discoveries in cell biology before 1900?",
            "How was evolution theory developed in the 1800s?",
            "What did Victorian medicine know about disease pathology?",
        ]
        
        for i, question in enumerate(questions, 1):
            print(f"\n{'='*80}")
            print(f"QUERY {i}/{len(questions)}")
            print(f"{'='*80}\n")
            
            result = self.ask(question)
            
            # Show first TCL seed if available
            if result['sources']:
                print(f"{'─'*80}")
                print("TCL SEED EXAMPLE")
                print(f"{'─'*80}\n")
                self.inspect_tcl_seed(result['sources'][0]['tcl_seed'])
            
            input("Press Enter to continue...")
    
    def custom_query(self):
        """Allow custom historical queries"""
        print("\n" + "="*80)
        print("CUSTOM QUERY MODE")
        print("="*80)
        print("Ask Jarvis about any historical scientific topic (1800-1950)")
        print("Topics: Quantum physics, medicine, cancer, evolution, cell biology, etc.")
        print("Type 'quit' to exit")
        print()
        
        while True:
            try:
                question = input("❓ Your question: ").strip()
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                if not question:
                    continue
                
                print()
                result = self.ask(question)
                
                if result['sources']:
                    print(f"\n{'─'*80}")
                    view_seed = input("View a TCL seed? (y/n): ").strip().lower()
                    if view_seed == 'y':
                        print(f"{'─'*80}\n")
                        self.inspect_tcl_seed(result['sources'][0]['tcl_seed'])
                
                print()
                
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Jarvis Historical Recall Demo - Query 150 years of scientific knowledge"
    )
    parser.add_argument(
        '--mode',
        choices=['demo', 'interactive', 'query'],
        default='demo',
        help='Demo mode: pre-set questions, interactive: custom queries, query: single query'
    )
    parser.add_argument(
        '--question',
        type=str,
        help='Single question to ask (use with --mode query)'
    )
    
    args = parser.parse_args()
    
    # Initialize demo system
    demo = JarvisHistoricalRecallDemo()
    
    if args.mode == 'demo':
        demo.run_interactive_demo()
    elif args.mode == 'interactive':
        demo.custom_query()
    elif args.mode == 'query':
        if args.question:
            demo.ask(args.question)
        else:
            print("❌ Please provide a --question argument")
            return 1
    
    print("\n" + "="*80)
    print("✅ Demo Complete - Jarvis's historical knowledge is ready!")
    print("="*80)
    print()
    print("📚 Knowledge Base Location: ./jarvis_historical_knowledge/")
    print("🔧 Adapter Storage: ./adapters/")
    print("📊 Training Report: ./jarvis_historical_knowledge/TRAINING_REPORT.json")
    print()
    print("To extend knowledge, run: python3 jarvis_historical_training_pipeline.py")
    print()


if __name__ == "__main__":
    main()
