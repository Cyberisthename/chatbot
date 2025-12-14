#!/usr/bin/env python3
"""
Knowledge Ingestion for JARVIS-2v
RAG-style ingestion that adds to memory without modifying adapters

Use this for quick facts, snippets, or small knowledge additions
For larger documents, use train_adapters.py instead
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class KnowledgeIngestor:
    """Ingest knowledge into memory system"""
    
    def __init__(self, memory_file: str = "./jarvis_memory.json"):
        self.memory_file = Path(memory_file)
        self.memory = self._load_memory()
    
    def _load_memory(self) -> Dict[str, Any]:
        """Load memory from disk"""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                pass
        return {
            "facts": [],
            "chats": [],
            "topics": {},
            "preferences": {},
            "last_topics": [],
            "knowledge_base": []  # Dedicated knowledge entries
        }
    
    def _save_memory(self):
        """Save memory to disk"""
        with open(self.memory_file, 'w') as f:
            json.dump(self.memory, f, indent=2)
    
    def ingest_fact(self, fact: str, tags: List[str] = None):
        """Ingest a single fact"""
        if not fact.strip():
            return
        
        # Check for duplicates
        if fact in self.memory["facts"]:
            print(f"  ⚠️  Fact already exists, skipping")
            return
        
        # Add to facts
        self.memory["facts"].append(fact)
        
        # Add to knowledge base with metadata
        self.memory.setdefault("knowledge_base", []).append({
            "content": fact,
            "tags": tags or [],
            "timestamp": time.time(),
            "source": "ingestion"
        })
        
        # Update topics
        for tag in (tags or []):
            self.memory["topics"][tag] = self.memory["topics"].get(tag, 0) + 1
        
        print(f"  ✅ Ingested: {fact[:80]}...")
    
    def ingest_from_text(self, text: str, tags: List[str] = None):
        """Ingest facts from text (one per line)"""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        print(f"\n📥 Ingesting {len(lines)} facts...")
        
        for line in lines:
            self.ingest_fact(line, tags)
        
        self._save_memory()
        print(f"\n✅ Ingestion complete! Memory saved to {self.memory_file}")
    
    def ingest_from_file(self, filepath: Path, tags: List[str] = None):
        """Ingest facts from file"""
        print(f"\n📄 Reading from: {filepath}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            
            self.ingest_from_text(text, tags)
        
        except Exception as e:
            print(f"❌ Error reading file: {e}")
    
    def ingest_json_facts(self, filepath: Path):
        """Ingest from JSON file with structure"""
        print(f"\n📄 Reading JSON from: {filepath}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Support different JSON structures
            if isinstance(data, list):
                # List of strings or objects
                for item in data:
                    if isinstance(item, str):
                        self.ingest_fact(item)
                    elif isinstance(item, dict):
                        fact = item.get('fact') or item.get('content') or str(item)
                        tags = item.get('tags', [])
                        self.ingest_fact(fact, tags)
            
            elif isinstance(data, dict):
                # Dictionary with 'facts' key
                if 'facts' in data:
                    for fact in data['facts']:
                        self.ingest_fact(fact)
                else:
                    # Treat each key-value as a fact
                    for key, value in data.items():
                        fact = f"{key}: {value}"
                        self.ingest_fact(fact)
            
            self._save_memory()
            print(f"\n✅ JSON ingestion complete!")
        
        except Exception as e:
            print(f"❌ Error reading JSON: {e}")
    
    def show_stats(self):
        """Show memory statistics"""
        print(f"\n📊 Memory Statistics:")
        print(f"   Total facts: {len(self.memory.get('facts', []))}")
        print(f"   Knowledge base entries: {len(self.memory.get('knowledge_base', []))}")
        print(f"   Topics tracked: {len(self.memory.get('topics', {}))}")
        print(f"   Chat history: {len(self.memory.get('chats', []))}")
        
        if self.memory.get('topics'):
            print(f"\n   Top topics:")
            sorted_topics = sorted(
                self.memory['topics'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            for topic, count in sorted_topics:
                print(f"     - {topic}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description="Ingest knowledge into JARVIS-2v memory"
    )
    parser.add_argument(
        "--fact",
        type=str,
        help="Single fact to ingest"
    )
    parser.add_argument(
        "--file",
        type=str,
        help="File to ingest (text or JSON)"
    )
    parser.add_argument(
        "--tags",
        type=str,
        nargs="+",
        help="Tags to associate with facts"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show memory statistics"
    )
    parser.add_argument(
        "--memory",
        type=str,
        default="./jarvis_memory.json",
        help="Path to memory file (default: ./jarvis_memory.json)"
    )
    
    args = parser.parse_args()
    
    # Create ingestor
    ingestor = KnowledgeIngestor(memory_file=args.memory)
    
    # Show stats
    if args.stats:
        ingestor.show_stats()
        return
    
    # Ingest single fact
    if args.fact:
        ingestor.ingest_fact(args.fact, args.tags)
        ingestor._save_memory()
        print(f"\n✅ Fact ingested!")
        return
    
    # Ingest from file
    if args.file:
        filepath = Path(args.file)
        if not filepath.exists():
            print(f"❌ File not found: {filepath}")
            return
        
        if filepath.suffix == '.json':
            ingestor.ingest_json_facts(filepath)
        else:
            ingestor.ingest_from_file(filepath, args.tags)
        return
    
    # No action specified
    print("⚠️  No action specified. Use --fact, --file, or --stats")
    parser.print_help()


if __name__ == "__main__":
    main()
