#!/usr/bin/env python3
"""
Local Adapter Trainer for JARVIS-2v
Converts local files into adapters and memory entries

Supports: .txt, .md, .json, .csv
NON-DESTRUCTIVE: Only adds new adapters, never overwrites
"""

import os
import sys
import json
import csv
import time
import argparse
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from src.core.adapter_engine import AdapterEngine, Adapter


class LocalAdapterTrainer:
    """Train adapters from local files"""
    
    def __init__(self, config_path: str = "./config.yaml"):
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize adapter engine
        self.adapter_engine = AdapterEngine(self.config)
        
        # Memory file
        self.memory_file = Path(self.config.get('memory', {}).get('file', './jarvis_memory.json'))
        self.memory = self._load_memory()
        
        # Training metadata
        self.training_metadata_file = Path("./local_training_metadata.json")
        self.training_metadata = self._load_training_metadata()
    
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
            "local_sources": []
        }
    
    def _save_memory(self):
        """Save memory to disk"""
        with open(self.memory_file, 'w') as f:
            json.dump(self.memory, f, indent=2)
    
    def _load_training_metadata(self) -> Dict[str, Any]:
        """Load training metadata"""
        if self.training_metadata_file.exists():
            try:
                with open(self.training_metadata_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                pass
        return {
            "files_processed": [],
            "total_adapters": 0,
            "last_updated": None
        }
    
    def _save_training_metadata(self):
        """Save training metadata"""
        self.training_metadata["last_updated"] = time.time()
        with open(self.training_metadata_file, 'w') as f:
            json.dump(self.training_metadata, f, indent=2)
    
    def chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 128) -> List[str]:
        """Chunk text with overlap"""
        words = text.split()
        chunks = []
        
        i = 0
        while i < len(words):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():  # Only add non-empty chunks
                chunks.append(chunk)
            i += chunk_size - overlap
        
        return chunks
    
    def infer_domains(self, text: str, filename: str = "") -> List[str]:
        """Infer domains from text and filename"""
        combined = (text + " " + filename).lower()
        domains = []
        
        domain_keywords = {
            "programming": ["code", "function", "class", "variable", "programming", "python", "javascript"],
            "science": ["experiment", "theory", "hypothesis", "research", "study"],
            "mathematics": ["equation", "theorem", "proof", "calculate", "formula"],
            "quantum": ["quantum", "qubit", "entanglement", "superposition"],
            "documentation": ["readme", "guide", "tutorial", "documentation", "manual"],
            "configuration": ["config", "settings", "setup", "parameters"],
            "data": ["dataset", "data", "csv", "json", "table"],
        }
        
        for domain, keywords in domain_keywords.items():
            if any(kw in combined for kw in keywords):
                domains.append(domain)
        
        if not domains:
            domains.append("general")
        
        return domains[:3]
    
    def create_y_bits(self, domains: List[str]) -> List[int]:
        """Create Y-bits from domains"""
        y_bits = [0] * 16
        
        domain_bit_map = {
            "programming": 0,
            "mathematics": 1,
            "quantum": 2,
            "science": 3,
            "documentation": 4,
            "configuration": 5,
            "data": 6,
            "general": 15,
        }
        
        for domain in domains:
            bit_pos = domain_bit_map.get(domain.lower(), 15)
            y_bits[bit_pos] = 1
        
        return y_bits
    
    def read_file(self, filepath: Path) -> Optional[str]:
        """Read file content"""
        try:
            if filepath.suffix == '.csv':
                # Convert CSV to text representation
                with open(filepath, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    text = f"CSV file with {len(rows)} rows\n"
                    text += f"Columns: {', '.join(rows[0].keys() if rows else [])}\n"
                    # Add sample rows
                    for row in rows[:10]:
                        text += json.dumps(row) + "\n"
                    return text
            
            elif filepath.suffix == '.json':
                # Convert JSON to text
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return json.dumps(data, indent=2)
            
            else:
                # Plain text (txt, md, etc.)
                with open(filepath, 'r', encoding='utf-8') as f:
                    return f.read()
        
        except Exception as e:
            print(f"  ⚠️  Error reading {filepath}: {e}")
            return None
    
    def process_file(self, filepath: Path, chunk_size: int = 512) -> int:
        """
        Process a single file: chunk, create adapters
        Returns number of adapters created
        """
        print(f"  📄 Processing: {filepath.name}")
        
        # Check if already processed
        file_id = str(filepath.absolute())
        if file_id in self.training_metadata["files_processed"]:
            print(f"    ✓  Already processed, skipping")
            return 0
        
        # Read file
        text = self.read_file(filepath)
        if not text or len(text) < 50:
            print(f"    ⏭️  Skipping (too short or empty)")
            return 0
        
        # Chunk text
        chunks = self.chunk_text(text, chunk_size=chunk_size)
        print(f"    Split into {len(chunks)} chunks")
        
        # Infer domains
        domains = self.infer_domains(text, filepath.name)
        print(f"    Detected domains: {', '.join(domains)}")
        
        # Create Y-bits
        y_bits = self.create_y_bits(domains)
        
        # Create adapters
        adapters_created = 0
        parent_adapter_id = None
        
        for i, chunk in enumerate(chunks):
            adapter = self.adapter_engine.create_adapter(
                task_tags=["local", filepath.suffix[1:], *domains],
                y_bits=y_bits,
                z_bits=[0] * 8,
                x_bits=[0] * 8,
                parameters={
                    "source": "local",
                    "filename": filepath.name,
                    "filepath": str(filepath),
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "domains": domains,
                    "text_preview": chunk[:200]
                },
                parent_ids=[parent_adapter_id] if parent_adapter_id else []
            )
            
            if parent_adapter_id:
                self.adapter_engine.adapter_graph.add_dependency(parent_adapter_id, adapter.id, weight=1.0)
            
            parent_adapter_id = adapter.id
            adapters_created += 1
        
        # Add to memory
        memory_entry = f"Learned from file '{filepath.name}' ({len(chunks)} sections, domains: {', '.join(domains)})"
        if memory_entry not in self.memory["facts"]:
            self.memory["facts"].append(memory_entry)
            if "local_sources" not in self.memory:
                self.memory["local_sources"] = []
            self.memory["local_sources"].append({
                "filename": filepath.name,
                "domains": domains,
                "chunk_count": len(chunks),
                "timestamp": time.time()
            })
        
        # Update metadata
        self.training_metadata["files_processed"].append(file_id)
        self.training_metadata["total_adapters"] += adapters_created
        
        print(f"    ✅ Created {adapters_created} adapters")
        
        return adapters_created
    
    def train_from_directory(self, directory: Path, recursive: bool = True, extensions: List[str] = None):
        """
        Train from all files in directory
        
        Args:
            directory: Directory to scan
            recursive: Whether to scan subdirectories
            extensions: List of extensions to process (default: .txt, .md, .json, .csv)
        """
        if extensions is None:
            extensions = ['.txt', '.md', '.json', '.csv']
        
        print(f"\n🚀 Training from: {directory}")
        print(f"   Recursive: {recursive}")
        print(f"   Extensions: {', '.join(extensions)}\n")
        
        # Find files
        if recursive:
            files = [f for f in directory.rglob('*') if f.is_file() and f.suffix in extensions]
        else:
            files = [f for f in directory.glob('*') if f.is_file() and f.suffix in extensions]
        
        print(f"📁 Found {len(files)} files to process\n")
        
        if not files:
            print("⚠️  No files found to process")
            return
        
        files_processed = 0
        adapters_created = 0
        
        for filepath in files:
            num_adapters = self.process_file(filepath)
            
            if num_adapters > 0:
                files_processed += 1
                adapters_created += num_adapters
                
                # Save progress periodically
                if files_processed % 5 == 0:
                    self._save_memory()
                    self._save_training_metadata()
                    print(f"\n💾 Progress saved: {files_processed} files, {adapters_created} adapters\n")
        
        # Final save
        self._save_memory()
        self._save_training_metadata()
        
        print(f"\n✅ Local Training Complete!")
        print(f"   Files processed: {files_processed}")
        print(f"   Adapters created: {adapters_created}")
        print(f"   Metadata saved to: {self.training_metadata_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Train adapters from local files"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="./data/raw",
        help="Input directory (default: ./data/raw)"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="Scan subdirectories recursively (default: True)"
    )
    parser.add_argument(
        "--extensions",
        type=str,
        nargs="+",
        default=[".txt", ".md", ".json", ".csv"],
        help="File extensions to process (default: .txt .md .json .csv)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Chunk size in words (default: 512)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./config.yaml",
        help="Path to config file (default: ./config.yaml)"
    )
    
    args = parser.parse_args()
    
    # Check if input directory exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Directory not found: {input_path}")
        print(f"   Creating directory: {input_path}")
        input_path.mkdir(parents=True, exist_ok=True)
    
    # Create trainer
    trainer = LocalAdapterTrainer(config_path=args.config)
    
    # Run training
    trainer.train_from_directory(
        directory=input_path,
        recursive=args.recursive,
        extensions=args.extensions
    )


if __name__ == "__main__":
    main()
