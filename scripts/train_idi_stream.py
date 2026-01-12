#!/usr/bin/env python3
"""
Institutional Data Initiative (IDI) Streaming Trainer
Streams books from HuggingFace datasets and converts them into adapters + memory

IMPORTANT: This uses streaming mode to avoid downloading the full dataset
"""

import os
import sys
import json
import time
import argparse
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("⚠️  Warning: 'datasets' library not installed. Install with: pip install datasets")

import yaml
from src.core.adapter_engine import AdapterEngine, Adapter


class IDIStreamingTrainer:
    """Streams books from IDI dataset and creates adapters + memory entries"""
    
    def __init__(self, config_path: str = "./config.yaml"):
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize adapter engine
        self.adapter_engine = AdapterEngine(self.config)
        
        # Memory file
        self.memory_file = Path(self.config.get('memory', {}).get('file', './jarvis_memory.json'))
        self.memory = self._load_memory()
        
        # IDI metadata
        self.idi_metadata_file = Path("./idi_training_metadata.json")
        self.idi_metadata = self._load_idi_metadata()
    
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
            "idi_sources": []  # Track IDI sources
        }
    
    def _save_memory(self):
        """Save memory to disk"""
        with open(self.memory_file, 'w') as f:
            json.dump(self.memory, f, indent=2)
    
    def _load_idi_metadata(self) -> Dict[str, Any]:
        """Load IDI training metadata"""
        if self.idi_metadata_file.exists():
            try:
                with open(self.idi_metadata_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                pass
        return {
            "books_processed": [],
            "total_chunks": 0,
            "total_adapters": 0,
            "last_updated": None
        }
    
    def _save_idi_metadata(self):
        """Save IDI metadata"""
        self.idi_metadata["last_updated"] = time.time()
        with open(self.idi_metadata_file, 'w') as f:
            json.dump(self.idi_metadata, f, indent=2)
    
    def chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 128) -> List[str]:
        """
        Chunk text with overlap for better context preservation
        Uses word boundaries to avoid cutting mid-word
        """
        words = text.split()
        chunks = []
        
        i = 0
        while i < len(words):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
            i += chunk_size - overlap
        
        return chunks
    
    def infer_domain_from_text(self, text: str, title: str = "") -> List[str]:
        """Infer domain/topic from text content"""
        text_lower = (text + " " + title).lower()
        domains = []
        
        # Define domain keywords
        domain_keywords = {
            "science": ["physics", "chemistry", "biology", "experiment", "theory", "hypothesis"],
            "mathematics": ["equation", "theorem", "proof", "calculate", "algebra", "geometry"],
            "history": ["century", "historical", "ancient", "medieval", "revolution", "era"],
            "literature": ["novel", "poetry", "story", "narrative", "character", "plot"],
            "philosophy": ["philosophy", "ethics", "logic", "metaphysics", "epistemology"],
            "technology": ["computer", "software", "programming", "algorithm", "digital"],
            "medicine": ["medical", "health", "disease", "treatment", "diagnosis", "patient"],
            "law": ["legal", "court", "justice", "statute", "law", "regulation"],
            "economics": ["economy", "market", "trade", "finance", "economic", "business"],
            "art": ["painting", "sculpture", "artist", "aesthetic", "creative", "gallery"],
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                domains.append(domain)
        
        # Default to "general" if no specific domain detected
        if not domains:
            domains.append("general")
        
        return domains[:3]  # Limit to top 3 domains
    
    def create_y_bits_from_domains(self, domains: List[str]) -> List[int]:
        """Create Y-bits (task/domain bits) from domain tags"""
        y_bits = [0] * 16
        
        domain_bit_map = {
            "science": 0,
            "mathematics": 1,
            "quantum": 2,
            "programming": 3,
            "technology": 3,
            "history": 4,
            "literature": 5,
            "philosophy": 6,
            "medicine": 7,
            "law": 8,
            "economics": 9,
            "art": 10,
            "general": 15,
        }
        
        for domain in domains:
            bit_pos = domain_bit_map.get(domain.lower(), 15)
            y_bits[bit_pos] = 1
        
        return y_bits
    
    def process_book(self, book: Dict[str, Any], chunk_size: int = 512) -> int:
        """
        Process a single book: chunk text, create adapters, add to memory
        Returns number of adapters created
        """
        title = book.get('title', 'Unknown')
        author = book.get('author', 'Unknown')
        text = book.get('text', '')
        
        if not text or len(text) < 100:
            print(f"  ⏭️  Skipping '{title}' (too short)")
            return 0
        
        print(f"  📖 Processing: {title} by {author}")
        
        # Check if already processed
        book_id = f"{title}::{author}"
        if book_id in self.idi_metadata["books_processed"]:
            print(f"  ✓  Already processed, skipping")
            return 0
        
        # Chunk the text
        chunks = self.chunk_text(text, chunk_size=chunk_size)
        print(f"    Split into {len(chunks)} chunks")
        
        # Infer domains
        domains = self.infer_domain_from_text(text, title)
        print(f"    Detected domains: {', '.join(domains)}")
        
        # Create Y-bits
        y_bits = self.create_y_bits_from_domains(domains)
        
        # Create adapters from chunks
        adapters_created = 0
        parent_adapter_id = None
        
        for i, chunk in enumerate(chunks):
            # Create adapter for this chunk
            adapter = self.adapter_engine.create_adapter(
                task_tags=["idi", "knowledge", *domains],
                y_bits=y_bits,
                z_bits=[0] * 8,  # Default difficulty
                x_bits=[0] * 8,  # No experimental flags
                parameters={
                    "source": "IDI",
                    "book_title": title,
                    "book_author": author,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "domains": domains,
                    "text_preview": chunk[:200]  # Store preview for reference
                },
                parent_ids=[parent_adapter_id] if parent_adapter_id else []
            )
            
            # Link chunks in sequence
            if parent_adapter_id:
                self.adapter_engine.adapter_graph.add_dependency(parent_adapter_id, adapter.id, weight=1.0)
            
            parent_adapter_id = adapter.id
            adapters_created += 1
        
        # Add to memory
        memory_entry = f"Read '{title}' by {author} (IDI source, {len(chunks)} sections, domains: {', '.join(domains)})"
        if memory_entry not in self.memory["facts"]:
            self.memory["facts"].append(memory_entry)
            if "idi_sources" not in self.memory:
                self.memory["idi_sources"] = []
            self.memory["idi_sources"].append({
                "title": title,
                "author": author,
                "domains": domains,
                "chunk_count": len(chunks),
                "timestamp": time.time()
            })
        
        # Update metadata
        self.idi_metadata["books_processed"].append(book_id)
        self.idi_metadata["total_chunks"] += len(chunks)
        self.idi_metadata["total_adapters"] += adapters_created
        
        print(f"    ✅ Created {adapters_created} adapters")
        
        return adapters_created
    
    def stream_and_train(self, max_books: int = 100, language: str = "en", min_length: int = 1000):
        """
        Stream books from IDI dataset and train adapters
        
        Args:
            max_books: Maximum number of books to process
            language: Language filter (default: English)
            min_length: Minimum text length to process
        """
        if not HF_AVAILABLE:
            print("❌ Cannot run IDI streaming without 'datasets' library")
            print("   Install with: pip install datasets")
            return
        
        print(f"\n🚀 Starting IDI Streaming Trainer")
        print(f"   Max books: {max_books}")
        print(f"   Language: {language}")
        print(f"   Min length: {min_length} chars")
        print(f"\n⚠️  Using streaming mode - no full dataset download\n")
        
        try:
            # Load dataset in streaming mode
            print("📡 Connecting to HuggingFace datasets...")
            dataset = load_dataset(
                "institutional/institutional-books-1.0",
                streaming=True,
                split="train"
            )
            
            print("✅ Connected! Starting to stream books...\n")
            
            books_processed = 0
            adapters_created = 0
            
            # Stream and process books
            for book in dataset:
                if books_processed >= max_books:
                    break
                
                # Filter by language if specified
                if language and book.get('language', '').lower() != language.lower():
                    continue
                
                # Filter by minimum length
                text = book.get('text', '')
                if len(text) < min_length:
                    continue
                
                # Process book
                num_adapters = self.process_book(book)
                
                if num_adapters > 0:
                    books_processed += 1
                    adapters_created += num_adapters
                    
                    # Save progress periodically
                    if books_processed % 10 == 0:
                        self._save_memory()
                        self._save_idi_metadata()
                        print(f"\n💾 Progress saved: {books_processed} books, {adapters_created} adapters\n")
            
            # Final save
            self._save_memory()
            self._save_idi_metadata()
            
            print(f"\n✅ IDI Training Complete!")
            print(f"   Books processed: {books_processed}")
            print(f"   Adapters created: {adapters_created}")
            print(f"   Metadata saved to: {self.idi_metadata_file}")
            
        except Exception as e:
            print(f"❌ Error during streaming: {e}")
            import traceback
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Stream and train from Institutional Data Initiative (IDI) dataset"
    )
    parser.add_argument(
        "--max-books",
        type=int,
        default=100,
        help="Maximum number of books to process (default: 100)"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language filter (default: en for English)"
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=1000,
        help="Minimum text length to process (default: 1000 chars)"
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
    
    # Create trainer
    trainer = IDIStreamingTrainer(config_path=args.config)
    
    # Run streaming training
    trainer.stream_and_train(
        max_books=args.max_books,
        language=args.language,
        min_length=args.min_length
    )


if __name__ == "__main__":
    main()
