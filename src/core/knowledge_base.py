"""
Knowledge Base Engine for JARVIS-2v
Implements RAG (Retrieval-Augmented Generation) with file ingestion and vector search
"""

import json
import os
import hashlib
import uuid
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import re
import mimetypes

# Try to import FAISS, fallback to sqlite-vec if not available
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("FAISS not available, using sqlite-vec fallback")

import numpy as np


@dataclass
class DocumentChunk:
    """Document chunk with metadata"""
    id: str
    content: str
    source_file: str
    source_type: str  # 'txt', 'md', 'pdf', 'json', 'csv'
    chunk_index: int
    start_char: int
    end_char: int
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = None
    created_at: float = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "content": self.content,
            "source_file": self.source_file,
            "source_type": self.source_type,
            "chunk_index": self.chunk_index,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "embedding": self.embedding,
            "metadata": self.metadata,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentChunk':
        """Create from dictionary"""
        return cls(**data)


class DocumentProcessor:
    """Processes various file formats and extracts text content"""
    
    @staticmethod
    def extract_text_from_file(file_path: Path) -> Tuple[str, str]:
        """Extract text content and determine file type"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_ext = file_path.suffix.lower()
        file_type = file_ext[1:] if file_ext.startswith('.') else 'unknown'
        
        try:
            if file_type == 'txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
            elif file_type == 'md':
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
            elif file_type == 'json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    content = json.dumps(data, indent=2)
                    
            elif file_type == 'csv':
                import csv
                content_lines = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        content_lines.append(','.join(row))
                content = '\n'.join(content_lines)
                
            elif file_type == 'pdf':
                # Try to extract text from PDF
                try:
                    import PyPDF2
                    with open(file_path, 'rb') as f:
                        reader = PyPDF2.PdfReader(f)
                        content = ""
                        for page in reader.pages:
                            content += page.extract_text() + "\n"
                except ImportError:
                    raise ValueError("PyPDF2 not available for PDF processing")
                except Exception:
                    raise ValueError("Could not extract text from PDF")
                    
            else:
                # Fallback: try to read as text
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                except UnicodeDecodeError:
                    with open(file_path, 'r', encoding='latin-1') as f:
                        content = f.read()
                        
            return content, file_type
            
        except Exception as e:
            raise ValueError(f"Error processing file {file_path}: {e}")
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[Tuple[str, int, int]]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            
            # Try to break at sentence or paragraph boundaries
            if end < len(text):
                # Look for sentence endings
                sentence_end = max(
                    text.rfind('. ', start, end),
                    text.rfind('! ', start, end),
                    text.rfind('? ', start, end),
                    text.rfind('\n\n', start, end)
                )
                if sentence_end > start + chunk_size // 2:
                    end = sentence_end + 1
            
            chunk_content = text[start:end].strip()
            if chunk_content:
                chunks.append((chunk_content, start, end))
                chunk_index += 1
            
            # Move start position with overlap
            start = max(start + chunk_size - overlap, end)
            
        return chunks


class EmbeddingManager:
    """Manages text embeddings using simple TF-IDF or fallback methods"""
    
    def __init__(self, vector_dim: int = 384):
        self.vector_dim = vector_dim
        self.vocab = {}
        self.idf_weights = {}
        
    def _simple_embedding(self, text: str) -> List[float]:
        """Generate simple hash-based embedding as fallback"""
        # Simple bag-of-words with hash trick
        embedding = [0.0] * self.vector_dim
        
        words = re.findall(r'\w+', text.lower())
        for word in words:
            # Hash word to fixed dimension
            hash_val = hash(word) % self.vector_dim
            embedding[hash_val] += 1.0
            
        # Normalize
        norm = sum(x * x for x in embedding) ** 0.5
        if norm > 0:
            embedding = [x / norm for x in embedding]
            
        return embedding
    
    def _tfidf_embedding(self, texts: List[str]) -> List[List[float]]:
        """Generate TF-IDF embeddings for a batch of texts"""
        # Build vocabulary
        all_words = set()
        for text in texts:
            words = re.findall(r'\w+', text.lower())
            all_words.update(words)
        
        vocab_list = list(all_words)
        self.vocab = {word: i for i, word in enumerate(vocab_list)}
        
        # Calculate IDF weights
        doc_count = len(texts)
        doc_freq = {word: 0 for word in vocab_list}
        
        for text in texts:
            words_in_doc = set(re.findall(r'\w+', text.lower()))
            for word in words_in_doc:
                doc_freq[word] += 1
        
        self.idf_weights = {
            word: np.log(doc_count / (freq + 1))
            for word, freq in doc_freq.items()
        }
        
        # Generate embeddings
        embeddings = []
        for text in texts:
            words = re.findall(r'\w+', text.lower())
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            
            embedding = [0.0] * self.vector_dim
            for word, count in word_counts.items():
                if word in self.vocab:
                    idx = self.vocab[word] % self.vector_dim
                    tf = count / len(words)
                    idf = self.idf_weights[word]
                    embedding[idx] = tf * idf
            
            # Normalize
            norm = sum(x * x for x in embedding) ** 0.5
            if norm > 0:
                embedding = [x / norm for x in embedding]
                
            embeddings.append(embedding)
        
        return embeddings
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        return self._simple_embedding(text)
    
    def generate_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts"""
        if len(texts) > 1:
            return self._tfidf_embedding(texts)
        else:
            return [self.generate_embedding(texts[0])]


class VectorIndex:
    """Lightweight vector index for similarity search"""
    
    def __init__(self, vector_dim: int = 384, index_path: str = "./vector_index"):
        self.vector_dim = vector_dim
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        # Use FAISS if available, otherwise use simple numpy search
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(vector_dim)  # Inner product
            self.use_faiss = True
        else:
            self.vectors = []
            self.use_faiss = False
        
        self.chunk_ids = []
        self.chunk_map = {}
        
        # Load existing index
        self._load_index()
    
    def add_chunks(self, chunks: List[DocumentChunk]):
        """Add document chunks to the index"""
        if not chunks:
            return
        
        embeddings = [chunk.embedding for chunk in chunks]
        
        if self.use_faiss:
            # Convert to numpy array
            vectors_array = np.array(embeddings, dtype=np.float32)
            self.index.add(vectors_array)
        else:
            # Simple list-based storage
            self.vectors.extend(embeddings)
        
        # Store chunk IDs and mapping
        for chunk in chunks:
            self.chunk_ids.append(chunk.id)
            self.chunk_map[chunk.id] = chunk
        
        self._save_index()
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """Search for similar chunks"""
        if not self.chunk_ids:
            return []
        
        query_vector = np.array([query_embedding], dtype=np.float32)
        
        if self.use_faiss and self.index.ntotal > 0:
            # Use FAISS search
            scores, indices = self.index.search(query_vector, min(top_k, len(self.chunk_ids)))
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1:  # Valid index
                    chunk_id = self.chunk_ids[idx]
                    chunk = self.chunk_map[chunk_id]
                    results.append((chunk, float(score)))
            
            return results
        else:
            # Simple numpy search
            query_vec = np.array(query_embedding)
            similarities = []
            
            for i, vector in enumerate(self.vectors):
                # Cosine similarity
                similarity = np.dot(query_vec, vector) / (
                    np.linalg.norm(query_vec) * np.linalg.norm(vector) + 1e-8
                )
                similarities.append((i, similarity))
            
            # Sort by similarity and return top k
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            results = []
            for i, similarity in similarities[:top_k]:
                chunk_id = self.chunk_ids[i]
                chunk = self.chunk_map[chunk_id]
                results.append((chunk, similarity))
            
            return results
    
    def _save_index(self):
        """Save index to disk"""
        # Save chunk mapping
        chunks_data = {
            chunk_id: chunk.to_dict() if hasattr(chunk, 'to_dict') else asdict(chunk)
            for chunk_id, chunk in self.chunk_map.items()
        }
        
        with open(self.index_path / "chunks.json", 'w') as f:
            json.dump(chunks_data, f, indent=2)
        
        # Save vector index
        if self.use_faiss:
            faiss.write_index(self.index, str(self.index_path / "faiss.index"))
        else:
            np.save(self.index_path / "vectors.npy", self.vectors)
        
        # Save chunk IDs
        with open(self.index_path / "chunk_ids.json", 'w') as f:
            json.dump(self.chunk_ids, f)
    
    def _load_index(self):
        """Load index from disk"""
        try:
            # Load chunk mapping
            chunks_file = self.index_path / "chunks.json"
            if chunks_file.exists():
                with open(chunks_file, 'r') as f:
                    chunks_data = json.load(f)
                    # Reconstruct DocumentChunk objects
                    for chunk_id, chunk_data in chunks_data.items():
                        self.chunk_map[chunk_id] = DocumentChunk(**chunk_data)
                
                self.chunk_ids = list(chunks_data.keys())
            
            # Load vector index
            if self.use_faiss:
                faiss_file = self.index_path / "faiss.index"
                if faiss_file.exists():
                    self.index = faiss.read_index(str(faiss_file))
            else:
                vectors_file = self.index_path / "vectors.npy"
                if vectors_file.exists():
                    self.vectors = np.load(vectors_file).tolist()
                    
        except Exception as e:
            print(f"Warning: Could not load existing index: {e}")


class KnowledgeBase:
    """Main knowledge base class for JARVIS-2v"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_path = Path(config.get("data_path", "./data"))
        self.chunks_path = self.data_path / "processed" / "chunks.jsonl"
        self.vector_index = VectorIndex(
            vector_dim=config.get("embedding_dim", 384),
            index_path=str(self.data_path.parent / "vector_index")
        )
        self.processor = DocumentProcessor()
        self.embedding_manager = EmbeddingManager(
            vector_dim=config.get("embedding_dim", 384)
        )
        
        # Ensure directories exist
        self.data_path.mkdir(parents=True, exist_ok=True)
        (self.data_path / "processed").mkdir(parents=True, exist_ok=True)
    
    def ingest_file(self, file_path: str, metadata: Dict[str, Any] = None) -> List[str]:
        """Ingest a single file and return created chunk IDs"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Extract text and determine type
        content, file_type = self.processor.extract_text_from_file(file_path)
        
        # Chunk the text
        chunks = self.processor.chunk_text(
            content, 
            chunk_size=self.config.get("chunk_size", 500),
            overlap=self.config.get("chunk_overlap", 50)
        )
        
        # Create DocumentChunk objects
        chunk_objects = []
        for i, (chunk_content, start_char, end_char) in enumerate(chunks):
            chunk_id = f"chunk_{hashlib.md5(f'{file_path}_{i}_{start_char}'.encode()).hexdigest()[:12]}"
            
            chunk = DocumentChunk(
                id=chunk_id,
                content=chunk_content,
                source_file=str(file_path),
                source_type=file_type,
                chunk_index=i,
                start_char=start_char,
                end_char=end_char,
                metadata=metadata or {}
            )
            chunk_objects.append(chunk)
        
        # Generate embeddings
        texts = [chunk.content for chunk in chunk_objects]
        embeddings = self.embedding_manager.generate_batch_embeddings(texts)
        
        for chunk, embedding in zip(chunk_objects, embeddings):
            chunk.embedding = embedding
        
        # Add to vector index
        self.vector_index.add_chunks(chunk_objects)
        
        # Save to chunks file
        self._save_chunks(chunk_objects)
        
        return [chunk.id for chunk in chunk_objects]
    
    def ingest_directory(self, directory_path: str, patterns: List[str] = None) -> List[str]:
        """Ingest all matching files from a directory"""
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        # Default patterns for supported file types
        if patterns is None:
            patterns = ["*.txt", "*.md", "*.pdf", "*.json", "*.csv"]
        
        all_chunk_ids = []
        
        for pattern in patterns:
            for file_path in directory_path.glob(pattern):
                if file_path.is_file():
                    try:
                        chunk_ids = self.ingest_file(
                            str(file_path),
                            metadata={"ingested_from": str(directory_path), "pattern": pattern}
                        )
                        all_chunk_ids.extend(chunk_ids)
                        print(f"✓ Ingested {len(chunk_ids)} chunks from {file_path}")
                    except Exception as e:
                        print(f"✗ Error ingesting {file_path}: {e}")
        
        return all_chunk_ids
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """Search for relevant chunks given a query"""
        query_embedding = self.embedding_manager.generate_embedding(query)
        results = self.vector_index.search(query_embedding, top_k)
        
        # Add query context to metadata
        for chunk, score in results:
            chunk.metadata["query"] = query
            chunk.metadata["relevance_score"] = score
        
        return results
    
    def get_context_for_query(self, query: str, max_chars: int = 2000) -> str:
        """Get contextual text for a query"""
        results = self.search(query, top_k=3)
        
        context_parts = []
        total_chars = 0
        
        for chunk, score in results:
            if total_chars + len(chunk.content) > max_chars:
                # Truncate chunk if needed
                remaining = max_chars - total_chars
                if remaining > 100:  # Only include if substantial content
                    chunk_text = chunk.content[:remaining] + "..."
                else:
                    break
            else:
                chunk_text = chunk.content
            
            context_parts.append(f"[Source: {Path(chunk.source_file).name}, Score: {score:.3f}]\n{chunk_text}")
            total_chars += len(chunk_text)
            
            if total_chars >= max_chars:
                break
        
        return "\n\n".join(context_parts)
    
    def _save_chunks(self, chunks: List[DocumentChunk]):
        """Save chunks to JSONL file"""
        with open(self.chunks_path, 'a') as f:
            for chunk in chunks:
                f.write(json.dumps(chunk.to_dict()) + '\n')
    
    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        chunks_count = len(self.vector_index.chunk_ids)
        
        # Count files by type
        file_types = {}
        for chunk in self.vector_index.chunk_map.values():
            file_type = chunk.source_type
            file_types[file_type] = file_types.get(file_type, 0) + 1
        
        return {
            "total_chunks": chunks_count,
            "file_types": file_types,
            "vector_dim": self.vector_index.vector_dim,
            "using_faiss": self.vector_index.use_faiss,
            "chunks_file": str(self.chunks_path)
        }


__all__ = ["KnowledgeBase", "DocumentProcessor", "EmbeddingManager", "VectorIndex", "DocumentChunk"]