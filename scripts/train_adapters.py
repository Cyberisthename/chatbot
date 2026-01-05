"""
Adapter Training System for JARVIS-2v
Converts user documents into domain-specific adapters with Y/Z/X bit patterns
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import hashlib
import time

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.core.knowledge_base import KnowledgeBase
from src.core.adapter_engine import AdapterEngine, Adapter


@dataclass
class AdapterLesson:
    """Adapter lesson derived from document analysis"""
    domain: str
    topic: str
    task_patterns: List[str]
    y_bits: List[int]
    z_bits: List[int]
    x_bits: List[int]
    key_concepts: List[str]
    example_prompts: List[str]
    responses: List[str]
    confidence_score: float
    source_chunks: List[str]
    parameters: Dict[str, Any]


class DomainAnalyzer:
    """Analyzes documents to extract domain information"""
    
    def __init__(self):
        self.domain_keywords = {
            "programming": [
                "code", "function", "class", "variable", "algorithm", "debug", "compile",
                "python", "javascript", "java", "c++", "api", "framework", "library",
                "script", "loop", "condition", "object", "method", "return"
            ],
            "mathematics": [
                "equation", "formula", "calculation", "number", "integer", "float",
                "algebra", "geometry", "calculus", "statistics", "probability",
                "matrix", "vector", "function", "derivative", "integral"
            ],
            "science": [
                "experiment", "hypothesis", "theory", "observation", "data",
                "research", "analysis", "quantum", "physics", "chemistry", "biology",
                "laboratory", "measurement", "variable", "control"
            ],
            "technology": [
                "system", "network", "database", "server", "client", "protocol",
                "security", "encryption", "authentication", "performance",
                "architecture", "infrastructure", "deployment", "monitoring"
            ],
            "general": [
                "explain", "describe", "what", "how", "why", "when", "where",
                "information", "help", "assist", "support", "guide"
            ]
        }
        
        self.topic_extractors = {
            "programming": self._extract_programming_topics,
            "mathematics": self._extract_math_topics,
            "science": self._extract_science_topics,
            "technology": self._extract_tech_topics,
            "general": self._extract_general_topics
        }
    
    def analyze_document(self, content: str) -> Dict[str, Any]:
        """Analyze document content and extract domain information"""
        content_lower = content.lower()
        
        # Calculate domain scores
        domain_scores = {}
        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in content_lower)
            domain_scores[domain] = score
        
        # Determine primary domain
        primary_domain = max(domain_scores, key=domain_scores.get)
        if domain_scores[primary_domain] == 0:
            primary_domain = "general"
        
        # Extract topics using domain-specific methods
        topics = self.topic_extractors[primary_domain](content)
        
        # Calculate complexity (Z-bits indicators)
        complexity_indicators = [
            len(content.split()),  # Length
            content.count('\n'),   # Structure
            content.count(';'),    # Code indicators
            content.count('=='),   # Logic indicators
            content.count('('),    # Function indicators
        ]
        
        complexity_score = sum(1 for indicator in complexity_indicators if indicator > 5)
        
        return {
            "primary_domain": primary_domain,
            "domain_scores": domain_scores,
            "topics": topics,
            "complexity_score": complexity_score,
            "word_count": len(content.split())
        }
    
    def _extract_programming_topics(self, content: str) -> List[str]:
        """Extract programming-specific topics"""
        topics = []
        content_lower = content.lower()
        
        if any(word in content_lower for word in ["function", "def", "return"]):
            topics.append("functions")
        if any(word in content_lower for word in ["class", "object", "method"]):
            topics.append("oop")
        if any(word in content_lower for word in ["if", "else", "switch"]):
            topics.append("conditionals")
        if any(word in content_lower for word in ["for", "while", "loop"]):
            topics.append("loops")
        if any(word in content_lower for word in ["api", "endpoint", "request"]):
            topics.append("api_development")
        if any(word in content_lower for word in ["database", "sql", "query"]):
            topics.append("database")
        
        return topics if topics else ["general_programming"]
    
    def _extract_math_topics(self, content: str) -> List[str]:
        """Extract mathematics-specific topics"""
        topics = []
        content_lower = content.lower()
        
        if any(word in content_lower for word in ["algebra", "equation", "solve"]):
            topics.append("algebra")
        if any(word in content_lower for word in ["geometry", "angle", "triangle"]):
            topics.append("geometry")
        if any(word in content_lower for word in ["calculus", "derivative", "integral"]):
            topics.append("calculus")
        if any(word in content_lower for word in ["statistics", "mean", "average"]):
            topics.append("statistics")
        if any(word in content_lower for word in ["probability", "chance", "random"]):
            topics.append("probability")
        
        return topics if topics else ["general_math"]
    
    def _extract_science_topics(self, content: str) -> List[str]:
        """Extract science-specific topics"""
        topics = []
        content_lower = content.lower()
        
        if any(word in content_lower for word in ["quantum", "physics", "particle"]):
            topics.append("quantum_physics")
        if any(word in content_lower for word in ["chemistry", "reaction", "molecule"]):
            topics.append("chemistry")
        if any(word in content_lower for word in ["biology", "cell", "dna"]):
            topics.append("biology")
        if any(word in content_lower for word in ["experiment", "hypothesis", "observation"]):
            topics.append("scientific_method")
        
        return topics if topics else ["general_science"]
    
    def _extract_tech_topics(self, content: str) -> List[str]:
        """Extract technology-specific topics"""
        topics = []
        content_lower = content.lower()
        
        if any(word in content_lower for word in ["network", "server", "client"]):
            topics.append("networking")
        if any(word in content_lower for word in ["security", "encryption", "authentication"]):
            topics.append("security")
        if any(word in content_lower for word in ["database", "sql", "storage"]):
            topics.append("database_systems")
        if any(word in content_lower for word in ["performance", "optimization", "speed"]):
            topics.append("performance")
        
        return topics if topics else ["general_technology"]
    
    def _extract_general_topics(self, content: str) -> List[str]:
        """Extract general topics"""
        # Simple keyword-based topic extraction
        common_topics = []
        content_lower = content.lower()
        
        if any(word in content_lower for word in ["explain", "how", "why"]):
            common_topics.append("explanation")
        if any(word in content_lower for word in ["help", "assist", "guide"]):
            common_topics.append("assistance")
        if any(word in content_lower for word in ["tutorial", "step", "learn"]):
            common_topics.append("tutorial")
        
        return common_topics if common_topics else ["general_help"]


class BitPatternGenerator:
    """Generates Y/Z/X bit patterns for adapters based on content analysis"""
    
    def __init__(self, y_size: int = 16, z_size: int = 8, x_size: int = 8):
        self.y_size = y_size
        self.z_size = z_size
        self.x_size = x_size
        
        # Y-bit domain mappings
        self.domain_y_bits = {
            "programming": 0,
            "mathematics": 1,
            "science": 2,
            "technology": 3,
            "general": 4
        }
        
        # X-bit feature mappings
        self.feature_x_bits = {
            "code_generation": 0,
            "problem_solving": 1,
            "explanation": 2,
            "analysis": 3,
            "tutorial": 4
        }
    
    def generate_bits(self, domain: str, topics: List[str], complexity: int, 
                     features: List[str]) -> Tuple[List[int], List[int], List[int]]:
        """Generate Y/Z/X bit patterns"""
        
        # Y-bits: domain and topic classification
        y_bits = [0] * self.y_size
        if domain in self.domain_y_bits:
            y_bits[self.domain_y_bits[domain]] = 1
        
        # Add topic-specific Y-bits
        topic_offset = 8
        for i, topic in enumerate(topics[:8]):  # Limit to 8 topics
            topic_hash = hash(topic) % 8
            y_bits[topic_offset + topic_hash] = 1
        
        # Z-bits: complexity and precision indicators
        z_bits = [0] * self.z_size
        z_bits[0] = 1 if complexity > 2 else 0  # High complexity
        z_bits[1] = 1 if complexity > 5 else 0  # Very high complexity
        z_bits[2] = 1 if complexity > 10 else 0  # Expert level
        
        # X-bits: experimental and feature toggles
        x_bits = [0] * self.x_size
        for feature in features:
            if feature in self.feature_x_bits:
                x_bits[self.feature_x_bits[feature]] = 1
        
        return y_bits, z_bits, x_bits


class AdapterTrainer:
    """Main adapter training system"""
    
    def __init__(self, kb_config: Dict[str, Any], adapter_config: Dict[str, Any]):
        self.kb_config = kb_config
        self.adapter_config = adapter_config
        self.kb = KnowledgeBase(kb_config)
        self.domain_analyzer = DomainAnalyzer()
        self.bit_generator = BitPatternGenerator(
            y_size=adapter_config.get("y_bits", 16),
            z_size=adapter_config.get("z_bits", 8),
            x_size=adapter_config.get("x_bits", 8)
        )
        self.adapter_engine = AdapterEngine(adapter_config)
    
    def train_from_training_data(self, profile: str = "standard") -> List[str]:
        """Train adapters from training-data/ directory"""
        print(f"🚀 Training adapters from training-data/ with profile: {profile}")
        
        # Ingest training data
        chunk_ids = self.kb.ingest_directory("./training-data")
        print(f"✓ Ingested {len(chunk_ids)} chunks from training data")
        
        # Analyze chunks and create lessons
        lessons = self._create_lessons_from_chunks(chunk_ids)
        
        # Train adapters from lessons
        adapter_ids = self._train_adapters(lessons, profile)
        
        return adapter_ids
    
    def train_from_raw_data(self, profile: str = "standard") -> List[str]:
        """Train adapters from data/raw/ directory"""
        print(f"🚀 Training adapters from data/raw/ with profile: {profile}")
        
        # Ingest raw data
        chunk_ids = self.kb.ingest_directory("./data/raw")
        print(f"✓ Ingested {len(chunk_ids)} chunks from raw data")
        
        # Analyze chunks and create lessons
        lessons = self._create_lessons_from_chunks(chunk_ids)
        
        # Train adapters from lessons
        adapter_ids = self._train_adapters(lessons, profile)
        
        return adapter_ids
    
    def _create_lessons_from_chunks(self, chunk_ids: List[str]) -> List[AdapterLesson]:
        """Create adapter lessons from document chunks"""
        lessons = []
        
        # Group chunks by domain
        domain_groups = {}
        
        for chunk_id in chunk_ids:
            chunk = self.kb.vector_index.chunk_map.get(chunk_id)
            if not chunk:
                continue
            
            # Analyze chunk content
            analysis = self.domain_analyzer.analyze_document(chunk.content)
            domain = analysis["primary_domain"]
            
            if domain not in domain_groups:
                domain_groups[domain] = []
            domain_groups[domain].append((chunk, analysis))
        
        # Create lessons per domain
        for domain, chunk_data in domain_groups.items():
            lesson = self._create_domain_lesson(domain, chunk_data)
            if lesson:
                lessons.append(lesson)
        
        print(f"✓ Created {len(lessons)} adapter lessons")
        return lessons
    
    def _create_domain_lesson(self, domain: str, chunk_data: List[Tuple]) -> Optional[AdapterLesson]:
        """Create a lesson for a specific domain"""
        if not chunk_data:
            return None
        
        # Combine content from all chunks in domain
        all_content = " ".join([chunk.content for chunk, _ in chunk_data])
        
        # Analyze combined content
        analysis = self.domain_analyzer.analyze_document(all_content)
        
        # Extract key concepts (simple approach)
        words = all_content.lower().split()
        word_freq = {}
        for word in words:
            if len(word) > 3:  # Skip short words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top concepts
        top_concepts = sorted(word_freq.keys(), key=word_freq.get, reverse=True)[:10]
        
        # Generate example prompts based on domain
        example_prompts = self._generate_example_prompts(domain, top_concepts)
        
        # Generate appropriate responses
        responses = self._generate_responses(domain, top_concepts)
        
        # Generate bit patterns
        features = ["explanation", "assistance"]  # Default features
        y_bits, z_bits, x_bits = self.bit_generator.generate_bits(
            domain, analysis["topics"], analysis["complexity_score"], features
        )
        
        # Calculate confidence score
        confidence = min(analysis["domain_scores"][domain] / 10.0, 1.0)
        
        # Get source chunk IDs
        source_chunks = [chunk.id for chunk, _ in chunk_data]
        
        return AdapterLesson(
            domain=domain,
            topic="_".join(analysis["topics"][:2]),  # Combine top 2 topics
            task_patterns=analysis["topics"],
            y_bits=y_bits,
            z_bits=z_bits,
            x_bits=x_bits,
            key_concepts=top_concepts,
            example_prompts=example_prompts,
            responses=responses,
            confidence_score=confidence,
            source_chunks=source_chunks,
            parameters={
                "domain": domain,
                "topics": analysis["topics"],
                "complexity": analysis["complexity_score"],
                "source": "training_data"
            }
        )
    
    def _generate_example_prompts(self, domain: str, concepts: List[str]) -> List[str]:
        """Generate example prompts based on domain and concepts"""
        prompts = []
        
        if domain == "programming":
            prompts = [
                f"How do I implement {concepts[0] if concepts else 'a function'} in Python?",
                f"Explain {concepts[0] if concepts else 'algorithms'} with examples",
                f"What's the best way to debug {concepts[0] if concepts else 'code'}?",
                f"Help me understand {concepts[0] if concepts else 'object-oriented programming'}"
            ]
        elif domain == "mathematics":
            prompts = [
                f"Explain {concepts[0] if concepts else 'algebraic equations'} step by step",
                f"How do I solve problems involving {concepts[0] if concepts else 'calculus'}?",
                f"What's the formula for {concepts[0] if concepts else 'geometric calculations'}?",
                f"Help me understand {concepts[0] if concepts else 'mathematical concepts'}"
            ]
        elif domain == "science":
            prompts = [
                f"Explain the concept of {concepts[0] if concepts else 'quantum mechanics'}",
                f"How does {concepts[0] if concepts else 'scientific experimentation'} work?",
                f"What are the principles behind {concepts[0] if concepts else 'physics'}?",
                f"Help me understand {concepts[0] if concepts else 'scientific theory'}"
            ]
        else:
            prompts = [
                f"Explain {concepts[0] if concepts else 'this topic'} clearly",
                f"Help me understand {concepts[0] if concepts else 'this concept'}",
                f"What can you tell me about {concepts[0] if concepts else 'this subject'}?",
                f"How do I approach problems related to {concepts[0] if concepts else 'this area'}?"
            ]
        
        return prompts
    
    def _generate_responses(self, domain: str, concepts: List[str]) -> List[str]:
        """Generate example responses based on domain"""
        # These are template responses that would be refined during training
        responses = []
        
        if domain == "programming":
            responses = [
                f"Based on your {domain} knowledge, here's how to approach {concepts[0] if concepts else 'programming problems'}:",
                f"For {concepts[0] if concepts else 'coding tasks'}, I recommend following these best practices:",
                f"When working with {concepts[0] if concepts else 'programming concepts'}, consider these key points:"
            ]
        elif domain == "mathematics":
            responses = [
                f"Here's how to understand {concepts[0] if concepts else 'mathematical concepts'}:",
                f"The solution to {concepts[0] if concepts else 'math problems'} involves these steps:",
                f"For {concepts[0] if concepts else 'mathematical calculations'}, remember these principles:"
            ]
        else:
            responses = [
                f"Based on the information about {concepts[0] if concepts else 'this topic'}:",
                f"Here's what I can explain about {concepts[0] if concepts else 'this subject'}:",
                f"The key concepts you should understand about {concepts[0] if concepts else 'this area'} are:"
            ]
        
        return responses
    
    def _train_adapters(self, lessons: List[AdapterLesson], profile: str) -> List[str]:
        """Train adapters from lessons with specified profile"""
        adapter_ids = []
        
        for lesson in lessons:
            try:
                # Create adapter from lesson
                adapter = self.adapter_engine.create_adapter(
                    task_tags=lesson.task_patterns,
                    y_bits=lesson.y_bits,
                    z_bits=lesson.z_bits,
                    x_bits=lesson.x_bits,
                    parameters={
                        **lesson.parameters,
                        "confidence_score": lesson.confidence_score,
                        "key_concepts": lesson.key_concepts,
                        "profile": profile,
                        "training_source": "knowledge_base"
                    }
                )
                
                # Add lesson-specific metadata
                adapter.parameters["example_prompts"] = lesson.example_prompts
                adapter.parameters["example_responses"] = lesson.responses
                adapter.parameters["lesson_source_chunks"] = lesson.source_chunks
                
                # Update adapter in storage
                self.adapter_engine._save_adapter(adapter)
                self.adapter_engine.adapter_graph.add_adapter(adapter)
                
                adapter_ids.append(adapter.id)
                print(f"✓ Created adapter {adapter.id} for {lesson.domain} domain")
                
            except Exception as e:
                print(f"✗ Error creating adapter for {lesson.domain}: {e}")
        
        return adapter_ids
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        kb_stats = self.kb.get_stats()
        adapter_stats = {
            "total_adapters": len(self.adapter_engine.list_adapters()),
            "active_adapters": len(self.adapter_engine.list_adapters(status="active")),
            "frozen_adapters": len(self.adapter_engine.list_adapters(status="frozen"))
        }
        
        return {
            "knowledge_base": kb_stats,
            "adapters": adapter_stats,
            "training_profiles": ["low_power", "standard", "jetson_orin"]
        }


def main():
    """Main training script"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train JARVIS-2v adapters from documents")
    parser.add_argument("--input", default="./training-data", help="Input directory")
    parser.add_argument("--profile", default="standard", choices=["low_power", "standard", "jetson_orin"])
    parser.add_argument("--output", default="./kb_adapters", help="Output directory for training results")
    
    args = parser.parse_args()
    
    # Configuration
    kb_config = {
        "data_path": "./data",
        "chunk_size": 500,
        "chunk_overlap": 50,
        "embedding_dim": 384
    }
    
    adapter_config = {
        "adapters": {"storage_path": "./adapters", "auto_create": True},
        "bits": {"y_bits": 16, "z_bits": 8, "x_bits": 8}
    }
    
    # Initialize trainer
    trainer = AdapterTrainer(kb_config, adapter_config)
    
    # Run training
    if Path(args.input).exists():
        if args.input == "./training-data":
            adapter_ids = trainer.train_from_training_data(args.profile)
        else:
            # For custom input, ingest and train
            chunk_ids = trainer.kb.ingest_directory(args.input)
            lessons = trainer._create_lessons_from_chunks(chunk_ids)
            adapter_ids = trainer._train_adapters(lessons, args.profile)
        
        print(f"\n🎯 Training completed! Created {len(adapter_ids)} adapters")
        
        # Save training report
        report = {
            "timestamp": time.time(),
            "profile": args.profile,
            "input_directory": args.input,
            "adapter_ids": adapter_ids,
            "stats": trainer.get_training_stats()
        }
        
        Path(args.output).mkdir(parents=True, exist_ok=True)
        with open(f"{args.output}/training_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📊 Training report saved to {args.output}/training_report.json")
        
    else:
        print(f"❌ Input directory {args.input} not found")


if __name__ == "__main__":
    main()