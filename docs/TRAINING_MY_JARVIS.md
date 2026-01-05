# Training Your JARVIS-2v: Complete Guide

## Overview

JARVIS-2v implements a three-layer training system that converts your documents into a personalized AI assistant:

1. **Layer A**: RAG (Retrieval-Augmented Generation) - Data → Knowledge Index
2. **Layer B**: Adapter Training - Domain-specific adapters from your documents
3. **Layer C**: Optional LoRA Fine-tuning - Custom model weights (if needed)

## Quick Start

### 1. Prepare Your Training Data

Place your documents in one of these directories:

- `./training-data/` - Core training data (recommended)
- `./data/raw/` - Additional raw data

**Supported file formats**: `.txt`, `.md`, `.pdf`, `.json`, `.csv`

Example structure:
```
training-data/
├── my_code_notes.md
├── project_docs.json
└── research_papers.pdf
```

### 2. Run Training

```bash
# Train from training-data directory
python scripts/train_adapters.py --input ./training-data --profile standard

# Train from custom directory
python scripts/train_adapters.py --input ./my_docs --profile standard

# Available profiles:
# - low_power: For embedded/Jetson devices
# - standard: For desktop/laptop usage  
# - jetson_orin: Optimized for Jetson Orin NX
```

### 3. Verify Training Results

Check the knowledge base:
```bash
# Test knowledge base functionality
python test_kb.py

# View training report
cat kb_adapters/training_report.json
```

## Layer A: Knowledge Base (RAG)

### How It Works

1. **Document Processing**: Files are chunked into overlapping segments (500 chars, 50 overlap)
2. **Embedding Generation**: Text is converted to vector embeddings using TF-IDF
3. **Vector Storage**: Stored in FAISS index for fast similarity search
4. **Retrieval**: Relevant chunks are retrieved and added to chat context

### API Endpoints

- `POST /kb/ingest` - Ingest single file
- `POST /kb/ingest/directory` - Ingest all files from directory
- `POST /kb/search` - Search knowledge base
- `GET /kb/stats` - Get knowledge base statistics
- `POST /kb/context` - Get contextual text for queries

### Configuration

```yaml
knowledge_base:
  data_path: "./data"
  chunk_size: 500
  chunk_overlap: 50
  embedding_dim: 384
  max_context_chars: 2000
```

## Layer B: Adapter Training

### How It Works

1. **Domain Analysis**: Documents are analyzed to determine primary domain (programming, mathematics, science, etc.)
2. **Bit Pattern Generation**: Y/Z/X bits are generated based on domain, complexity, and features
3. **Lesson Creation**: Domain-specific "lessons" are extracted from document chunks
4. **Adapter Creation**: Specialized adapters are created with routing patterns

### Bit System

- **Y-bits (16)**: Task/domain classification
  - Bits 0-7: Primary domains
  - Bits 8-15: Specific topics
  
- **Z-bits (8)**: Complexity/precision indicators
  - Bit 0: High complexity
  - Bit 1: Very high complexity
  - Bit 2: Expert level
  
- **X-bits (8)**: Feature toggles
  - Bit 0: Code generation
  - Bit 1: Problem solving
  - Bit 2: Explanation
  - etc.

### Training Output

Each adapter includes:
- Domain classification
- Task patterns
- Example prompts/responses
- Confidence scores
- Routing metadata

## Layer C: Optional LoRA Fine-tuning

**Only implement this if Layer A + B are insufficient for your use case.**

### When to Consider LoRA

- Domain requires deep understanding beyond RAG retrieval
- Need for consistent style/voice adaptation
- Complex reasoning tasks requiring custom model weights

### Implementation

If needed, implement using:
- Small open-weight base model (compatible with GGUF)
- Your domain-specific training data
- LoRA adapter training
- Optional merge into base model

**Note**: This requires significant computational resources and careful evaluation.

## Verification Checklist

### Basic Functionality Test

1. **Knowledge Base**: Upload a document and query it
   ```bash
   curl -X POST "http://localhost:3001/kb/ingest" \
        -H "Content-Type: application/json" \
        -d '{"file_path": "./test_doc.txt"}'
   ```

2. **Search**: Verify search returns relevant results
   ```bash
   curl -X POST "http://localhost:3001/kb/search" \
        -H "Content-Type: application/json" \
        -d '{"query": "your search query", "top_k": 3}'
   ```

3. **Chat with RAG**: Test that chat uses your knowledge
   - Ask questions about your uploaded documents
   - Verify responses reference specific content

4. **Adapters**: Check that domain-specific adapters were created
   ```bash
   curl "http://localhost:3001/adapters"
   ```

### Performance Tests

1. **Query Relevance**: Test various query types
   - Specific facts from your documents
   - Conceptual questions
   - Technical queries

2. **Adapter Routing**: Verify Y/Z/X bits route to appropriate adapters
3. **Memory Persistence**: Check that learned patterns persist across sessions
4. **Scalability**: Test with larger document collections

## Troubleshooting

### Common Issues

**No documents ingested:**
- Check file formats are supported
- Verify file paths are correct
- Check file permissions

**Poor search results:**
- Increase chunk size for better context
- Reduce overlap if redundant results
- Check embedding quality

**Adapters not routing correctly:**
- Review bit pattern generation
- Check domain analysis accuracy
- Verify adapter creation

**LLM not using knowledge base:**
- Check KB context is being retrieved
- Verify chat endpoint integration
- Check model context length limits

### Performance Optimization

**For larger datasets:**
- Use FAISS for faster search
- Increase embedding dimension for better recall
- Implement document caching

**For faster training:**
- Reduce chunk size for more parallel processing
- Use batch processing for embeddings
- Implement incremental learning

## Advanced Usage

### Custom Domain Analysis

Modify `scripts/train_adapters.py` to add custom domain keywords:

```python
self.domain_keywords = {
    "your_domain": ["keyword1", "keyword2", ...],
    # ... existing domains
}
```

### Bit Pattern Customization

Adjust bit generation in `BitPatternGenerator`:

```python
# Modify bit assignments based on your needs
def generate_bits(self, domain, topics, complexity, features):
    # Custom bit generation logic
    return y_bits, z_bits, x_bits
```

### Integration with External Systems

JARVIS-2v can be integrated with:

- **Vector Databases**: Replace FAISS with Milvus, Pinecone, etc.
- **LLM APIs**: Use OpenAI, Anthropic, or other APIs instead of local models
- **Knowledge Graphs**: Enhance with Neo4j or similar
- **Document Stores**: Connect to Elasticsearch, MongoDB, etc.

## Next Steps

After training:

1. **Deploy**: Use deployment guides for production
2. **Monitor**: Track performance and usage patterns
3. **Iterate**: Add more data and retrain based on usage
4. **Expand**: Add new domains and capabilities as needed

For detailed deployment instructions, see `docs/DEPLOYMENT.md`.