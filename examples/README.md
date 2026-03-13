# Audio-to-OpenRAG Examples

Minimal, video-friendly demos showing how to use Docling and OpenRAG SDK for audio transcription and knowledge management.

## 🎯 What These Examples Show

These demos are **intentionally simplified** to show the core concepts clearly. They demonstrate:

- ✅ How to transcribe audio with Docling (Whisper Turbo)
- ✅ How to upload documents to OpenRAG
- ✅ How to manage knowledge filters
- ✅ How to combine everything into a complete pipeline

**What they DON'T include** (but production code does):
- ❌ Comprehensive error handling
- ❌ Logging and monitoring
- ❌ State management and resume capability
- ❌ Batch processing
- ❌ Configuration validation
- ❌ Progress tracking

👉 **For production use, see the main [`pipeline/`](../pipeline/) directory.**

## 📋 Prerequisites

Before running these examples, ensure you have:

1. **Python 3.12+** installed
2. **ffmpeg** installed (required by Docling for audio processing)
   ```bash
   # macOS
   brew install ffmpeg
   
   # Ubuntu/Debian
   sudo apt-get install ffmpeg
   
   # Windows (using Chocolatey)
   choco install ffmpeg
   ```
3. **OpenRAG instance** running (local or remote)
4. **Dependencies** installed:
   ```bash
   pip install docling openrag-sdk python-dotenv
   ```

## 🚀 Quick Start

### 1. Set Up Environment

Copy the example environment file and add your OpenRAG credentials:

```bash
cd examples/
cp .env.example .env
# Edit .env and add your OPENRAG_API_KEY
```

### 2. Add Test Audio

Place a test audio file in `sample_files/`:

```bash
# Example: copy an audio file
cp ~/Downloads/podcast.mp3 sample_files/test.mp3
```

See [`sample_files/README.md`](sample_files/README.md) for supported formats and recommendations.

### 3. Run a Demo

```bash
# Transcribe audio only
python demo_docling.py sample_files/test.mp3

# Upload document only
python demo_openrag.py sample_files/transcript.md

# Full pipeline (transcribe + upload)
python demo_full_pipeline.py sample_files/test.mp3
```

## 📚 Demo Files

### [`demo_docling.py`](demo_docling.py) (~60 lines)

**What it shows:** Minimal Docling transcription

**Key concepts:**
- Configuring Whisper Turbo model
- Converting audio to markdown
- Exporting transcripts

**What it doesn't include:**
- Error handling for corrupted files
- Progress tracking for long files
- Timestamp extraction options
- Multiple model support

**Key takeaway:** Docling makes audio transcription simple with just a few lines of code.

```bash
python demo_docling.py sample_files/test.mp3
```

---

### [`demo_openrag.py`](demo_openrag.py) (~58 lines)

**What it shows:** Minimal OpenRAG document upload

**Key concepts:**
- Using the `OpenRAGClient` wrapper
- Uploading documents with wait=True
- Basic error handling

**What it doesn't include:**
- Retry logic (handled by client)
- Batch uploads
- Document validation
- Metadata management

**Key takeaway:** The OpenRAG client wrapper simplifies async SDK operations.

```bash
python demo_openrag.py sample_files/transcript.md
```

---

### [`demo_openrag_filters.py`](demo_openrag_filters.py) (~62 lines)

**What it shows:** Knowledge filter management

**Key concepts:**
- Creating filters for content organization
- Using `ensure_filter` for idempotent operations
- Adding documents to filters

**What it doesn't include:**
- Filter search and listing
- Complex filter queries
- Filter deletion
- Bulk filter operations

**Key takeaway:** Filters help organize content by topic, source, or date for targeted retrieval.

```bash
python demo_openrag_filters.py
```

---

### [`demo_full_pipeline.py`](demo_full_pipeline.py) (~118 lines)

**What it shows:** Complete audio-to-RAG workflow

**Key concepts:**
- Combining Docling + OpenRAG
- Three-step process: transcribe → save → upload
- Progress indicators for user feedback

**What it doesn't include:**
- State management (resume on failure)
- Parallel processing
- Duplicate detection
- Comprehensive validation

**Key takeaway:** The full pipeline is straightforward when you combine the right tools.

```bash
python demo_full_pipeline.py sample_files/test.mp3
```

## 📊 Examples vs. Production

| Feature | Examples | Production (`pipeline/`) |
|---------|----------|-------------------------|
| **Lines of code** | 20-120 per file | 100-500 per file |
| **Error handling** | Basic | Comprehensive |
| **Logging** | Print statements | Structured logging |
| **State management** | None | Full state tracking |
| **Retry logic** | Client-level only | Multi-level retries |
| **Configuration** | Hardcoded | Pydantic validation |
| **Progress tracking** | Simple prints | Detailed progress bars |
| **Batch processing** | Single files | Parallel processing |
| **Testing** | Manual | Automated tests |
| **Documentation** | Inline comments | Full docstrings |

## 🔧 Troubleshooting

### "ffmpeg not found"

Docling requires ffmpeg for audio processing. Install it:

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg
```

### "OPENRAG_API_KEY not set"

Create a `.env` file in the `examples/` directory:

```bash
cp .env.example .env
# Edit .env and add your API key
```

### "File not found"

Make sure your audio file is in the `sample_files/` directory:

```bash
ls sample_files/
# Should show your test files
```

### "Authentication failed"

Check that your OpenRAG API key is correct and the OpenRAG instance is running:

```bash
# Test connection
curl -H "Authorization: Bearer YOUR_API_KEY" http://localhost:3000/api/health
```

### Transcription is slow

For faster transcription, use shorter audio clips (10-30 seconds) for testing. Whisper Turbo is optimized for speed but still processes in real-time or faster.

## 🎓 Learning Path

We recommend exploring the examples in this order:

1. **Start with `demo_docling.py`** - Understand audio transcription
2. **Try `demo_openrag.py`** - Learn document upload
3. **Explore `demo_openrag_filters.py`** - See content organization
4. **Run `demo_full_pipeline.py`** - Experience the complete flow
5. **Study `pipeline/` code** - See production patterns

## 🔗 Related Documentation

- **Production Code:** [`../pipeline/`](../pipeline/) - Full implementation with error handling
- **Architecture:** [`../docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) - System design
- **Technology Stack:** [`../docs/TECHNOLOGY_STACK.md`](../docs/TECHNOLOGY_STACK.md) - Tools and libraries
- **Main README:** [`../README.md`](../README.md) - Project overview

## 💡 Tips for Video Demos

These examples are designed to be video-friendly:

- **Clear variable names** - Easy to read on screen
- **Progress indicators** - Visual feedback with emojis
- **Minimal output** - No verbose logging
- **Self-contained** - Each demo runs independently
- **Good spacing** - Code is easy to follow

## 🤝 Contributing

Found an issue or have a suggestion? These examples are meant to be simple and educational. If you have ideas for improvement:

1. Keep examples under 150 lines
2. Focus on clarity over completeness
3. Add inline comments for key concepts
4. Test with short audio files (< 30 seconds)

## 📝 License

Same as the main project - see [`../LICENSE`](../LICENSE).

---

**Remember:** These are learning examples. For production use, see the [`pipeline/`](../pipeline/) directory for robust, production-ready code with comprehensive error handling, logging, and state management.

Happy coding! 🚀