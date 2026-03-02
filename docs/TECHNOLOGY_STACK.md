# Technology Stack: Docling & OpenRAG SDK Usage

This document highlights the specific components from **Docling** and the **OpenRAG SDK** that power the audio-to-RAG pipeline.

---

## 🎯 Docling Components

[Docling](https://github.com/DS4SD/docling) is IBM's open-source document understanding framework. This pipeline uses Docling for **two distinct purposes**:

### 1. Audio Transcription (ASR Pipeline)

**Location:** `pipeline/transcribe.py`

**Components Used:**
```python
from docling.datamodel import asr_model_specs
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import AsrPipelineOptions
from docling.document_converter import AudioFormatOption, DocumentConverter
from docling.pipeline.asr_pipeline import AsrPipeline
```

**What It Does:**
- **`AsrPipeline`** — Docling's audio processing pipeline that handles speech-to-text conversion
- **`asr_model_specs.WHISPER_TURBO`** — Pre-configured Whisper Turbo model specification for fast, accurate transcription
- **`DocumentConverter`** — Docling's unified converter that can process multiple document types (PDF, audio, images, etc.)
- **`AudioFormatOption`** — Configuration class that tells DocumentConverter to use the ASR pipeline for audio files

**Key Implementation:**
```python
# Configure ASR pipeline with Whisper Turbo
pipeline_options = AsrPipelineOptions()
pipeline_options.asr_options = asr_model_specs.WHISPER_TURBO

# Create converter with audio processing enabled
converter = DocumentConverter(
    format_options={
        InputFormat.AUDIO: AudioFormatOption(
            pipeline_cls=AsrPipeline,
            pipeline_options=pipeline_options,
        )
    }
)

# Transcribe audio file
result = converter.convert(audio_path)
document = result.document
markdown = document.export_to_markdown()
```

**Why This Matters:**
- Docling handles all the complexity of audio processing (ffmpeg integration, model loading, chunking)
- Returns a structured `DoclingDocument` with rich metadata
- Provides clean Markdown export for downstream processing
- No need to manage Whisper models directly

---

### 2. Markdown Document Processing

**Location:** `pipeline/document.py`

**Components Used:**
```python
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter
```

**What It Does:**
- **`DocumentConverter`** with **`InputFormat.MD`** — Parses Markdown files into structured DoclingDocument objects
- Provides a unified document model for both transcribed audio and written content
- Enables programmatic access to document structure (headings, paragraphs, metadata)

**Key Implementation:**
```python
def _convert_markdown_to_docling(md_path: Path) -> object:
    """Convert Markdown transcript to DoclingDocument."""
    converter = DocumentConverter(
        allowed_formats=[InputFormat.MD],
    )
    result = converter.convert(str(md_path))
    return result.document
```

**Why This Matters:**
- Creates a consistent document model across the pipeline
- Markdown processing is **lightweight** (no ML models required, unlike PDF processing)
- Enables future enhancements (e.g., extracting specific sections, metadata enrichment)
- The Markdown file remains the primary artifact for OpenRAG ingestion

---

## 🚀 OpenRAG SDK Components

[OpenRAG SDK](https://github.com/langflow-ai/openrag) is the Python client for the OpenRAG platform. This pipeline uses it for document ingestion and knowledge management.

**Location:** `pipeline/ingest.py`

### Core SDK Classes

```python
from openrag_sdk import OpenRAGClient
from openrag_sdk.exceptions import AuthenticationError, NotFoundError
from openrag_sdk.models import (
    IngestTaskStatus,
    CreateKnowledgeFilterOptions,
    KnowledgeFilterQueryData,
    UpdateKnowledgeFilterOptions,
)
```

### 1. Document Ingestion

**Component:** `OpenRAGClient.documents.ingest()`

**What It Does:**
- Uploads Markdown transcripts to OpenRAG
- Chunks documents for semantic search
- Returns task status with polling support

**Key Implementation:**
```python
client = OpenRAGClient(api_key=api_key, base_url=base_url)

# Upload with wait=True to poll until completion
result = await client.documents.ingest(
    file_path=str(transcript_path),
    wait=True,  # Polls until task completes
)

# result is IngestTaskStatus with task_id, status, failed_files
task_id = result.task_id
status = result.status
```

**Response Schema:**
- **`task_id`** — Unique identifier for the ingestion task (stored in `state.json`)
- **`status`** — One of: `"completed"`, `"failed"`, `"processing"`
- **`failed_files`** — List of files that failed to process (if any)

---

### 2. Document Deletion

**Component:** `OpenRAGClient.documents.delete()`

**What It Does:**
- Removes documents from OpenRAG by filename
- Deletes all associated chunks from the vector store
- Used with `--force` flag to enable re-ingestion

**Key Implementation:**
```python
result = await client.documents.delete(filename)
logger.info(
    "Deleted document '%s' (chunks removed: %d)",
    filename,
    result.deleted_chunks,
)
```

**Response Schema:**
- **`success`** — Boolean indicating deletion success
- **`deleted_chunks`** — Number of vector chunks removed

---

### 3. Knowledge Filters

**Component:** `OpenRAGClient.knowledge_filters`

**What It Does:**
- Creates query-time filters to scope searches to specific document sets
- Automatically maintains a "Podcast" filter that tracks all ingested transcripts
- Enables users to search only podcast content in OpenRAG

**Key Implementation:**
```python
# Search for existing filter
existing = await client.knowledge_filters.search("Podcast", limit=20)

# Create new filter
options = CreateKnowledgeFilterOptions(
    name="Podcast",
    description="Auto-created filter for podcast transcripts",
    queryData=KnowledgeFilterQueryData(
        query="",
        filters={
            "data_sources": [filename],  # Track filenames
            "document_types": ["*"]
        },
        limit=10,
        scoreThreshold=0.0,
        color="blue",
        icon="book",
    ),
)
result = await client.knowledge_filters.create(options)

# Update existing filter (append new filename)
update_options = UpdateKnowledgeFilterOptions(
    queryData=KnowledgeFilterQueryData(
        filters={
            "data_sources": updated_sources,  # Append new file
            "document_types": ["*"]
        },
        # ... other fields
    ),
)
await client.knowledge_filters.update(filter_id, update_options)
```

**Why This Matters:**
- Users can filter searches to only podcast transcripts
- Automatically maintained — no manual filter management
- Enables multi-tenant use cases (e.g., separate filters per channel)

---

### 4. Error Handling

**Components:** `AuthenticationError`, `NotFoundError`

**What It Does:**
- Provides typed exceptions for common failure modes
- Enables graceful error handling and retry logic

**Key Implementation:**
```python
try:
    result = await client.documents.ingest(...)
except AuthenticationError:
    raise RuntimeError(
        "OpenRAG authentication failed. "
        "Check that OPENRAG_API_KEY is correct."
    )
except NotFoundError:
    logger.debug("Document not found (nothing to delete).")
```

---

## 🔄 Pipeline Flow

Here's how Docling and OpenRAG SDK work together:

```
1. YouTube URL
   ↓
2. yt-dlp downloads audio → MP3 file
   ↓
3. Docling AsrPipeline (Whisper Turbo) → DoclingDocument
   ↓
4. Export to Markdown → .md file
   ↓
5. Docling DocumentConverter (Markdown parser) → DoclingDocument
   ↓
6. OpenRAG SDK ingest() → Chunks + Vector embeddings
   ↓
7. OpenRAG SDK knowledge_filters → "Podcast" filter updated
   ↓
8. State tracking → state.json
```

---

## 📊 Key Advantages

### Docling Benefits
1. **Unified API** — Same `DocumentConverter` for audio, PDF, Markdown, images
2. **Production-Ready** — IBM-maintained, battle-tested in enterprise environments
3. **Rich Output** — Structured documents with metadata, not just raw text
4. **Flexible Models** — Easy to swap ASR backends (Whisper Turbo, Whisper Large, etc.)

### OpenRAG SDK Benefits
1. **Async-First** — Built for high-throughput ingestion pipelines
2. **Task Polling** — `wait=True` handles long-running ingestion automatically
3. **Knowledge Filters** — Query-time filtering without re-indexing
4. **Self-Hosted** — Full control over your data and infrastructure

---

## 🔧 Configuration Points

### Docling Configuration
- **Model Selection:** `asr_model_specs.WHISPER_TURBO` (can swap to `WHISPER_LARGE` for higher accuracy)
- **Pipeline Options:** `AsrPipelineOptions()` (can configure language, task type, etc.)
- **Input Formats:** `InputFormat.AUDIO`, `InputFormat.MD` (can add PDF, DOCX, etc.)

### OpenRAG SDK Configuration
- **Base URL:** `OPENRAG_URL` environment variable (default: `http://localhost:3000`)
- **API Key:** `OPENRAG_API_KEY` environment variable (required)
- **Retry Logic:** 3 attempts with exponential backoff (1s, 2s, 4s delays)
- **Wait Mode:** `wait=True` polls until ingestion completes

---

## 📚 References

- **Docling Documentation:** https://github.com/DS4SD/docling
- **Docling Audio Example:** https://github.com/TejasQ/example-docling-media
- **OpenRAG Documentation:** https://github.com/langflow-ai/openrag
- **OpenRAG SDK:** https://pypi.org/project/openrag-sdk/

---

## 🎓 Learning Resources

### Understanding Docling
- Docling is designed for **document understanding**, not just text extraction
- It preserves structure (headings, tables, lists) across formats
- The ASR pipeline is just one of many specialized pipelines (OCR, layout analysis, etc.)

### Understanding OpenRAG
- OpenRAG is a **self-hosted RAG platform**, not a cloud service
- It handles chunking, embedding, and vector storage automatically
- Knowledge filters enable **query-time filtering** without re-indexing

---

*This document is maintained as part of the audio_to_openrag pipeline. Last updated: 2026-03-02*