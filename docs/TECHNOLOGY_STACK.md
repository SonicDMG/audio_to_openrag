# Technology Stack: Docling & OpenRAG SDK Usage

This document highlights the specific components from **Docling** and the **OpenRAG SDK** that power the video-to-RAG pipeline with structure-preserving DocTags export.

> **Last Updated:** 2026-03-03
> **Architecture Version:** 1.1.0 — Docling-Based with DocTags Export

---

## 🎯 Docling Components

[Docling](https://github.com/DS4SD/docling) is IBM's open-source document understanding framework. This pipeline uses Docling for **video transcription and structure-preserving document export**:

### 1. Video Transcription (ASR Pipeline)

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
- **`AsrPipeline`** — Docling's media processing pipeline that handles speech-to-text conversion from video or audio files
- **`asr_model_specs.WHISPER_TURBO`** — Pre-configured Whisper Turbo model specification for fast, accurate transcription
- **`DocumentConverter`** — Docling's unified converter that can process multiple document types (PDF, video, audio, images, etc.)
- **`AudioFormatOption`** — Configuration class that tells DocumentConverter to use the ASR pipeline for media files
- **Video file support** — Docling automatically extracts audio from video files internally using ffmpeg

**Key Implementation:**
```python
# Configure ASR pipeline with Whisper Turbo
pipeline_options = AsrPipelineOptions()
pipeline_options.asr_options = asr_model_specs.WHISPER_TURBO

# Create converter with audio/video processing enabled
converter = DocumentConverter(
    format_options={
        InputFormat.AUDIO: AudioFormatOption(
            pipeline_cls=AsrPipeline,
            pipeline_options=pipeline_options,
        )
    }
)

# Transcribe video file (Docling extracts audio internally)
result = converter.convert(video_path)  # Pass Path object, not string
document = result.document  # DoclingDocument preserved for export
```

**Why This Matters:**
- Docling handles all the complexity of video/audio processing (ffmpeg integration, model loading, chunking)
- Returns a structured `DoclingDocument` with rich metadata
- **No separate audio extraction needed** — processes video files directly
- DoclingDocument is preserved throughout the pipeline for structure-aware export
- No need to manage Whisper models directly

---

### 2. Structure-Preserving Document Export

**Location:** `pipeline/document.py`

**Components Used:**
```python
from docling_core.types.doc import DoclingDocument
```

**What It Does:**
- **`DoclingDocument.export_to_doctags()`** — Exports to Docling's structure-preserving JSON format optimized for RAG
- **`DoclingDocument.export_to_markdown()`** — Exports to human-readable Markdown format
- **DocTags format** — Maintains document hierarchy, semantics, and metadata for better RAG retrieval
- **Dual export strategy** — DocTags for OpenRAG ingestion, Markdown for human review

**Key Implementation:**
```python
def export_document(document: DoclingDocument, metadata: EpisodeMetadata, output_dir: Path):
    """Export DoclingDocument to dual formats."""
    # Export to DocTags (structure-preserving for RAG)
    doctags_data = document.export_to_doctags()
    doctags_path = output_dir / f"{video_id}_{title}.doctags"
    doctags_path.write_text(json.dumps(doctags_data, indent=2))
    
    # Export to Markdown (human-readable reference)
    markdown = document.export_to_markdown()
    md_path = output_dir / f"{video_id}_{title}.md"
    md_path.write_text(markdown)
    
    return doctags_path, md_path
```

**Why This Matters:**
- **No wasteful re-parsing** — DoclingDocument from transcription is exported directly
- **Structure preservation** — DocTags maintains document semantics for better RAG performance
- **Dual artifacts** — DocTags for machine consumption, Markdown for human review
- **Better retrieval quality** — Structure-aware chunking in OpenRAG produces more relevant results
- **Debug capabilities** — Comprehensive logging to inspect DoclingDocument structure

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

Here's how Docling and OpenRAG SDK work together in the new architecture:

```
1. YouTube URL
   ↓
2. yt-dlp downloads video → .mp4/.webm file
   ↓
3. Docling AsrPipeline (Whisper Turbo) → DoclingDocument (PRESERVED)
   ↓
4. Dual Export:
   ├─→ DocTags format (.doctags) → Structure-preserving JSON
   └─→ Markdown format (.md) → Human-readable reference
   ↓
5. OpenRAG SDK ingest() → Ingests .doctags file
   ↓
6. Structure-aware chunking → Vector embeddings with preserved semantics
   ↓
7. OpenRAG SDK knowledge_filters → "Podcast" filter updated
   ↓
8. State tracking → state.json
```

**Key Architectural Changes:**
- ✅ **No FFmpeg audio extraction** — Docling processes video files directly
- ✅ **DoclingDocument preserved** — No wasteful Markdown re-parsing
- ✅ **DocTags export** — Structure-preserving format for better RAG performance
- ✅ **Dual format strategy** — DocTags for machines, Markdown for humans
- ✅ **Debug logging** — Comprehensive inspection of DoclingDocument structure

---

## 📊 Key Advantages

### Docling Benefits
1. **Unified API** — Same `DocumentConverter` for video, audio, PDF, Markdown, images
2. **Production-Ready** — IBM-maintained, battle-tested in enterprise environments
3. **Rich Output** — Structured documents with metadata, not just raw text
4. **Flexible Models** — Easy to swap ASR backends (Whisper Turbo, Whisper Large, etc.)
5. **Video Support** — Processes video files directly without separate audio extraction
6. **Structure Preservation** — DocTags export maintains document semantics for better RAG
7. **No FFmpeg Dependency** — Docling handles video processing internally

### OpenRAG SDK Benefits
1. **Async-First** — Built for high-throughput ingestion pipelines
2. **Task Polling** — `wait=True` handles long-running ingestion automatically
3. **Knowledge Filters** — Query-time filtering without re-indexing
4. **Self-Hosted** — Full control over your data and infrastructure
5. **DocTags Support** — Structure-aware chunking for better retrieval quality

### Architecture Benefits
1. **Efficiency** — No wasteful Markdown → DoclingDocument re-parsing
2. **Quality** — Structure-preserving DocTags format improves RAG retrieval
3. **Simplicity** — Fewer dependencies (no FFmpeg for audio extraction)
4. **Debuggability** — Comprehensive logging to inspect document structure
5. **Dual Artifacts** — DocTags for machines, Markdown for humans

---

## 🔧 Configuration Points

### Docling Configuration
- **Model Selection:** `asr_model_specs.WHISPER_TURBO` (can swap to `WHISPER_LARGE` for higher accuracy)
- **Pipeline Options:** `AsrPipelineOptions()` (can configure language, task type, etc.)
- **Input Formats:** `InputFormat.AUDIO` (handles both video and audio files)
- **Export Formats:** DocTags (structure-preserving) + Markdown (human-readable)
- **Debug Logging:** Set `PIPELINE_LOG_LEVEL=DEBUG` to inspect DoclingDocument structure

### OpenRAG SDK Configuration
- **Base URL:** `OPENRAG_URL` environment variable (default: `http://localhost:3000`)
- **API Key:** `OPENRAG_API_KEY` environment variable (required)
- **Retry Logic:** 3 attempts with exponential backoff (1s, 2s, 4s delays)
- **Wait Mode:** `wait=True` polls until ingestion completes
- **Ingestion Format:** DocTags files (`.doctags`) for structure-aware chunking

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
- **DocTags format** is Docling's native structure-preserving export format
- Video files are processed directly — no separate audio extraction needed

### Understanding OpenRAG
- OpenRAG is a **self-hosted RAG platform**, not a cloud service
- It handles chunking, embedding, and vector storage automatically
- Knowledge filters enable **query-time filtering** without re-indexing
- **DocTags support** enables structure-aware chunking for better retrieval

### Understanding the New Architecture
- **DoclingDocument preservation** — No wasteful re-parsing of Markdown
- **Dual export strategy** — DocTags for RAG optimization, Markdown for humans
- **No FFmpeg dependency** — Docling handles video files end-to-end
- **Debug capabilities** — Comprehensive logging to inspect document structure
- **Timestamp investigation** — Debug logging added to understand timestamp availability

---

*This document is maintained as part of the audio_to_openrag pipeline. Last updated: 2026-03-03*