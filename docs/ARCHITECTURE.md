# Audio to OpenRAG Pipeline — Technical Architecture & Project Specification

> **Version:** 1.1.0
> **Last Updated:** 2026-03-05
> **Status:** Production — Whisper-Based Transcription (via Docling Wrapper) with Timestamp Support
> **Project Root:** `~/audio_to_openrag/`

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture Diagram](#2-system-architecture-diagram)
3. [Component Specifications](#3-component-specifications)
4. [Data Flow](#4-data-flow)
5. [Dependencies Table](#5-dependencies-table)
6. [Environment Variables](#6-environment-variables)
7. [Idempotency Design](#7-idempotency-design)
8. [Error Handling Strategy](#8-error-handling-strategy)
9. [Security Considerations](#9-security-considerations)
10. [Known Limitations](#10-known-limitations)
11. [Future Enhancements](#11-future-enhancements)

---

## 1. Executive Summary

**Audio to OpenRAG Pipeline** is a fully local, automated Python pipeline that ingests video content from any YouTube channel and makes it semantically searchable via a self-hosted OpenRAG instance.

### What It Does

Given a YouTube video URL, playlist URL, or channel URL, the pipeline:

1. **Downloads** the episode video using `yt-dlp`, extracting structured metadata (title, upload date, video ID, description, canonical URL).
2. **Transcribes** the video locally using OpenAI's Whisper Turbo (accessed via Docling's ASR wrapper) — Docling extracts audio from video internally, producing a structured `DoclingDocument` without sending data to any external API.
3. **Preserves** the `DoclingDocument` throughout the pipeline, maintaining document structure and metadata with timestamp information.
4. **Exports** the document in **dual formats**:
   - **DocTags** (`.doctags`) — Docling's structure-preserving format for reference
   - **Markdown** (`.md`) — Human-readable format with timestamps for OpenRAG ingestion
5. **Ingests** the Markdown file into the self-hosted OpenRAG instance via the async `openrag-sdk`, enabling semantic search and RAG-powered Q&A with timestamp-aware transcripts.
6. **Tracks** ingested episodes in a local `state.json` file keyed by YouTube video ID to prevent duplicate ingestion (idempotency). A `--force` flag overrides this check.

### What It Does NOT Do

- It does not publish or distribute content anywhere.
- It does not use any cloud transcription service (Whisper runs 100% locally).
- It does not modify or re-upload content to YouTube.
- It does not perform real-time or live-stream ingestion.
- It includes timestamp extraction from transcription but speaker diarization is not implemented (processing time would be too expensive for long-form content).
- It requires FFmpeg for video processing (used by Docling internally).

### Deployment Context

The pipeline runs as a local CLI tool on the user's macOS machine. The only external network calls are:
- **Outbound to YouTube** via `yt-dlp` (video download)
- **Outbound to `localhost:3000`** (self-hosted OpenRAG Docker container)
- **Outbound to OpenAI API** for embedding generation via OpenRAG's configured `text-embedding-3-small` model

All transcription happens locally via Whisper (accessed through Docling's ASR wrapper). Video files are processed directly without intermediate audio extraction.

---

## 2. System Architecture Diagram

```mermaid
flowchart TD
    CLI[main.py\nCLI Entrypoint\nargparse] --> STATE_CHECK{state.py\nEpisode in\nstate.json?}

    STATE_CHECK -- Yes, no --force --> SKIP[Exit: Already ingested\nSkip episode]
    STATE_CHECK -- No OR --force flag --> ACQ

    subgraph STAGE_1 [Stage 1: Acquire]
        ACQ[acquire.py\nyt-dlp YoutubeDL\nDownload video + metadata]
    end

    subgraph STAGE_2 [Stage 2: Transcribe]
        TRANS[transcribe.py\nWhisper Turbo via Docling\nAsrPipeline wrapper\nProduces DoclingDocument]
    end

    subgraph STAGE_3 [Stage 3: Export Document]
        DOC[document.py\nPreserve DoclingDocument\nDual Export:\nDocTags + Markdown]
    end

    subgraph STAGE_4 [Stage 4: Ingest and Track]
        INGEST[ingest.py\nopenrag-sdk async\nclient.documents.ingest\nfile_path=transcript.doctags]
        STATE_WRITE[state.py\nWrite video_id entry\nto state.json atomically]
    end

    ACQ -->|video file path + metadata dict| TRANS
    TRANS -->|DoclingDocument with timestamps| DOC
    DOC -->|transcripts/video-id.doctags (reference)\ntranscripts/video-id.md (for OpenRAG)| INGEST
    INGEST -->|openrag task_id| STATE_WRITE
    STATE_WRITE --> DONE[Done]

    EXT_YT[(YouTube\nyt-dlp HTTP)] -.->|video stream| ACQ
    EXT_OR[(OpenRAG\nlocalhost:3000\ntext-embedding-3-small)] -.->|HTTP POST multipart\nDocTags format| INGEST
```

### Stage Summary Table

| # | Stage | Module | Primary Input | Primary Output |
|---|-------|--------|---------------|----------------|
| 0 | State Check | `state.py` | YouTube video ID | Skip or proceed decision |
| 1 | Acquire | `acquire.py` | YouTube URL | Video file path + `EpisodeMetadata` |
| 2 | Transcribe | `transcribe.py` | Video file path | `DoclingDocument` with timestamps |
| 3 | Export Document | `document.py` | `DoclingDocument` + metadata | `.doctags` (reference) + `.md` (for OpenRAG) |
| 4 | Ingest | `ingest.py` | `.md` file path | OpenRAG `task_id` |
| 5 | Track State | `state.py` | video ID + `task_id` | Updated `state.json` |

---

## 3. Component Specifications

### 3.1 `pipeline/acquire.py` — Video Acquisition

**Responsibility:** Download a single podcast episode from YouTube as a video file and extract structured metadata.

**Key Inputs:**
- `url: str` — A YouTube video URL (e.g., `https://www.youtube.com/watch?v=VIDEO_ID`) or playlist/channel URL
- `output_dir: Path` — Directory to write the downloaded video file

**Key Outputs:**
- `audio_path: Path` — Absolute path to the downloaded video file (`.mp4`, `.webm`, etc.)
  - Note: Despite the name, this can be any video format that Docling can process
- `metadata: EpisodeMetadata` — A dataclass containing:
  - `video_id: str` — YouTube video ID (used as the idempotency key)
  - `title: str` — Episode title
  - `upload_date: str` — ISO 8601 date string (`YYYYMMDD` from yt-dlp, normalized)
  - `description: str` — Full episode description
  - `url: str` — Canonical YouTube URL
  - `duration_seconds: int` — Episode duration

**Key Dependencies:**
- `yt-dlp` — `yt_dlp.YoutubeDL` class

**Design Decisions:**

1. **Video format:** Download as `best` format. Docling's ASR wrapper extracts audio from video files internally (using ffmpeg), eliminating the need for separate audio extraction. This simplifies the pipeline and reduces processing time.

2. **Filename convention:** Files are named `{video_id}.{ext}` (not the episode title) to avoid filesystem-unsafe characters and to make the idempotency key directly traceable to the file on disk. The extension varies based on the downloaded format (`.mp4`, `.webm`, `.mkv`, etc.).

3. **Playlist support:** When a playlist or channel URL is provided, `acquire.py` yields one `(video_path, metadata)` tuple per episode. The caller (`main.py`) iterates and processes each episode independently, allowing partial failures without aborting the entire batch.

4. **yt-dlp options used:**
   ```python
   ydl_opts = {
       "format": "best",  # Download full video - Docling extracts audio internally
       "outtmpl": str(output_dir / "%(id)s.%(ext)s"),
       "quiet": True,
       "no_warnings": True,
       "extract_flat": False,
       "download_archive": str(audio_dir / ".yt_dlp_archive"),
   }
   ```

5. **No cookies or authentication:** The pipeline assumes the YouTube channel is public. If private/unlisted videos are needed in the future, `yt-dlp`'s `--cookies-from-browser` mechanism should be used (never hardcoded credentials).

6. **File detection:** The code dynamically detects the downloaded file extension using glob patterns, supporting any video format yt-dlp produces.

> ✅ **No FFmpeg dependency:** Unlike the previous architecture, FFmpeg is NOT required for audio extraction. Docling handles video files directly.

---

### 3.2 `pipeline/transcribe.py` — ASR Transcription

**Responsibility:** Transcribe a video file using Whisper Turbo (via Docling's ASR wrapper), producing a structured `DoclingDocument`.

**Key Inputs:**
- `audio_path: Path` — Path to the video file (`.mp4`, `.webm`, etc.)
  - Note: Despite the parameter name, this accepts video files

**Key Outputs:**
- `TranscriptResult` dataclass containing:
  - `document: DoclingDocument` — The structured document object with timestamp information (preserved for export)
  - `markdown: str` — Markdown export of the transcript
  - `model_info: str` — Model identification string (e.g., "Whisper Turbo via Docling AsrPipeline")

**Key Dependencies:**
- `docling` — `DocumentConverter`, `AsrPipeline`, `AsrPipelineOptions`, `asr_model_specs`, `AudioFormatOption`, `InputFormat`

**Core Implementation Pattern:**
```python
from docling.datamodel import asr_model_specs
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import AsrPipelineOptions
from docling.document_converter import AudioFormatOption, DocumentConverter
from docling.pipeline.asr_pipeline import AsrPipeline

pipeline_options = AsrPipelineOptions()
pipeline_options.asr_options = asr_model_specs.WHISPER_TURBO

converter = DocumentConverter(
    format_options={
        InputFormat.AUDIO: AudioFormatOption(
            pipeline_cls=AsrPipeline,
            pipeline_options=pipeline_options,
        )
    }
)
# Pass Path object (not string) to preserve full path for ffmpeg
result = converter.convert(audio_path)
document = result.document  # DoclingDocument preserved for export
markdown = document.export_to_markdown()
```

**Design Decisions:**

1. **Model choice — Whisper Turbo:** Selected for its balance of speed and accuracy. It is significantly faster than `whisper-large-v3` while maintaining high transcription quality for English podcast content. The model runs entirely locally, accessed via Docling's ASR wrapper.

2. **DoclingDocument preservation:** The `DoclingDocument` object is returned and preserved throughout the pipeline, containing timestamp information from the ASR process. This enables timestamp-aware transcript exports.

3. **Video file support:** Docling's ASR wrapper automatically extracts audio from video files using its internal ffmpeg integration. No separate audio extraction step is needed.

4. **Timestamp extraction:** Whisper provides timestamps which are captured in the `DoclingDocument.texts` items as `TrackSource` objects with `start_time` and `end_time` attributes. These timestamps are preserved in the Markdown export for temporal navigation.

5. **Debug logging:** Comprehensive debug logging inspects the `DoclingDocument` structure to report timestamp availability and segment counts. This provides visibility into the transcription quality.

6. **Path handling:** The video file path is passed as a `Path` object (not string) to preserve the full absolute path through Docling's internal processing chain.

---

### 3.3 `pipeline/document.py` — Document Export

**Responsibility:** Export the preserved `DoclingDocument` to dual formats: DocTags (for OpenRAG) and Markdown (for humans).

**Key Inputs:**
- `document: DoclingDocument` — The structured document from transcription (preserved)
- `metadata: EpisodeMetadata`
- `output_dir: Path` — Directory to write the export files

**Key Outputs:**
- `doctags_path: Path` — Path to the exported `.doctags` file (e.g., `transcripts/{video_id}_{title}.doctags`) for reference
- `md_path: Path` — Path to the exported `.md` file (e.g., `transcripts/{video_id}_{title}.md`) with timestamps for OpenRAG ingestion

**Key Dependencies:**
- `docling-core` — `DoclingDocument` with `export_to_doctags()` and `export_to_markdown()` methods

**Design Decisions:**

1. **Dual export strategy:**
   - **DocTags format** (`.doctags`): Docling's structure-preserving JSON format that maintains document semantics, hierarchy, and metadata. Exported for reference and potential future use.
   - **Markdown format** (`.md`): Human-readable format with timestamps that is actually ingested into OpenRAG. Includes `[MM:SS]` or `[H:MM:SS]` timestamps for each segment.

2. **DoclingDocument preservation:** The `DoclingDocument` object is passed through from transcription without re-parsing. This avoids wasteful Markdown → DoclingDocument conversion and preserves all timestamp information from the ASR pipeline.

3. **Filename convention:** Files are named `{video_id}_{sanitized_title}.{ext}` to make them easily identifiable while maintaining filesystem safety.

4. **Timestamp preservation:** Timestamps are extracted from the `DoclingDocument.texts` items (which contain `TrackSource` objects) and formatted into the Markdown output as `**[MM:SS]** text` for segments without speaker labels.

5. **Markdown export format:**
   ```markdown
   # {Episode Title}

   **Channel:** {channel}
   **Date:** {upload_date}
   **YouTube:** {youtube_url}

   ---

   ## Transcript

   **[0:05]** First segment text...

   **[0:23]** Second segment text...

   **[1:15]** Third segment text...
   ```

6. **No re-parsing:** Unlike the previous architecture, we do NOT convert Markdown back to DoclingDocument. The original `DoclingDocument` from transcription is exported directly to both formats, preserving timestamp information.

---

### 3.4 `pipeline/ingest.py` — OpenRAG Ingestion

**Responsibility:** Upload the exported DocTags transcript to the self-hosted OpenRAG instance using the async `openrag-sdk`.

**Key Inputs:**
- `transcript_path: Path` — Path to the exported `.md` transcript file (Markdown format with timestamps)
- `force: bool` — Whether to delete existing document before re-ingesting
- `filter_name: str` — OpenRAG knowledge filter name (default: "Videos")

**Key Outputs:**
- `task_id: str` — The OpenRAG task ID for the ingestion operation

**Key Dependencies:**
- `openrag-sdk` — `OpenRAGClient`, `client.documents.ingest()`, `client.documents.delete()`
- `asyncio` — The SDK is fully async

**Core Implementation Pattern:**
```python
import asyncio, os
from pathlib import Path
from openrag import OpenRAGClient

async def _ingest(transcript_path: Path, filter_name: str = "Videos", force: bool = False) -> str:
    client = OpenRAGClient(
        api_key=os.environ["OPENRAG_API_KEY"],
        base_url=os.environ.get("OPENRAG_URL", "http://localhost:3000"),
    )
    
    # Delete existing document if force=True
    if force:
        await client.documents.delete(transcript_path.name)
    
    # Ingest Markdown file with wait=True to poll until completion
    result = await client.documents.ingest(
        file_path=str(transcript_path),
        wait=True,
    )

    # Update knowledge filter to include this document
    await _ensure_podcast_filter(client, transcript_path.name, filter_name)

    return result.task_id

def ingest_transcript(transcript_path: Path, filter_name: str = "Videos", force: bool = False) -> str:
    return asyncio.run(_ingest(transcript_path, filter_name, force))
```

**Design Decisions:**

1. **Markdown format for ingestion:** OpenRAG ingests the `.md` file (not DocTags) for compatibility. The Markdown file includes timestamps in `[MM:SS]` format for temporal navigation within transcripts.

2. **Async bridging:** The `openrag-sdk` is fully async. Since the pipeline is otherwise synchronous, `asyncio.run()` bridges the sync/async boundary at the module boundary, avoiding `asyncio` complexity throughout the pipeline.

3. **OpenRAG configuration (confirmed from live instance):**
   - Embedding model: `text-embedding-3-small` (OpenAI)
   - Chunk size: 1000 tokens, overlap: 200 tokens
   - Table structure extraction: enabled
   - OCR: disabled
   - **Markdown ingestion:** OpenRAG processes Markdown files with standard chunking, preserving timestamp markers in the text
   For a typical 60-minute podcast episode (~10,000–15,000 words), this produces approximately 30–50 chunks per episode.

4. **Knowledge filter management:** After ingestion, the document filename is automatically added to the specified OpenRAG knowledge filter (default: "Videos"). This enables query-time filtering to scope searches to specific content categories. The filter is created if it doesn't exist, or updated if it does.

5. **Retry logic:** A simple exponential backoff retry (max 3 attempts, delays: 1s, 2s, 4s) wraps the `ingest()` call to handle transient network errors against the local Docker container.

6. **Task polling:** The `wait=True` parameter causes the SDK to poll the ingestion task until completion, returning a `task_id` when done. This simplifies error handling and state tracking.

7. **Force re-ingestion:** When `--force` is used, the existing document is deleted before re-ingesting to prevent duplicate chunks in the vector store.

---

### 3.5 `pipeline/state.py` — Idempotency State Tracking

**Responsibility:** Read and write the `state.json` file to track which episodes have been ingested, preventing duplicate processing.

**Key Inputs / Outputs:** See [Section 7](#7-idempotency-design) for the full schema and logic.

**Key Dependencies:** Python standard library only (`json`, `pathlib`, `datetime`, `tempfile`, `os`).

---

### 3.6 `main.py` — CLI Entrypoint

**Responsibility:** Parse CLI arguments, run preflight checks, orchestrate the pipeline stages in order, and handle top-level error reporting.

**CLI Interface:**
```
usage: python main.py ingest [-h] [--force] [--dry-run] [--filter TEXT] url

positional arguments:
  url                   YouTube video URL or playlist/channel URL

optional arguments:
  --force               Re-ingest even if episode is already in state.json
  --dry-run             Download and transcribe but do not ingest into OpenRAG
  --filter TEXT         OpenRAG knowledge filter name (default: Videos)
```

**Orchestration Flow:**
```python
for video_path, metadata in acquire(url, output_dir):
    if state.is_ingested(metadata.video_id) and not args.force:
        logger.info(f"Skipping {metadata.video_id}: already ingested")
        continue

    transcript_result = transcribe(video_path)  # Returns DoclingDocument with timestamps
    doctags_path, md_path = export_document(transcript_result.document, metadata)

    if not args.dry_run:
        task_id = ingest(md_path, force=args.force)  # Ingest Markdown file
        state.mark_ingested(metadata.video_id, task_id, metadata)
```

---

## 4. Data Flow

This section traces how data transforms at each stage of the pipeline.

```
YouTube URL (string)
        │
        ▼  [acquire.py — yt-dlp YoutubeDL]
        │
        ├── EpisodeMetadata (dataclass)
        │     video_id, title, upload_date,
        │     description, url, duration_seconds
        │
        └── video_path (Path → .mp4/.webm file on disk)
                │
                ▼  [transcribe.py — Whisper via Docling wrapper]
                │
        DoclingDocument with timestamps
          Structured document with transcript + TrackSource timing data from Whisper
                │
                ▼  [document.py — Dual export]
                │
        ├── transcripts/{video_id}_{title}.doctags (DocTags format - reference)
        └── transcripts/{video_id}_{title}.md (Markdown with timestamps - for OpenRAG)
                │
                ▼  [ingest.py — openrag-sdk async]
                │
        OpenRAG task_id (string)
                │
                ▼  [state.py — atomic JSON write]
                │
        state.json entry keyed by video_id
```

### Data Transformation Summary

| Stage | Input Type | Output Type | Key Transformation |
|-------|-----------|-------------|-------------------|
| Acquire | `str` URL | `Path` + `EpisodeMetadata` | HTTP stream → Video file + JSON metadata |
| Transcribe | `Path` (video) | `DoclingDocument` with timestamps | Video → structured document with TrackSource timing data (Whisper via Docling wrapper extracts audio internally) |
| Export | `DoclingDocument` + metadata | `.doctags` (reference) + `.md` (with timestamps) | Dual format export: DocTags for reference, Markdown with timestamps for ingestion |
| Ingest | `.md` file path | `task_id: str` | HTTP multipart upload → Markdown with timestamps chunked in OpenRAG |
| Track | `video_id` + `task_id` | `state.json` | JSON file append (atomic) |

---

## 5. Dependencies Table

| Package | Min Version | Purpose | Notes |
|---------|------------|---------|-------|
| `yt-dlp` | `>=2024.1.0` | YouTube video download and metadata extraction | Update frequently; YouTube API changes often |
| `docling[asr]` | `>=2.74.0` | ASR wrapper, `DocumentConverter`, `DoclingDocument`, timestamp extraction | Provides wrapper around Whisper; bundles Whisper Turbo weights (~1.5 GB); handles video files directly |
| `docling-core` | `>=2.0.0` | `DoclingDocument` data model primitives with `TrackSource` | Transitive dependency via `docling` |
| `openrag-sdk` | `>=0.1.3` | Async client for OpenRAG document ingestion | Fully async; uses `asyncio` |
| `python-dotenv` | `>=1.0.0` | Load `.env` file into environment variables | Dev convenience |
| `click` | `>=8.1.0` | CLI framework for command parsing | Provides `@click.command()` decorators |
| `rich` | `>=13.0.0` | Terminal formatting and progress bars | Beautiful CLI output with colors and spinners |
| `pyannote.audio` | `>=3.1.0` | Speaker diarization (currently disabled) | Optional future feature for speaker identification |
| `torch` | `>=2.0.0` | PyTorch for ML models | Required by `pyannote.audio` and Docling |
| `ffmpeg` | system dep | Video/audio processing | **Required** — Used by Docling internally for video processing |

### `pyproject.toml` Dependencies Block

```toml
[project]
name = "the-flow-pipeline"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "yt-dlp>=2024.1.0",
    "docling[asr]>=2.74.0",
    "pyannote.audio>=3.1.0",
    "torch>=2.0.0",
    "openrag-sdk>=0.1.3",
    "python-dotenv>=1.0.0",
    "click>=8.1.0",
    "rich>=13.0.0",
]

[project.scripts]
the-flow = "main:cli"
```

---

## 6. Environment Variables

All secrets and configuration are passed via environment variables. The pipeline uses `python-dotenv` to load a `.env` file from the project root at startup. **The `.env` file must never be committed to version control.**

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENRAG_API_KEY` | **Yes** | — | API key for the self-hosted OpenRAG instance. Set in the OpenRAG Docker configuration. |
| `OPENRAG_URL` | No | `http://localhost:3000` | Base URL of the OpenRAG instance. Override if running on a non-default port or remote host. |
| `PIPELINE_AUDIO_DIR` | No | `./audio` | Directory where downloaded video files are stored. |
| `PIPELINE_TRANSCRIPT_DIR` | No | `./transcripts` | Directory where exported `.doctags` and `.md` transcript files are stored. |
| `PIPELINE_STATE_FILE` | No | `./state.json` | Path to the idempotency state file. |
| `PIPELINE_LOG_LEVEL` | No | `INFO` | Python logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`). Set to `DEBUG` to inspect DoclingDocument structure. |

### `.env.example`

```dotenv
# OpenRAG configuration
OPENRAG_API_KEY=your_openrag_api_key_here
OPENRAG_URL=http://localhost:3000

# Optional pipeline configuration
# PIPELINE_AUDIO_DIR=./audio
# PIPELINE_TRANSCRIPT_DIR=./transcripts
# PIPELINE_STATE_FILE=./state.json
# PIPELINE_LOG_LEVEL=INFO
```

---

## 7. Idempotency Design

### 7.1 Purpose

The pipeline must be safe to run multiple times against the same YouTube channel or playlist without re-downloading, re-transcribing, or re-ingesting episodes that have already been processed. This is critical for incremental updates (e.g., running weekly to pick up new episodes).

### 7.2 `state.json` Schema

```json
{
  "schema_version": "1.0",
  "last_updated": "2026-02-27T17:00:00Z",
  "episodes": {
    "dQw4w9WgXcQ": {
      "video_id": "dQw4w9WgXcQ",
      "title": "Episode 42: The Future of AI",
      "upload_date": "2026-01-15",
      "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
      "openrag_document_id": "doc_abc123xyz",
      "ingested_at": "2026-02-27T17:00:00Z",
      "transcript_path": "transcripts/dQw4w9WgXcQ.md",
      "pipeline_version": "0.1.0"
    }
  }
}
```

**Key design choices:**
- **Keyed by `video_id`:** YouTube video IDs are stable, unique, and URL-safe. They are the natural idempotency key.
- **`openrag_document_id` stored:** Enables future operations (e.g., deleting a document from OpenRAG before re-ingesting with `--force`).
- **`ingested_at` timestamp:** Enables auditing and debugging.
- **`pipeline_version` stored:** Enables future migrations when the pipeline output format changes.
- **`schema_version` at root:** Enables forward-compatible schema migrations without breaking existing state files.

### 7.3 State Check and Write Logic

```python
import json, os, tempfile
from datetime import datetime, timezone
from pathlib import Path

STATE_PATH = Path(os.environ.get("PIPELINE_STATE_FILE", "./state.json"))

def load_state() -> dict:
    if not STATE_PATH.exists():
        return {"schema_version": "1.0", "last_updated": "", "episodes": {}}
    try:
        return json.loads(STATE_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        # Corrupt state file — log warning and treat as empty
        return {"schema_version": "1.0", "last_updated": "", "episodes": {}}

def is_ingested(video_id: str) -> bool:
    return video_id in load_state().get("episodes", {})

def mark_ingested(video_id: str, document_id: str, metadata) -> None:
    state = load_state()
    state["episodes"][video_id] = {
        "video_id": video_id,
        "title": metadata.title,
        "upload_date": metadata.upload_date,
        "youtube_url": metadata.url,
        "openrag_document_id": document_id,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "transcript_path": str(Path("transcripts") / f"{video_id}.md"),
        "pipeline_version": "0.1.0",
    }
    state["last_updated"] = datetime.now(timezone.utc).isoformat()
    _atomic_write(STATE_PATH, state)

def _atomic_write(path: Path, data: dict) -> None:
    """Write JSON atomically using temp file + rename (POSIX-safe)."""
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, suffix=".tmp", delete=False
    ) as f:
        json.dump(data, f, indent=2)
        tmp_path = f.name
    os.replace(tmp_path, path)  # atomic on POSIX systems
```

### 7.4 `--force` Flag Behavior

When `--force` is passed:
1. The `is_ingested()` check is bypassed — the episode is processed regardless of `state.json`.
2. If the episode was previously ingested, the old `openrag_document_id` is retrieved from state and used to **delete** the existing document from OpenRAG before re-ingesting, preventing duplicate chunks in the knowledge base.
3. The `state.json` entry is overwritten with the new `openrag_document_id`.

> ⚠️ **Assumption:** The `openrag-sdk` provides a `client.documents.delete(document_id)` method. This must be verified against the SDK documentation before implementing `--force` deletion.

---

## 8. Error Handling Strategy

The pipeline uses a **fail-fast per episode, continue across episodes** strategy. A failure in one episode does not abort processing of subsequent episodes in a batch run.

### 8.1 Stage-by-Stage Error Handling

| Stage | Failure Mode | Handling Strategy |
|-------|-------------|-------------------|
| **State Check** | `state.json` corrupt or unreadable | Log `WARNING`, treat as empty state (safe default — worst case is re-processing) |
| **Acquire** | Network error, video unavailable, geo-blocked | Log `ERROR` with video ID, skip episode, continue batch |
| **Acquire** | `ffmpeg` not found on PATH | Raise `SystemExit` with install instructions — fatal configuration error caught at preflight |
| **Acquire** | Disk full | Raise `OSError`, log `CRITICAL`, abort entire run |
| **Transcribe** | Whisper model not yet downloaded | Whisper model downloads automatically on first use via Docling; log progress |
| **Transcribe** | Audio file corrupt or unreadable | Log `ERROR`, skip episode |
| **Transcribe** | Out of memory (very long episode) | Log `ERROR` with suggestion to use a smaller Whisper model variant, skip episode |
| **Build Document** | Empty transcript | Log `WARNING`, write empty transcript body, continue to ingest (empty doc is valid) |
| **Ingest** | OpenRAG unreachable | Retry 3× with exponential backoff (1s, 2s, 4s); if all fail, log `ERROR`, skip episode — do NOT mark as ingested |
| **Ingest** | OpenRAG returns 4xx error | Log `ERROR` with response body, skip episode |
| **State Write** | Disk full or permission error | Log `CRITICAL`; episode was ingested but state not saved — log the `document_id` to stderr for manual recovery |

### 8.2 Startup Preflight Checks

Before processing any episodes, `main.py` validates the environment:

```python
import shutil, os, re

def preflight_check(url: str) -> None:
    # 1. Required environment variables
    if not os.environ.get("OPENRAG_API_KEY"):
        raise EnvironmentError(
            "Required environment variable 'OPENRAG_API_KEY' is not set. "
            "Copy .env.example to .env and fill in your values."
        )

    # 2. ffmpeg on PATH
    if not shutil.which("ffmpeg"):
        raise SystemExit(
            "ffmpeg is not installed or not on PATH.\n"
            "Install with: brew install ffmpeg"
        )

    # 3. YouTube URL format validation
    _validate_youtube_url(url)

    # 4. OpenRAG reachability (HTTP GET to health endpoint)
    _check_openrag_health()
```

### 8.3 Logging Strategy

- All log output uses Python's standard `logging` module with structured formatting.
- Log level is configurable via `PIPELINE_LOG_LEVEL` env var (default: `INFO`).
- **Secrets are never logged.** The `OPENRAG_API_KEY` value is masked in all log output using a custom `logging.Filter`.
- Each episode's processing is wrapped in a `try/except` block that logs the full traceback at `DEBUG` level and a one-line summary at `ERROR` level.
- A final summary is printed at the end of a batch run:
  ```
  Pipeline complete: 5 episodes processed — 4 ingested, 1 skipped (already ingested), 0 failed
  ```

---

## 9. Security Considerations

This section documents security controls aligned with OWASP best practices applicable to a local CLI pipeline.

### 9.1 Secrets Management — OWASP A02: Cryptographic Failures

| Control | Implementation |
|---------|---------------|
| No hardcoded secrets | All tokens and API keys loaded exclusively from environment variables via `python-dotenv` |
| `.env` excluded from VCS | `.gitignore` must include `.env` and `*.env` |
| `.env.example` committed | Provides template without real values; safe to commit |
| Secrets masked in logs | Custom `logging.Filter` replaces known secret values with `[REDACTED]` before any log output |
| No secrets in CLI arguments | Tokens are not accepted as CLI arguments (which appear in `ps` output and shell history) |
| Secrets loaded once at startup | `os.environ` values are read once; not re-read in hot paths |

### 9.2 Input Validation — OWASP A03: Injection

The primary external input is the YouTube URL provided by the user. This URL is passed to `yt-dlp` and must be validated before use to prevent the pipeline from making unintended network requests.

**URL Validation:**
```python
import re

# Matches YouTube video URLs, playlist URLs, and channel URLs
YOUTUBE_URL_PATTERN = re.compile(
    r'^https?://(www\.)?(youtube\.com/(watch\?v=[\w\-]{11}|playlist\?list=[\w\-]+|@[\w\-]+(/videos)?)|youtu\.be/[\w\-]{11})(\?.*)?$'
)

def _validate_youtube_url(url: str) -> str:
    url = url.strip()
    if not YOUTUBE_URL_PATTERN.match(url):
        raise ValueError(
            f"Invalid YouTube URL: {url!r}\n"
            "Expected format: https://www.youtube.com/watch?v=VIDEO_ID"
        )
    return url
```

**File path sanitization:** All file paths constructed from external data (e.g., video IDs used in filenames) are validated to contain only safe characters:

```python
def safe_video_id(video_id: str) -> str:
    """YouTube video IDs are 11 alphanumeric + hyphen + underscore chars."""
    if not re.match(r'^[\w\-]{1,64}$', video_id):
        raise ValueError(f"Unexpected video ID format: {video_id!r}")
    return video_id
```

### 9.3 Dependency Security — OWASP A06: Vulnerable and Outdated Components

- All dependencies are pinned to minimum versions in `pyproject.toml`.
- Run `pip audit` periodically to scan for known CVEs in the dependency tree.
- `yt-dlp` must be updated frequently (`pip install --upgrade yt-dlp`) as YouTube changes its extraction API regularly; outdated versions will silently fail to download.

### 9.4 Data Privacy Disclosure

| Data | Destination | Notes |
|------|------------|-------|
| Audio files | Local disk only | Never transmitted externally |
| Transcript text | OpenRAG → OpenAI API | OpenRAG uses `text-embedding-3-small`; transcript text is sent to OpenAI for embedding generation |
| Episode metadata | Local `state.json` only | Never transmitted externally |

> ⚠️ **Privacy Note:** Transcript text is sent to OpenAI's API for embedding generation via the configured OpenRAG instance. If the podcast contains sensitive personal information, consider switching the OpenRAG embedding model to a locally-hosted Ollama model to keep all data fully local. The live OpenRAG instance already supports Ollama as an embedding provider.

### 9.5 Local Network Security

- The OpenRAG instance runs on `localhost:3000` inside Docker. It must not be exposed to external networks.
- Verify Docker's port binding uses `127.0.0.1:3000:3000` (not `0.0.0.0:3000:3000`) to prevent external access.
- The `OPENRAG_API_KEY` provides authentication against the local instance.

---

## 10. Known Limitations

| # | Limitation | Impact | Workaround / Mitigation |
|---|-----------|--------|------------------------|
| 1 | **Whisper Turbo is optimized for English** | Non-English episodes will have degraded transcription quality | Switch `asr_model_specs` to `WHISPER_LARGE_V3` for multilingual support |
| 2 | **No real-time or live-stream support** | Pipeline only processes completed, uploaded YouTube videos | Not applicable for current use case |
| 3 | **Transcript text sent to OpenAI for embeddings** | Privacy implication for sensitive conversations | Switch OpenRAG to Ollama embeddings for fully local operation |
| 4 | **No automatic new-episode detection** | Pipeline must be run manually or via cron to pick up new episodes | Add a cron job or scheduled GitHub Actions workflow |
| 5 | **`state.json` is not safe for concurrent access** | Running two pipeline instances simultaneously may corrupt state | Use file locking (`fcntl.flock`) or migrate to SQLite for concurrent access |
| 6 | **Video files are not cleaned up after ingestion** | Disk usage grows with each episode (~100–500 MB per hour of video) | Add `--cleanup-video` flag to delete video files after successful ingestion |
| 7 | **No transcript quality scoring** | There is no automated way to detect poor transcription quality (e.g., heavy background noise, strong accents) | Manual review of `transcripts/` directory; future enhancement could add a confidence score |
| 8 | **Speaker diarization not implemented** | Transcripts include timestamps but no speaker labels (processing time would be too expensive for long-form content) | pyannote.audio is installed but not used; could be enabled for short clips if needed |
| 9 | **Timestamps extracted but format limited** | Timestamps are extracted from DoclingDocument TrackSource objects and formatted as [MM:SS] or [H:MM:SS] in Markdown | Current implementation provides segment-level timestamps; word-level timestamps not yet supported |

---

## 11. Future Enhancements

These are optional improvements that are out of scope for v1.0 but worth tracking for future iterations.

| Priority | Enhancement | Description |
|----------|------------|-------------|
| High | **Automatic new-episode detection** | Poll the YouTube channel RSS feed or use `yt-dlp`'s `--dateafter` flag to automatically detect and process new videos without manual invocation |
| Medium | **Optional speaker diarization** | Add optional speaker identification using pyannote.audio for short clips where processing time is acceptable (not suitable for long-form content due to computational cost) |
| Medium | **Local embeddings via Ollama** | Switch OpenRAG's embedding model from `text-embedding-3-small` (OpenAI) to a locally-hosted Ollama model (e.g., `nomic-embed-text`) to keep all data fully local and eliminate OpenAI API costs |
| Medium | **Metadata enrichment at ingestion** | Pass structured metadata (episode date, guest name, topic tags) to OpenRAG at ingestion time once the SDK supports custom metadata fields, enabling filtered RAG queries (e.g., "What did Alice say about X?") |
| Medium | **Audio cleanup flag** | Add `--cleanup-audio` to delete `.mp3` files after successful ingestion to manage disk usage |
| Medium | **SQLite state backend** | Replace `state.json` with a SQLite database for concurrent-safe access, richer querying, and easier migration |
| Medium | **Transcript quality validation** | Add a post-transcription step that checks for common Whisper failure modes (e.g., hallucinated repetitions, very low word count relative to duration) and flags episodes for manual review |
| Low | **Spotify/Apple Podcasts as fallback source** | If a YouTube video is unavailable (deleted, private), fall back to downloading from Spotify or Apple Podcasts using their respective APIs |
| Low | **Cloud deployment** | Package the pipeline as a Docker container with a scheduled trigger (GitHub Actions, AWS EventBridge) for fully automated weekly ingestion |
| Low | **Web UI for transcript review** | A simple local web interface (e.g., Streamlit) to browse ingested transcripts and re-ingest edited transcripts |
| Low | **Chapter/topic segmentation** | Use an LLM to automatically segment the transcript into named chapters (e.g., "Introduction", "Main Topic", "Q&A") and store them as document sections in OpenRAG for more granular retrieval |

---

## Appendix: Project Directory Structure

```
~/audio_to_openrag/
├── ARCHITECTURE.md          ← This document
├── README.md                ← Setup and usage instructions
├── pyproject.toml           ← Python project metadata and dependencies
├── .env.example             ← Environment variable template (safe to commit)
├── .env                     ← Actual secrets (NEVER commit)
├── .gitignore               ← Must include .env, audio/, state.json
├── main.py                  ← CLI entrypoint (argparse orchestration)
├── pipeline/
│   ├── __init__.py
│   ├── acquire.py           ← yt-dlp YouTube download + metadata extraction
│   ├── transcribe.py        ← Whisper Turbo transcription (via Docling ASR wrapper)
│   ├── document.py          ← DoclingDocument build, .md export
│   ├── ingest.py            ← openrag-sdk async ingestion
│   └── state.py             ← Idempotency state tracking (state.json)
├── audio/                   ← Downloaded .mp3 files (gitignored)
├── transcripts/             ← Exported .md transcript files
└── state.json               ← Ingestion state tracker (gitignored)
```

---

*End of Architecture Document — The Flow Pipeline v1.0.0*