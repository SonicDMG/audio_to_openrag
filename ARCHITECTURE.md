# The Flow Pipeline — Technical Architecture & Project Specification

> **Version:** 1.0.0
> **Last Updated:** 2026-02-27
> **Status:** Draft — Pending Implementation Review
> **Project Root:** `~/audio_to_openrag/`

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture Diagram](#2-system-architecture-diagram)
3. [Component Specifications](#3-component-specifications)
4. [Data Flow](#4-data-flow)
5. [Dependencies Table](#5-dependencies-table)
6. [Environment Variables](#6-environment-variables)
7. [Diarization Strategy](#7-diarization-strategy)
8. [Idempotency Design](#8-idempotency-design)
9. [Error Handling Strategy](#9-error-handling-strategy)
10. [Security Considerations](#10-security-considerations)
11. [Known Limitations](#11-known-limitations)
12. [Future Enhancements](#12-future-enhancements)

---

## 1. Executive Summary

**The Flow Pipeline** is a fully local, automated Python pipeline that ingests podcast episodes from *The Flow* YouTube channel and makes them semantically searchable via a self-hosted OpenRAG instance.

### What It Does

Given a YouTube video URL or playlist/channel URL, the pipeline:

1. **Downloads** the episode audio as a high-quality `.mp3` using `yt-dlp`, extracting structured metadata (title, upload date, video ID, description, canonical URL).
2. **Transcribes** the audio locally using Docling's ASR pipeline backed by OpenAI Whisper Turbo — producing word-level timestamped transcript segments without sending audio to any external API.
3. **Diarizes** the audio using `pyannote.audio` to identify distinct speakers, then merges speaker labels with Whisper's timestamped segments to produce a human-readable labeled transcript (`**[Host]:**` / `**[Guest]:**`).
4. **Constructs** a structured `DoclingDocument` from the labeled transcript, preserving episode metadata and speaker attribution.
5. **Exports** the document to a local Markdown (`.md`) file in the `transcripts/` directory.
6. **Ingests** the Markdown transcript into the self-hosted OpenRAG instance via the async `openrag-sdk`, enabling semantic search and RAG-powered Q&A over all podcast content.
7. **Tracks** ingested episodes in a local `state.json` file keyed by YouTube video ID to prevent duplicate ingestion (idempotency). A `--force` flag overrides this check.

### What It Does NOT Do

- It does not publish or distribute content anywhere.
- It does not use any cloud transcription service (Whisper runs 100% locally).
- It does not modify or re-upload content to YouTube, Spotify, or Apple Podcasts.
- It does not perform real-time or live-stream ingestion.

### Deployment Context

The pipeline runs as a local CLI tool on the user's macOS machine. The only external network calls are:
- **Outbound to YouTube** via `yt-dlp` (audio download)
- **Outbound to `localhost:3000`** (self-hosted OpenRAG Docker container)
- **One-time outbound to Hugging Face Hub** to download the `pyannote.audio` diarization model (requires `HF_TOKEN`)
- **Outbound to OpenAI API** for embedding generation via OpenRAG's configured `text-embedding-3-small` model

---

## 2. System Architecture Diagram

```mermaid
flowchart TD
    CLI[main.py\nCLI Entrypoint\nargparse] --> STATE_CHECK{state.py\nEpisode in\nstate.json?}

    STATE_CHECK -- Yes, no --force --> SKIP[Exit: Already ingested\nSkip episode]
    STATE_CHECK -- No OR --force flag --> ACQ

    subgraph STAGE_1 [Stage 1: Acquire]
        ACQ[acquire.py\nyt-dlp YoutubeDL\nDownload .mp3 + metadata]
    end

    subgraph STAGE_2 [Stage 2: Process Audio in Parallel]
        TRANS[transcribe.py\nDocling DocumentConverter\nAsrPipeline + WHISPER_TURBO\nWord-level segments]
        DIAR[diarize.py\npyannote.audio Pipeline\nSpeaker diarization-3.1\nSpeaker segments]
    end

    subgraph STAGE_3 [Stage 3: Build Document]
        DOC[document.py\nMerge Whisper + pyannote\nMax-overlap assignment\nBuild DoclingDocument\nExport to .md]
    end

    subgraph STAGE_4 [Stage 4: Ingest and Track]
        INGEST[ingest.py\nopenrag-sdk async\nclient.documents.ingest\nfile_path=transcript.md]
        STATE_WRITE[state.py\nWrite video_id entry\nto state.json atomically]
    end

    ACQ -->|.mp3 file path + metadata dict| TRANS
    ACQ -->|.mp3 file path| DIAR
    TRANS -->|List of WhisperSegment| DOC
    DIAR -->|List of DiarizationSegment| DOC
    DOC -->|transcripts/video-id.md| INGEST
    INGEST -->|openrag document_id| STATE_WRITE
    STATE_WRITE --> DONE[Done]

    EXT_YT[(YouTube\nyt-dlp HTTP)] -.->|audio stream| ACQ
    EXT_OR[(OpenRAG\nlocalhost:3000\ntext-embedding-3-small)] -.->|HTTP POST multipart| INGEST
    EXT_HF[(Hugging Face Hub\none-time model download\npyannote/speaker-diarization-3.1)] -.->|model weights| DIAR
```

### Stage Summary Table

| # | Stage | Module | Primary Input | Primary Output |
|---|-------|--------|---------------|----------------|
| 0 | State Check | `state.py` | YouTube video ID | Skip or proceed decision |
| 1 | Acquire | `acquire.py` | YouTube URL | `.mp3` path + `EpisodeMetadata` |
| 2a | Transcribe | `transcribe.py` | `.mp3` path | `List[WhisperSegment]` |
| 2b | Diarize | `diarize.py` | `.mp3` path | `List[DiarizationSegment]` |
| 3 | Build Document | `document.py` | Both segment lists + metadata | `DoclingDocument` + `.md` file |
| 4 | Ingest | `ingest.py` | `.md` file path | OpenRAG `document_id` |
| 5 | Track State | `state.py` | video ID + `document_id` | Updated `state.json` |

---

## 3. Component Specifications

### 3.1 `pipeline/acquire.py` — Audio Acquisition

**Responsibility:** Download a single podcast episode from YouTube as a high-quality `.mp3` file and extract structured metadata.

**Key Inputs:**
- `url: str` — A YouTube video URL (e.g., `https://www.youtube.com/watch?v=VIDEO_ID`) or playlist/channel URL
- `output_dir: Path` — Directory to write the downloaded `.mp3` file

**Key Outputs:**
- `audio_path: Path` — Absolute path to the downloaded `.mp3` file
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

1. **Audio format:** Download as `bestaudio/best` with `postprocessors` converting to `mp3` at 192kbps. This avoids downloading video streams and minimizes disk usage.

2. **Filename convention:** Files are named `{video_id}.mp3` (not the episode title) to avoid filesystem-unsafe characters and to make the idempotency key directly traceable to the file on disk.

3. **Playlist support:** When a playlist or channel URL is provided, `acquire.py` yields one `(audio_path, metadata)` tuple per episode. The caller (`main.py`) iterates and processes each episode independently, allowing partial failures without aborting the entire batch.

4. **yt-dlp options used:**
   ```python
   ydl_opts = {
       "format": "bestaudio/best",
       "outtmpl": str(output_dir / "%(id)s.%(ext)s"),
       "postprocessors": [{
           "key": "FFmpegExtractAudio",
           "preferredcodec": "mp3",
           "preferredquality": "192",
       }],
       "quiet": True,
       "no_warnings": True,
       "extract_flat": False,
   }
   ```

5. **No cookies or authentication:** The pipeline assumes the YouTube channel is public. If private/unlisted videos are needed in the future, `yt-dlp`'s `--cookies-from-browser` mechanism should be used (never hardcoded credentials).

> ⚠️ **Assumption:** `ffmpeg` is installed and available on `PATH` for the MP3 post-processing step. The `README.md` must document this as a system dependency.

---

### 3.2 `pipeline/transcribe.py` — ASR Transcription

**Responsibility:** Transcribe a `.mp3` audio file to a list of timestamped segments using Docling's ASR pipeline with Whisper Turbo.

**Key Inputs:**
- `audio_path: Path` — Path to the `.mp3` file

**Key Outputs:**
- `List[WhisperSegment]` — A list of segment objects, each containing:
  - `start: float` — Segment start time in seconds
  - `end: float` — Segment end time in seconds
  - `text: str` — Transcribed text for this segment

**Key Dependencies:**
- `docling` — `DocumentConverter`, `AsrPipeline`, `AsrPipelineOptions`, `asr_model_specs`, `AudioFormatOption`, `InputFormat`

**Core Implementation Pattern** (from reference `transcribe.py`):
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
result = converter.convert(str(audio_path))
doc = result.document  # DoclingDocument with raw transcript
```

**Design Decisions:**

1. **Model choice — Whisper Turbo:** Selected for its balance of speed and accuracy. It is significantly faster than `whisper-large-v3` while maintaining high transcription quality for English podcast content. The model runs entirely locally via Docling.

2. **Raw `DoclingDocument` from Docling:** The `result.document` returned by Docling's ASR pipeline contains the raw transcript without speaker labels. This document is used as an intermediate to extract timestamped segments before diarization merging in `document.py`.

3. **Segment extraction:** The `WhisperSegment` list is extracted from `result.document` by iterating over Docling's internal segment representation. The exact attribute path must be confirmed against the installed Docling version at implementation time.

> ⚠️ **Unresolved Question:** The exact API for extracting timestamped segments from a `DoclingDocument` produced by `AsrPipeline` (e.g., `doc.texts`, `doc.body.children`, or a dedicated `segments` attribute) must be verified against the Docling source code or documentation. The `result.document` structure for audio inputs may differ from document inputs.

---

### 3.3 `pipeline/diarize.py` — Speaker Diarization

**Responsibility:** Identify speaker-change boundaries in the audio and return a list of timestamped speaker segments.

**Key Inputs:**
- `audio_path: Path` — Path to the `.mp3` file
- `num_speakers: int` — Expected number of speakers (default: `2`)

**Key Outputs:**
- `List[DiarizationSegment]` — A list of segments, each containing:
  - `start: float` — Segment start time in seconds
  - `end: float` — Segment end time in seconds
  - `speaker: str` — Raw pyannote speaker label (e.g., `"SPEAKER_00"`, `"SPEAKER_01"`)

**Key Dependencies:**
- `pyannote.audio` — `Pipeline` class
- `torch` — Required by pyannote for model inference
- `HF_TOKEN` environment variable — Required to download the pretrained model from Hugging Face Hub

**Core Implementation Pattern:**
```python
from pyannote.audio import Pipeline
import torch, os

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=os.environ["HF_TOKEN"]
)

# Use MPS on Apple Silicon, CUDA if available, otherwise CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

pipeline = pipeline.to(device)

diarization = pipeline(str(audio_path), num_speakers=num_speakers)

segments = []
for turn, _, speaker in diarization.itertracks(yield_label=True):
    segments.append(DiarizationSegment(
        start=turn.start,
        end=turn.end,
        speaker=speaker
    ))
```

**Design Decisions:**

1. **Model:** `pyannote/speaker-diarization-3.1` is the current production-quality model. It requires accepting the model's license on Hugging Face Hub before the `HF_TOKEN` will grant access.

2. **`num_speakers=2` default:** For *The Flow* podcast (1 host + 1 guest), constraining to 2 speakers improves diarization accuracy. A `--num-speakers` CLI flag supports edge cases (e.g., panel episodes).

3. **Apple Silicon MPS support:** The pipeline explicitly checks for `torch.backends.mps.is_available()` to leverage Apple Silicon GPU acceleration, which provides significant speedup over CPU for a 60-minute episode.

4. **Model caching:** After the first run, `pyannote.audio` caches the model locally in `~/.cache/torch/pyannote/`. Subsequent runs do not require network access.

> ⚠️ **Assumption:** The user has accepted the `pyannote/speaker-diarization-3.1` model license on Hugging Face Hub at `https://huggingface.co/pyannote/speaker-diarization-3.1`. Without this, the `HF_TOKEN` will be rejected with a 403 error even if the token itself is valid.

---

### 3.4 `pipeline/document.py` — Document Construction & Export

**Responsibility:** Merge Whisper transcript segments with pyannote diarization segments to produce a speaker-labeled transcript, construct a `DoclingDocument`, and export it to a Markdown file.

**Key Inputs:**
- `whisper_segments: List[WhisperSegment]`
- `diarization_segments: List[DiarizationSegment]`
- `metadata: EpisodeMetadata`
- `output_dir: Path` — Directory to write the `.md` file

**Key Outputs:**
- `doc: DoclingDocument` — Fully constructed document with speaker-labeled content
- `md_path: Path` — Path to the exported `.md` file (e.g., `transcripts/{video_id}.md`)

**Key Dependencies:**
- `docling-core` — `DoclingDocument`, `TextItem`, `DocumentOrigin`, `GroupItem`

**Design Decisions:**

1. **Merging strategy:** See [Section 7](#7-diarization-strategy) for the full algorithm. The core approach is timestamp overlap assignment.

2. **Speaker label mapping:** Raw pyannote labels (`SPEAKER_00`, `SPEAKER_01`) are mapped to human-readable labels. The dominant speaker (most total speaking time) is labeled `Host`, the other `Guest`.

3. **Markdown export format:**
   ```markdown
   # {Episode Title}

   **Published:** {upload_date}
   **Source:** {youtube_url}
   **Video ID:** {video_id}

   ---

   **[Host]:** Welcome to The Flow. Today we're talking about...

   **[Guest]:** Thanks for having me...
   ```

4. **Why serialize to `.md` before OpenRAG ingestion:** The `openrag-sdk` does not accept `DoclingDocument` objects directly. Serializing to Markdown also produces a human-readable artifact stored in `transcripts/` for independent use.

> ⚠️ **Unresolved Question:** The exact `docling-core` API for programmatically constructing a `DoclingDocument` (vs. receiving one from a converter) must be verified at implementation time. The constructor signature and required fields for `DocumentOrigin` may change across minor versions.

---

### 3.5 `pipeline/ingest.py` — OpenRAG Ingestion

**Responsibility:** Upload the exported Markdown transcript to the self-hosted OpenRAG instance using the async `openrag-sdk`.

**Key Inputs:**
- `md_path: Path` — Path to the exported `.md` transcript file
- `metadata: EpisodeMetadata` — Used for logging/confirmation

**Key Outputs:**
- `document_id: str` — The OpenRAG document ID assigned to the ingested transcript

**Key Dependencies:**
- `openrag-sdk` — `OpenRAGClient`, `client.documents.ingest()`
- `asyncio` — The SDK is fully async

**Core Implementation Pattern:**
```python
import asyncio, os
from pathlib import Path
from openrag import OpenRAGClient

async def _ingest(md_path: Path) -> str:
    client = OpenRAGClient(
        api_key=os.environ["OPENRAG_API_KEY"],
        base_url=os.environ.get("OPENRAG_URL", "http://localhost:3000"),
    )
    result = await client.documents.ingest(file_path=str(md_path))
    return result.document_id

def ingest_transcript(md_path: Path) -> str:
    return asyncio.run(_ingest(md_path))
```

**Design Decisions:**

1. **Async bridging:** The `openrag-sdk` is fully async. Since the pipeline is otherwise synchronous, `asyncio.run()` bridges the sync/async boundary at the module boundary, avoiding `asyncio` complexity throughout the pipeline.

2. **OpenRAG configuration (confirmed from live instance):**
   - Embedding model: `text-embedding-3-small` (OpenAI)
   - Chunk size: 1000 tokens, overlap: 200 tokens
   - Table structure extraction: enabled
   - OCR: disabled
   For a typical 60-minute podcast episode (~10,000–15,000 words), this produces approximately 30–50 chunks per episode.

3. **Retry logic:** A simple exponential backoff retry (max 3 attempts, delays: 1s, 2s, 4s) wraps the `ingest()` call to handle transient network errors against the local Docker container.

> ⚠️ **Assumption:** The `openrag-sdk` `ingest()` method returns an object with a `document_id` attribute. The exact response schema must be verified against the installed SDK version.

---

### 3.6 `pipeline/state.py` — Idempotency State Tracking

**Responsibility:** Read and write the `state.json` file to track which episodes have been ingested, preventing duplicate processing.

**Key Inputs / Outputs:** See [Section 8](#8-idempotency-design) for the full schema and logic.

**Key Dependencies:** Python standard library only (`json`, `pathlib`, `datetime`, `tempfile`, `os`).

---

### 3.7 `main.py` — CLI Entrypoint

**Responsibility:** Parse CLI arguments, run preflight checks, orchestrate the pipeline stages in order, and handle top-level error reporting.

**CLI Interface:**
```
usage: python main.py [-h] [--force] [--num-speakers N] [--output-dir DIR] [--dry-run] url

positional arguments:
  url                   YouTube video URL or playlist/channel URL

optional arguments:
  --force               Re-ingest even if episode is already in state.json
  --num-speakers N      Number of speakers for diarization (default: 2)
  --output-dir DIR      Directory for downloaded audio (default: ./audio)
  --dry-run             Download and transcribe but do not ingest into OpenRAG
```

**Orchestration Flow:**
```python
for audio_path, metadata in acquire(url, output_dir):
    if state.is_ingested(metadata.video_id) and not args.force:
        logger.info(f"Skipping {metadata.video_id}: already ingested")
        continue

    whisper_segments = transcribe(audio_path)
    diarization_segments = diarize(audio_path, num_speakers=args.num_speakers)
    doc, md_path = build_document(whisper_segments, diarization_segments, metadata)

    if not args.dry_run:
        document_id = ingest(md_path)
        state.mark_ingested(metadata.video_id, document_id, metadata)
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
        └── audio_path (Path → .mp3 file on disk)
                │
                ├─────────────────────────────────────────┐
                │                                         │
                ▼  [transcribe.py — Docling + Whisper]    ▼  [diarize.py — pyannote.audio]
                │                                         │
        List[WhisperSegment]                   List[DiarizationSegment]
          start, end, text                       start, end, speaker
                │                                         │
                └─────────────┬───────────────────────────┘
                              │
                              ▼  [document.py — max-overlap merge]
                              │
                   List[LabeledSegment]
                   start, end, text, speaker_label
                              │
                              ▼  [document.py — consolidate + map labels]
                              │
                   Speaker-labeled paragraphs
                   "**[Host]:** Welcome to The Flow..."
                   "**[Guest]:** Thanks for having me..."
                              │
                              ▼  [document.py — DoclingDocument construction]
                              │
                       DoclingDocument
                         title, origin, body (TextItems)
                              │
                              ▼  [document.py — Markdown export]
                              │
                   transcripts/{video_id}.md
                              │
                              ▼  [ingest.py — openrag-sdk async]
                              │
                   OpenRAG document_id (string)
                              │
                              ▼  [state.py — atomic JSON write]
                              │
                   state.json entry keyed by video_id
```

### Data Transformation Summary

| Stage | Input Type | Output Type | Key Transformation |
|-------|-----------|-------------|-------------------|
| Acquire | `str` URL | `Path` + `EpisodeMetadata` | HTTP stream → MP3 file + JSON metadata |
| Transcribe | `Path` (.mp3) | `List[WhisperSegment]` | Audio waveform → text segments with timestamps |
| Diarize | `Path` (.mp3) | `List[DiarizationSegment]` | Audio waveform → speaker turn boundaries |
| Merge | Two segment lists | `List[LabeledSegment]` | Timestamp overlap → speaker-attributed text |
| Build Doc | `List[LabeledSegment]` + metadata | `DoclingDocument` | Structured Python object |
| Export | `DoclingDocument` | `.md` file | Serialized Markdown text |
| Ingest | `.md` file path | `document_id: str` | HTTP multipart upload → chunked embeddings in OpenRAG |
| Track | `video_id` + `document_id` | `state.json` | JSON file append (atomic) |

---

## 5. Dependencies Table

| Package | Min Version | Purpose | Notes |
|---------|------------|---------|-------|
| `yt-dlp` | `>=2024.1.0` | YouTube audio download and metadata extraction | Update frequently; YouTube API changes often |
| `docling` | `>=2.0.0` | ASR pipeline, `DocumentConverter`, `DoclingDocument` | Bundles Whisper Turbo weights (~1.5 GB) |
| `docling-core` | `>=2.0.0` | `DoclingDocument` data model primitives | Transitive dependency via `docling` |
| `pyannote.audio` | `>=3.1.0` | Speaker diarization via `Pipeline` | Requires HF license acceptance |
| `torch` | `>=2.1.0` | Neural network inference for pyannote and Whisper | Install per-platform (CPU/CUDA/MPS) |
| `openrag-sdk` | `>=0.1.0` | Async client for OpenRAG document ingestion | Fully async; uses `asyncio` |
| `python-dotenv` | `>=1.0.0` | Load `.env` file into environment variables | Dev convenience |
| `ffmpeg` | system dep | MP3 post-processing for yt-dlp audio extraction | Install via `brew install ffmpeg` |

### `pyproject.toml` Dependencies Block

```toml
[project]
name = "audio-to-openrag"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "yt-dlp>=2024.1.0",
    "docling>=2.0.0",
    "pyannote.audio>=3.1.0",
    "torch>=2.1.0",
    "openrag-sdk>=0.1.0",
    "python-dotenv>=1.0.0",
]

[project.scripts]
flow-pipeline = "main:main"
```

> ⚠️ **Note on `torch`:** The correct installation command depends on hardware. For Apple Silicon (MPS): `pip install torch torchvision torchaudio`. For CUDA: use the PyTorch index URL. The `pyproject.toml` specifies `torch>=2.1.0` without a CUDA-specific index to remain portable.

---

## 6. Environment Variables

All secrets and configuration are passed via environment variables. The pipeline uses `python-dotenv` to load a `.env` file from the project root at startup. **The `.env` file must never be committed to version control.**

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HF_TOKEN` | **Yes** | — | Hugging Face Hub access token. Required to download `pyannote/speaker-diarization-3.1`. Obtain at `https://huggingface.co/settings/tokens`. Must have `read` scope. |
| `OPENRAG_API_KEY` | **Yes** | — | API key for the self-hosted OpenRAG instance. Set in the OpenRAG Docker configuration. |
| `OPENRAG_URL` | No | `http://localhost:3000` | Base URL of the OpenRAG instance. Override if running on a non-default port or remote host. |
| `PIPELINE_AUDIO_DIR` | No | `./audio` | Directory where downloaded `.mp3` files are stored. |
| `PIPELINE_TRANSCRIPT_DIR` | No | `./transcripts` | Directory where exported `.md` transcript files are stored. |
| `PIPELINE_STATE_FILE` | No | `./state.json` | Path to the idempotency state file. |
| `PIPELINE_LOG_LEVEL` | No | `INFO` | Python logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |

### `.env.example`

```dotenv
# Hugging Face token (required for pyannote.audio model download)
# Get yours at: https://huggingface.co/settings/tokens
# You must also accept the model license at:
# https://huggingface.co/pyannote/speaker-diarization-3.1
HF_TOKEN=hf_your_token_here

# OpenRAG configuration
OPENRAG_API_KEY=your_openrag_api_key_here
OPENRAG_URL=http://localhost:3000

# Optional pipeline configuration
# PIPELINE_AUDIO_DIR=./audio
# PIPELINE_TRANSCRIPT_DIR=./transcripts
# PIPELINE_STATE_FILE=./state.json
# PIPELINE_LOG_LEVEL=INFO
---

## 7. Diarization Strategy

This section describes the algorithm for merging Whisper ASR segments with pyannote.audio diarization segments to produce a speaker-labeled transcript.

### 7.1 The Core Problem

Whisper and pyannote operate independently on the same audio file and produce two separate, non-aligned segment lists:

- **Whisper segments:** Sentence or phrase-level chunks with start/end timestamps and transcribed text. Boundaries are determined by speech pauses and sentence structure.
- **pyannote segments:** Speaker turn boundaries with start/end timestamps and speaker IDs. Boundaries are determined by voice characteristics and speaker changes.

These two segmentations do not align perfectly. A single Whisper segment may span a speaker change, and a single pyannote segment may contain multiple Whisper segments.

### 7.2 Merge Algorithm — Maximum Overlap Assignment

For each Whisper segment, find the diarization segment with the greatest temporal overlap and assign that speaker label:

```python
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class WhisperSegment:
    start: float
    end: float
    text: str

@dataclass
class DiarizationSegment:
    start: float
    end: float
    speaker: str

@dataclass
class LabeledSegment:
    start: float
    end: float
    text: str
    speaker: str

def merge_segments(
    whisper: List[WhisperSegment],
    diarization: List[DiarizationSegment],
) -> List[LabeledSegment]:
    labeled = []
    for w in whisper:
        best_speaker: Optional[str] = None
        best_overlap: float = 0.0
        for d in diarization:
            overlap = max(0.0, min(w.end, d.end) - max(w.start, d.start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = d.speaker
        labeled.append(LabeledSegment(
            start=w.start,
            end=w.end,
            text=w.text,
            speaker=best_speaker or "UNKNOWN",
        ))
    return labeled
```

**Complexity:** O(W × D) where W = number of Whisper segments and D = number of diarization segments. For a 60-minute episode, W ≈ 500–1000 and D ≈ 200–500, making this approximately 250,000 comparisons — negligible runtime cost.

### 7.3 Speaker Label Mapping — Dominant Speaker Heuristic

Raw pyannote labels (`SPEAKER_00`, `SPEAKER_01`) are mapped to human-readable labels using total speaking time:

```python
from collections import defaultdict

def build_speaker_map(diarization: List[DiarizationSegment]) -> dict[str, str]:
    speaking_time: dict[str, float] = defaultdict(float)
    for seg in diarization:
        speaking_time[seg.speaker] += seg.end - seg.start

    sorted_speakers = sorted(speaking_time, key=speaking_time.__getitem__, reverse=True)

    speaker_map = {}
    if len(sorted_speakers) >= 1:
        speaker_map[sorted_speakers[0]] = "Host"
    if len(sorted_speakers) >= 2:
        speaker_map[sorted_speakers[1]] = "Guest"
    for i, spk in enumerate(sorted_speakers[2:], start=3):
        speaker_map[spk] = f"Speaker {i}"

    return speaker_map
```

**Rationale:** In a standard interview podcast, the host typically accumulates more total speaking time than any single guest (introductions, transitions, questions, wrap-up). This heuristic is correct for the vast majority of episodes.

**Limitation:** If a guest delivers an unusually long monologue, the labels may be swapped. A future `--speaker-map SPEAKER_00=Host,SPEAKER_01=Guest` CLI override addresses this.

### 7.4 Consecutive Segment Consolidation

After speaker assignment, consecutive segments with the same speaker label are merged into a single paragraph to improve readability. Consolidation stops at a speaker change or when the gap between segments exceeds **2.0 seconds** (configurable), indicating a meaningful pause:

```python
CONSOLIDATION_GAP_SECONDS = 2.0

def consolidate_segments(labeled: List[LabeledSegment]) -> List[LabeledSegment]:
    if not labeled:
        return []
    consolidated = [labeled[0]]
    for seg in labeled[1:]:
        prev = consolidated[-1]
        gap = seg.start - prev.end
        if seg.speaker == prev.speaker and gap <= CONSOLIDATION_GAP_SECONDS:
            # Merge into previous segment
            consolidated[-1] = LabeledSegment(
                start=prev.start,
                end=seg.end,
                text=prev.text.rstrip() + " " + seg.text.lstrip(),
                speaker=prev.speaker,
            )
        else:
            consolidated.append(seg)
    return consolidated
```

### 7.5 Output Format

Each consolidated speaker block becomes one Markdown paragraph:

```markdown
**[Host]:** Welcome to The Flow. Today we're talking about AI and its impact on creative work.

**[Guest]:** Thanks for having me. It's a topic I've been thinking about a lot lately.

**[Host]:** Let's start with the basics. How do you define creativity in the context of AI?
```

---

## 8. Idempotency Design

### 8.1 Purpose

The pipeline must be safe to run multiple times against the same YouTube channel or playlist without re-downloading, re-transcribing, or re-ingesting episodes that have already been processed. This is critical for incremental updates (e.g., running weekly to pick up new episodes).

### 8.2 `state.json` Schema

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

### 8.3 State Check and Write Logic

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

### 8.4 `--force` Flag Behavior

When `--force` is passed:
1. The `is_ingested()` check is bypassed — the episode is processed regardless of `state.json`.
2. If the episode was previously ingested, the old `openrag_document_id` is retrieved from state and used to **delete** the existing document from OpenRAG before re-ingesting, preventing duplicate chunks in the knowledge base.
3. The `state.json` entry is overwritten with the new `openrag_document_id`.

> ⚠️ **Assumption:** The `openrag-sdk` provides a `client.documents.delete(document_id)` method. This must be verified against the SDK documentation before implementing `--force` deletion.

---

## 9. Error Handling Strategy

The pipeline uses a **fail-fast per episode, continue across episodes** strategy. A failure in one episode does not abort processing of subsequent episodes in a batch run.

### 9.1 Stage-by-Stage Error Handling

| Stage | Failure Mode | Handling Strategy |
|-------|-------------|-------------------|
| **State Check** | `state.json` corrupt or unreadable | Log `WARNING`, treat as empty state (safe default — worst case is re-processing) |
| **Acquire** | Network error, video unavailable, geo-blocked | Log `ERROR` with video ID, skip episode, continue batch |
| **Acquire** | `ffmpeg` not found on PATH | Raise `SystemExit` with install instructions — fatal configuration error caught at preflight |
| **Acquire** | Disk full | Raise `OSError`, log `CRITICAL`, abort entire run |
| **Transcribe** | Whisper model not yet downloaded | Docling downloads automatically on first use; log progress |
| **Transcribe** | Audio file corrupt or unreadable | Log `ERROR`, skip episode |
| **Transcribe** | Out of memory (very long episode) | Log `ERROR` with suggestion to use a smaller Whisper model variant, skip episode |
| **Diarize** | `HF_TOKEN` missing or invalid | Raise `EnvironmentError` at preflight — caught before any processing begins |
| **Diarize** | pyannote model license not accepted on HF Hub | Raise `RuntimeError` with direct link to `https://huggingface.co/pyannote/speaker-diarization-3.1` |
| **Diarize** | Audio file corrupt | Log `ERROR`, skip episode |
| **Build Document** | Merge produces zero labeled segments | Log `WARNING`, write empty transcript body, continue to ingest (empty doc is valid) |
| **Ingest** | OpenRAG unreachable | Retry 3× with exponential backoff (1s, 2s, 4s); if all fail, log `ERROR`, skip episode — do NOT mark as ingested |
| **Ingest** | OpenRAG returns 4xx error | Log `ERROR` with response body, skip episode |
| **State Write** | Disk full or permission error | Log `CRITICAL`; episode was ingested but state not saved — log the `document_id` to stderr for manual recovery |

### 9.2 Startup Preflight Checks

Before processing any episodes, `main.py` validates the environment:

```python
import shutil, os, re

def preflight_check(url: str) -> None:
    # 1. Required environment variables
    for var in ["HF_TOKEN", "OPENRAG_API_KEY"]:
        if not os.environ.get(var):
            raise EnvironmentError(
                f"Required environment variable '{var}' is not set. "
                f"Copy .env.example to .env and fill in your values."
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

### 9.3 Logging Strategy

- All log output uses Python's standard `logging` module with structured formatting.
- Log level is configurable via `PIPELINE_LOG_LEVEL` env var (default: `INFO`).
- **Secrets are never logged.** The `HF_TOKEN` and `OPENRAG_API_KEY` values are masked in all log output using a custom `logging.Filter`.
- Each episode's processing is wrapped in a `try/except` block that logs the full traceback at `DEBUG` level and a one-line summary at `ERROR` level.
- A final summary is printed at the end of a batch run:
  ```
  Pipeline complete: 5 episodes processed — 4 ingested, 1 skipped (already ingested), 0 failed
  ```

---

## 10. Security Considerations

This section documents security controls aligned with OWASP best practices applicable to a local CLI pipeline.

### 10.1 Secrets Management — OWASP A02: Cryptographic Failures

| Control | Implementation |
|---------|---------------|
| No hardcoded secrets | All tokens and API keys loaded exclusively from environment variables via `python-dotenv` |
| `.env` excluded from VCS | `.gitignore` must include `.env` and `*.env` |
| `.env.example` committed | Provides template without real values; safe to commit |
| Secrets masked in logs | Custom `logging.Filter` replaces known secret values with `[REDACTED]` before any log output |
| No secrets in CLI arguments | Tokens are not accepted as CLI arguments (which appear in `ps` output and shell history) |
| Secrets loaded once at startup | `os.environ` values are read once; not re-read in hot paths |

### 10.2 Input Validation — OWASP A03: Injection

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

### 10.3 Dependency Security — OWASP A06: Vulnerable and Outdated Components

- All dependencies are pinned to minimum versions in `pyproject.toml`.
- Run `pip audit` periodically to scan for known CVEs in the dependency tree.
- `yt-dlp` must be updated frequently (`pip install --upgrade yt-dlp`) as YouTube changes its extraction API regularly; outdated versions will silently fail to download.

### 10.4 Data Privacy Disclosure

| Data | Destination | Notes |
|------|------------|-------|
| Audio files | Local disk only | Never transmitted externally |
| Transcript text | OpenRAG → OpenAI API | OpenRAG uses `text-embedding-3-small`; transcript text is sent to OpenAI for embedding generation |
| Episode metadata | Local `state.json` only | Never transmitted externally |
| `HF_TOKEN` | Hugging Face Hub (one-time) | Used only for model download; not stored by the pipeline |

> ⚠️ **Privacy Note:** Transcript text is sent to OpenAI's API for embedding generation via the configured OpenRAG instance. If the podcast contains sensitive personal information, consider switching the OpenRAG embedding model to a locally-hosted Ollama model to keep all data fully local. The live OpenRAG instance already supports Ollama as an embedding provider.

### 10.5 Local Network Security

- The OpenRAG instance runs on `localhost:3000` inside Docker. It must not be exposed to external networks.
- Verify Docker's port binding uses `127.0.0.1:3000:3000` (not `0.0.0.0:3000:3000`) to prevent external access.
- The `OPENRAG_API_KEY` provides authentication against the local instance.

---

## 11. Known Limitations

| # | Limitation | Impact | Workaround / Mitigation |
|---|-----------|--------|------------------------|
| 1 | **Diarization accuracy degrades with more than 2 speakers** | Panel episodes with 3+ guests may have incorrect or merged speaker labels | Pass `--num-speakers N` to constrain pyannote; manually review transcript |
| 2 | **Host/Guest heuristic may fail for short or atypical episodes** | If a guest speaks more total time than the host, labels may be swapped | Future `--speaker-map SPEAKER_00=Host` CLI override |
| 3 | **Whisper Turbo is optimized for English** | Non-English episodes will have degraded transcription quality | Switch `asr_model_specs` to `WHISPER_LARGE_V3` for multilingual support |
| 4 | **No real-time or live-stream support** | Pipeline only processes completed, uploaded YouTube videos | Not applicable for The Flow's current use case |
| 5 | **Transcript text sent to OpenAI for embeddings** | Privacy implication for sensitive guest conversations | Switch OpenRAG to Ollama embeddings for fully local operation |
| 6 | **No automatic new-episode detection** | Pipeline must be run manually or via cron to pick up new episodes | Add a cron job or scheduled GitHub Actions workflow |
| 7 | **`state.json` is not safe for concurrent access** | Running two pipeline instances simultaneously may corrupt state | Use file locking (`fcntl.flock`) or migrate to SQLite for concurrent access |
| 8 | **Audio files are not cleaned up after ingestion** | Disk usage grows with each episode (~50–100 MB per hour of audio at 192kbps) | Add `--cleanup-audio` flag to delete `.mp3` after successful ingestion |
| 9 | **Docling ASR segment extraction API is not fully documented** | The exact attribute path for timestamped segments from `AsrPipeline` output may change across Docling versions | Pin `docling` to a specific version; add integration tests against the pinned version |
| 10 | **Whisper segment boundaries may split mid-sentence across speaker turns** | A single Whisper segment spanning a speaker change will be attributed to whichever speaker has the most overlap — the minority speaker's words in that segment will be misattributed | Acceptable for podcast use case; could be mitigated with word-level timestamps if Docling exposes them |
| 11 | **No transcript quality scoring** | There is no automated way to detect poor transcription quality (e.g., heavy background noise, strong accents) | Manual review of `transcripts/` directory; future enhancement could add a confidence score |

---

## 12. Future Enhancements

These are optional improvements that are out of scope for v1.0 but worth tracking for future iterations.

| Priority | Enhancement | Description |
|----------|------------|-------------|
| High | **Automatic new-episode detection** | Poll the YouTube channel RSS feed or use `yt-dlp`'s `--dateafter` flag to automatically detect and process new episodes without manual invocation |
| High | **Word-level timestamp diarization** | If Docling exposes word-level timestamps from Whisper, use them instead of segment-level timestamps for more precise speaker boundary alignment |
| High | **`--speaker-map` CLI override** | Allow manual specification of speaker labels: `--speaker-map SPEAKER_00=Host,SPEAKER_01=Alice` to correct heuristic misassignments |
| Medium | **Local embeddings via Ollama** | Switch OpenRAG's embedding model from `text-embedding-3-small` (OpenAI) to a locally-hosted Ollama model (e.g., `nomic-embed-text`) to keep all data fully local and eliminate OpenAI API costs |
| Medium | **Metadata enrichment at ingestion** | Pass structured metadata (episode date, guest name, topic tags) to OpenRAG at ingestion time once the SDK supports custom metadata fields, enabling filtered RAG queries (e.g., "What did Alice say about X?") |
| Medium | **Audio cleanup flag** | Add `--cleanup-audio` to delete `.mp3` files after successful ingestion to manage disk usage |
| Medium | **SQLite state backend** | Replace `state.json` with a SQLite database for concurrent-safe access, richer querying, and easier migration |
| Medium | **Transcript quality validation** | Add a post-transcription step that checks for common Whisper failure modes (e.g., hallucinated repetitions, very low word count relative to duration) and flags episodes for manual review |
| Low | **Spotify/Apple Podcasts as fallback source** | If a YouTube video is unavailable (deleted, private), fall back to downloading from Spotify or Apple Podcasts using their respective APIs |
| Low | **Cloud deployment** | Package the pipeline as a Docker container with a scheduled trigger (GitHub Actions, AWS EventBridge) for fully automated weekly ingestion |
| Low | **Web UI for transcript review** | A simple local web interface (e.g., Streamlit) to browse ingested transcripts, correct speaker labels, and re-ingest edited transcripts |
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
│   ├── transcribe.py        ← Docling ASR + Whisper Turbo transcription
│   ├── diarize.py           ← pyannote.audio speaker diarization
│   ├── document.py          ← Segment merge, DoclingDocument build, .md export
│   ├── ingest.py            ← openrag-sdk async ingestion
│   └── state.py             ← Idempotency state tracking (state.json)
├── audio/                   ← Downloaded .mp3 files (gitignored)
├── transcripts/             ← Exported .md transcript files
└── state.json               ← Ingestion state tracker (gitignored)
```

---

*End of Architecture Document — The Flow Pipeline v1.0.0*