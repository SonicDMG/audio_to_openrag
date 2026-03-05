# 🎙️ Audio/Video to OpenRAG Pipeline

<div align="center">

<img src="public/logos/youtube_logo.png" alt="YouTube" height="60"/>
<img src="public/arrow.svg" alt="→" height="60" style="margin: 0 15px; vertical-align: bottom;"/>
<img src="public/logos/docling_logo.svg" alt="Docling" height="60"/>
<img src="public/arrow.svg" alt="→" height="60" style="margin: 0 15px; vertical-align: bottom;"/>
<img src="public/logos/openrag-logo-dog.svg" alt="OpenRAG" height="60"/>

*Download videos from YouTube • Transcribe with timestamps using Whisper (via Docling) • Ingest into OpenRAG*

![Python](https://img.shields.io/badge/python-3.12+-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![Whisper](https://img.shields.io/badge/ASR-Whisper%20Turbo-orange) ![OpenRAG](https://img.shields.io/badge/RAG-OpenRAG-purple)

</div>

## 📖 About This Project

**This project demonstrates a complete video-to-RAG pipeline** using **OpenAI's Whisper** for transcription (accessed via Docling's ASR wrapper) and **[OpenRAG](https://github.com/langflow-ai/openrag)** for semantic search and retrieval.

We use **[Docling](https://github.com/DS4SD/docling)**'s `DocumentConverter` and `AsrPipeline` as a convenient interface to Whisper, benefiting from Docling's structured `DoclingDocument` format for consistent processing and export. This creates an end-to-end pipeline from YouTube videos to queryable knowledge bases with timestamp-aware transcripts.

## 🎯 Why These Tools?

### Whisper for Transcription (via Docling's Wrapper)

**[OpenAI's Whisper](https://github.com/openai/whisper)** is the state-of-the-art speech recognition model, and we access it through **[Docling](https://github.com/DS4SD/docling)**'s convenient wrapper:

- **🎙️ Whisper Turbo** — Fast, accurate multilingual speech-to-text from OpenAI
- **🎥 Video Support** — Docling handles video files directly (extracts audio internally via ffmpeg)
- **⏱️ Timestamp Precision** — Automatic timestamp extraction for temporal navigation
- **📄 Unified API** — Docling's `DocumentConverter` provides a consistent interface across formats
- **🏗️ Structure Preservation** — Results returned as structured `DoclingDocument` objects
- **🔧 Production Ready** — Docling is IBM Research's enterprise-grade document processing framework

**Why use Docling as a wrapper?** It provides a clean, unified API and structured document format (`DoclingDocument`) that simplifies downstream processing, export, and integration with RAG systems.

### OpenRAG for Streamlined RAG

**[OpenRAG](https://github.com/langflow-ai/openrag)** provides a focused, modern approach to RAG:

- **🎯 Purpose-Built** — Designed specifically for retrieval-augmented generation workflows
- **🏷️ Knowledge Filters** — First-class support for scoped searches and content organization
- **⚡ Lightweight** — Minimal dependencies, clean API, fast iteration
- **🔄 Simple Lifecycle** — Straightforward document ingestion, updates, and deletion
- **📊 Developer-Friendly** — Intuitive SDK without the complexity of larger frameworks

### What This Pipeline Demonstrates

- **Video-to-RAG Pipeline** — Complete workflow from YouTube to queryable knowledge base
- **Best-of-Breed Tools** — Whisper for transcription, Docling for structure, OpenRAG for search
- **Semantic Video Search** — Find content by meaning with timestamp links back to source
- **Scalable Ingestion** — Process entire channels or playlists efficiently with state management


## 🛠️ Technology Stack

This pipeline is built entirely on **open source** platforms:

### 🎯 Whisper via Docling (Video Transcription)
- **Whisper Turbo** — OpenAI's fast, accurate speech recognition model
- **AsrPipeline** — Docling's wrapper providing convenient access to Whisper
- **DocumentConverter** — Unified interface for processing audio/video files
- Returns structured `DoclingDocument` objects with timestamps

### 🚀 OpenRAG SDK (Document Ingestion)
- **documents.ingest()** — Uploads transcripts with automatic chunking and embedding
- **knowledge_filters** — Query-time filtering to scope searches to podcast content
- **documents.delete()** — Enables re-ingestion with `--force` flag

### 🔧 Supporting Tools
- **yt-dlp** — YouTube audio download
- **ffmpeg** — Audio format conversion
- **Rich** — Beautiful CLI progress output

📚 **[View detailed component documentation →](docs/TECHNOLOGY_STACK.md)**

## 🙏 Attribution

This project builds upon the excellent work by [Tejas Kumar](https://github.com/TejasQ/example-docling-media), which demonstrates how to use Docling's ASR wrapper for media transcription. We've extended this pattern to create an end-to-end ingestion pipeline for OpenRAG with state management, batch processing, and timestamp preservation.

**Original work:** https://github.com/TejasQ/example-docling-media
**Transcription engine:** [OpenAI Whisper](https://github.com/openai/whisper)

---

## 🚀 Quick Start

### ✅ Prerequisites

1. **Python 3.12+**
2. **ffmpeg** (system dependency - required by Docling for video/audio processing)
   ```bash
   # macOS
   brew install ffmpeg

   # Ubuntu/Debian
   sudo apt install ffmpeg
   ```
3. **OpenRAG instance** running (default: `http://localhost:3000`)
   - See setup instructions at: https://github.com/langflow-ai/openrag

### 📦 Installation

```bash
# Clone and enter directory
git clone https://github.com/your-org/audio_to_openrag.git
cd audio_to_openrag

# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies with uv
uv sync

# Configure environment
cp .env.example .env
# Edit .env and add your OPENRAG_API_KEY
```

### 💻 Usage

```bash
# Ingest a single video
uv run python main.py ingest "https://www.youtube.com/watch?v=VIDEO_ID"

# Ingest all videos from a channel
uv run python main.py ingest "https://www.youtube.com/@YourChannel"

# Ingest a playlist
uv run python main.py ingest "https://www.youtube.com/playlist?list=PLAYLIST_ID"

# Ingest with a custom knowledge filter
uv run python main.py ingest "https://www.youtube.com/watch?v=VIDEO_ID" --filter "TechTalks"

# Check ingestion status
uv run python main.py status

# Remove a video from state (after manually deleting from OpenRAG)
uv run python main.py remove VIDEO_ID
```

**⚙️ Useful flags:**
- `--force` — Re-ingest even if already processed (deletes from OpenRAG first)
- `--dry-run` — Transcribe locally without uploading to OpenRAG
- `--filter TEXT` — OpenRAG knowledge filter name (default: "Videos")
  - Associates ingested content with a named knowledge filter in OpenRAG
  - Enables scoped semantic searches (e.g., search only within "Videos" content)
  - The filter is automatically created/updated when documents are ingested
  - Use different filter names to organize content by topic, channel, or category

**📊 Managing State:**

The pipeline tracks processed videos in `state.json` to prevent duplicate work. If you manually delete a document from OpenRAG and want to re-ingest it:

1. **Option 1:** Use the `remove` command to clear it from state, then re-run `ingest`
   ```bash
   uv run python main.py remove VIDEO_ID
   uv run python main.py ingest "https://www.youtube.com/watch?v=VIDEO_ID"
   ```

2. **Option 2:** Use `--force` flag (deletes from OpenRAG first, then re-ingests)
   ```bash
   uv run python main.py ingest "https://www.youtube.com/watch?v=VIDEO_ID" --force
   ```

---

## ⚙️ Configuration

Edit `.env` with these required variables:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENRAG_API_KEY` | **Yes** | — | Your OpenRAG API key |
| `OPENRAG_URL` | No | `http://localhost:3000` | OpenRAG instance URL |
| `AUDIO_DIR` | No | `./audio` | Where to save downloaded audio |
| `TRANSCRIPTS_DIR` | No | `./transcripts` | Where to save transcripts |

---

## 🔄 How It Works

1. **Download** — Fetches video from YouTube (supports MP4, WebM, etc.)
2. **Transcribe** — Uses Whisper Turbo (via Docling's ASR wrapper) for speech-to-text with timestamp extraction
3. **Export** — Creates dual formats:
   - **DocTags** (`.doctags`) — Structure-preserving format for reference
   - **Markdown** (`.md`) — Human-readable with `[MM:SS]` timestamps for OpenRAG
4. **Ingest** — Uploads Markdown transcript to OpenRAG for semantic search
5. **Track** — Saves state to prevent duplicate processing

Output transcripts are saved in `./transcripts/` as both DocTags and Markdown files.

**Note:** Transcripts include timestamps but not speaker labels. Speaker diarization is not implemented due to processing time costs for long-form content.

---

## 🔧 Troubleshooting

**`ffmpeg not found`**
- Install ffmpeg and ensure it's in your PATH

**`Connection refused` to localhost:3000**
- Ensure OpenRAG is running (see https://github.com/langflow-ai/openrag for installation)

**Import errors or missing dependencies**
- Reinstall dependencies: `uv sync`
- Try running with: `uv run python main.py ingest <url>`

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

Copyright © 2026 SonicDMG
