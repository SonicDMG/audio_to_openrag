# 🎙️ Audio to OpenRAG Pipeline

<div align="center">

<img src="public/logos/youtube_logo.png" alt="YouTube" height="60"/>
<img src="public/arrow.svg" alt="→" height="60" style="margin: 0 15px; vertical-align: bottom;"/>
<img src="public/logos/docling_logo.svg" alt="Docling" height="60"/>
<img src="public/arrow.svg" alt="→" height="60" style="margin: 0 15px; vertical-align: bottom;"/>
<img src="public/logos/openrag-logo-dog.svg" alt="OpenRAG" height="60"/>

*Download audio from YouTube • Transcribe with Docling • Ingest into OpenRAG*

![Python](https://img.shields.io/badge/python-3.12+-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![Whisper](https://img.shields.io/badge/ASR-Whisper%20Turbo-orange) ![OpenRAG](https://img.shields.io/badge/RAG-OpenRAG-purple)

</div>

Download audio from any YouTube video, playlist, or channel, transcribe it using **[Docling](https://github.com/DS4SD/docling)**, and ingest it into **[OpenRAG](https://github.com/langflow-ai/openrag)** for semantic search and retrieval-augmented generation.

## Technology Stack

This pipeline is built entirely on **open source** platforms:

### 🎯 Docling (Audio Transcription)
- **AsrPipeline** — Docling's audio processing pipeline with Whisper Turbo backend
- **DocumentConverter** — Unified converter for audio and Markdown processing
- Handles speech-to-text conversion and structured document output

### 🚀 OpenRAG SDK (Document Ingestion)
- **documents.ingest()** — Uploads transcripts with automatic chunking and embedding
- **knowledge_filters** — Query-time filtering to scope searches to podcast content
- **documents.delete()** — Enables re-ingestion with `--force` flag

### 🔧 Supporting Tools
- **yt-dlp** — YouTube audio download
- **ffmpeg** — Audio format conversion
- **Rich** — Beautiful CLI progress output

📚 **[View detailed component documentation →](docs/TECHNOLOGY_STACK.md)**

## Attribution

This project builds upon the excellent audio processing work by [Tejas Kumar](https://github.com/TejasQ/example-docling-media). The core audio transcription pipeline was derived from his example, which demonstrates how to use Docling for media transcription. We've extended it to create an end-to-end ingestion pipeline for OpenRAG.

**Original work:** https://github.com/TejasQ/example-docling-media

---

## Quick Start

### Prerequisites

1. **Python 3.12+**
2. **ffmpeg** (for audio conversion)
   ```bash
   # macOS
   brew install ffmpeg
   
   # Ubuntu/Debian
   sudo apt install ffmpeg
   ```
3. **OpenRAG instance** running (default: `http://localhost:3000`)
   - Docling is included with the OpenRAG installation
   - See setup instructions at: https://github.com/langflow-ai/openrag

### Installation

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

### Usage

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

**Useful flags:**
- `--force` — Re-ingest even if already processed (deletes from OpenRAG first)
- `--dry-run` — Transcribe locally without uploading to OpenRAG
- `--filter TEXT` — OpenRAG knowledge filter name (default: "Videos")
  - Associates ingested content with a named knowledge filter in OpenRAG
  - Enables scoped semantic searches (e.g., search only within "Videos" content)
  - The filter is automatically created/updated when documents are ingested
  - Use different filter names to organize content by topic, channel, or category

**Managing State:**

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

## Configuration

Edit `.env` with these required variables:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENRAG_API_KEY` | **Yes** | — | Your OpenRAG API key |
| `OPENRAG_URL` | No | `http://localhost:3000` | OpenRAG instance URL |
| `AUDIO_DIR` | No | `./audio` | Where to save downloaded audio |
| `TRANSCRIPTS_DIR` | No | `./transcripts` | Where to save transcripts |

---

## How It Works

1. **Download** — Fetches audio from YouTube as MP3
2. **Transcribe** — Uses Docling with Whisper Turbo backend for speech-to-text (plain text, no speaker labels)
3. **Format** — Creates a structured Markdown transcript
4. **Ingest** — Uploads to OpenRAG for semantic search
5. **Track** — Saves state to prevent duplicate processing

Output transcripts are saved in `./transcripts/` as Markdown files.

**Note:** The current implementation produces plain text transcripts without speaker identification or labels. All audio is transcribed as continuous text.

---

## Troubleshooting

**`ffmpeg not found`**
- Install ffmpeg and ensure it's in your PATH

**`Connection refused` to localhost:3000**
- Ensure OpenRAG is running (see https://github.com/langflow-ai/openrag for installation)

**Import errors or missing dependencies**
- Reinstall dependencies: `uv sync`
- Try running with: `uv run python main.py ingest <url>`

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

Copyright © 2026 SonicDMG