# 🎙️ The Flow — Podcast Ingestion Pipeline

![Python](https://img.shields.io/badge/python-3.12+-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![Whisper](https://img.shields.io/badge/ASR-Whisper%20Turbo-orange) ![OpenRAG](https://img.shields.io/badge/RAG-OpenRAG-purple)

A local, end-to-end CLI pipeline that downloads podcast episodes from YouTube, transcribes them with speaker diarization, and ingests the results into a self-hosted [OpenRAG](https://github.com/openrag/openrag) instance — making every episode searchable and retrievable via RAG.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Usage](#usage)
5. [How It Works](#how-it-works)
6. [Output Format](#output-format)
7. [Known Limitations](#known-limitations)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

Before you begin, ensure the following are in place:

### System Requirements

- **Python 3.12+**
- **ffmpeg** — required by `yt-dlp` for MP3 conversion

  ```bash
  # macOS
  brew install ffmpeg

  # Ubuntu / Debian
  sudo apt install ffmpeg

  # Windows (via Chocolatey)
  choco install ffmpeg
  ```

  Verify it's on your PATH:

  ```bash
  ffmpeg -version
  ```

### OpenRAG Instance

A running OpenRAG instance is required for ingestion. The easiest way is via Docker:

```bash
docker run -d -p 3000:3000 openrag/openrag:latest
```

The pipeline defaults to `http://localhost:3000`. See the [Configuration](#configuration) section to point it elsewhere.

### Hugging Face Token & pyannote License

Speaker diarization uses the `pyannote/speaker-diarization-3.1` model, which requires:

1. A [Hugging Face account](https://huggingface.co/join)
2. Accepting the model license at:  
   👉 **https://huggingface.co/pyannote/speaker-diarization-3.1**
3. A Hugging Face access token (read scope is sufficient):  
   👉 **https://huggingface.co/settings/tokens**

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-org/audio_to_openrag.git
cd audio_to_openrag

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 3. Install the package and all dependencies
pip install -e .

# 4. Set up your environment variables
cp .env.example .env
# Open .env in your editor and fill in the required values
```

---

## Configuration

All configuration is done via environment variables. Copy `.env.example` to `.env` and populate the values:

```bash
cp .env.example .env
```

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENRAG_API_KEY` | **Yes** | — | API key from your OpenRAG instance |
| `OPENRAG_URL` | No | `http://localhost:3000` | URL of your OpenRAG instance |
| `HF_TOKEN` | **Yes** | — | Hugging Face token for `pyannote.audio` model access |
| `AUDIO_DIR` | No | `./audio` | Directory where downloaded `.mp3` files are stored |
| `TRANSCRIPTS_DIR` | No | `./transcripts` | Directory where generated `.md` transcripts are saved |
| `STATE_FILE` | No | `./state.json` | Path to the idempotency state file |
| `LOG_LEVEL` | No | `INFO` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

> **Note:** `AUDIO_DIR` and `TRANSCRIPTS_DIR` are gitignored by default. Do not commit audio files or transcripts to version control.

---

## Usage

> **Important:** Always ensure your virtual environment is activated before running commands:
> ```bash
> source .venv/bin/activate  # macOS / Linux
> # .venv\Scripts\activate   # Windows
> ```
> You can verify the correct Python is active with: `which python` (should show `.venv/bin/python`)

### Ingest Command

The `ingest` command downloads, transcribes, diarizes, and uploads one or more episodes to OpenRAG.

```bash
# Ingest a single episode by YouTube URL
python main.py ingest "https://www.youtube.com/watch?v=VIDEO_ID"

# Ingest all episodes from a channel or playlist
python main.py ingest "https://www.youtube.com/@YourChannel"

# Force re-ingest — deletes the existing document from OpenRAG first
python main.py ingest "https://www.youtube.com/watch?v=VIDEO_ID" --force

# Dry run — transcribes locally but does NOT upload to OpenRAG
python main.py ingest "https://www.youtube.com/watch?v=VIDEO_ID" --dry-run

# Specify the expected number of speakers for better diarization accuracy
python main.py ingest "https://www.youtube.com/watch?v=VIDEO_ID" --num-speakers 2
```

| Flag | Description |
|---|---|
| `--force` | Re-ingests even if the episode is already tracked in `state.json` |
| `--dry-run` | Runs the full pipeline locally but skips the OpenRAG upload step |
| `--num-speakers N` | Hints to pyannote how many speakers to expect (improves accuracy) |

### Status Command

Prints a formatted table of all episodes tracked in `state.json`:

```bash
python main.py status
```

Example output:

```
┌─────────────────────────────────┬──────────────────────┬────────────┬──────────┐
│ Title                           │ YouTube URL          │ Ingested   │ Doc ID   │
├─────────────────────────────────┼──────────────────────┼────────────┼──────────┤
│ Episode 42: The Flow of Ideas   │ youtube.com/watch?.. │ 2024-01-15 │ doc_abc  │
│ Episode 41: Deep Work           │ youtube.com/watch?.. │ 2024-01-08 │ doc_xyz  │
└─────────────────────────────────┴──────────────────────┴────────────┴──────────┘
```

---

## How It Works

The pipeline executes the following steps in sequence:

1. **Acquire** (`pipeline/acquire.py`) — Downloads the YouTube audio as an `.mp3` file using `yt-dlp`, skipping episodes already present in `state.json`.
2. **Transcribe** (`pipeline/transcribe.py`) — Runs local ASR on the audio using Docling's `DocumentConverter` with the Whisper Turbo backend, producing timestamped word segments.
3. **Diarize** (`pipeline/diarize.py`) — Applies `pyannote/speaker-diarization-3.1` to the audio to assign speaker labels ("Speaker 1", "Speaker 2", etc.) to each segment.
4. **Build Document** (`pipeline/document.py`) — Merges ASR segments with diarization output into a structured `DoclingDocument` and exports it as a `.md` transcript file.
5. **Ingest** (`pipeline/ingest.py`) — Uploads the transcript to the configured OpenRAG instance via `openrag-sdk`, storing the returned document ID.
6. **Track State** (`pipeline/state.py`) — Writes the episode metadata and OpenRAG document ID to `state.json` to prevent duplicate ingestion on future runs.

---

## Output Format

Each processed episode produces a Markdown transcript in `./transcripts/`. Here is an example:

```markdown
# Episode Title Here

**Channel:** The Flow  
**Date:** January 15, 2024  
**YouTube:** https://youtube.com/watch?v=abc123

---

## Transcript

**[Speaker 1]:** Welcome back to The Flow. Today we're diving into...

**[Speaker 2]:** Thanks for having me. I've been thinking a lot about...

**[Speaker 1]:** That's a great point. Let's unpack that...
```

These `.md` files are also what gets ingested into OpenRAG, making the full transcript text available for semantic search and retrieval-augmented generation.

---

## Known Limitations

- **Diarization accuracy** — `pyannote` speaker diarization is probabilistic. Accuracy degrades with overlapping speech, background noise, or more than ~4 speakers. Providing `--num-speakers` when known significantly improves results.
- **Whisper language bias** — Whisper Turbo performs best on English. Non-English or code-switched audio may produce lower-quality transcripts.
- **OpenRAG must be running** — The pipeline has no retry logic for a downed OpenRAG instance. If the service is unavailable, the ingest step will fail (the transcript is still saved locally).
- **YouTube-only** — There is no support for Spotify, Apple Podcasts, RSS feeds, or direct audio file uploads. Only YouTube URLs (videos, channels, playlists) are supported.
- **Local compute required** — Both Whisper Turbo and pyannote run locally. A machine with a CUDA-capable GPU is strongly recommended for reasonable throughput on long episodes.

---

## Troubleshooting

### `ffmpeg not found` or `FileNotFoundError: ffmpeg`

`yt-dlp` requires `ffmpeg` to convert downloaded audio to MP3. Install it and ensure it is on your system `PATH`:

```bash
# macOS
brew install ffmpeg

# Verify
ffmpeg -version
```

---

### `401 Unauthorized` from Hugging Face

You have not accepted the license for the `pyannote/speaker-diarization-3.1` model, or your `HF_TOKEN` is missing/invalid.

1. Visit **https://huggingface.co/pyannote/speaker-diarization-3.1** and accept the license agreement.
2. Ensure `HF_TOKEN` is set correctly in your `.env` file.
3. Confirm the token has at least **read** scope at **https://huggingface.co/settings/tokens**.

---

### `Connection refused` to `localhost:3000`

OpenRAG is not running. Start it with Docker:

```bash
docker run -d -p 3000:3000 openrag/openrag:latest

# Verify it's up
curl http://localhost:3000/health
```

If you're running OpenRAG at a different address, set `OPENRAG_URL` in your `.env`.

---

### `OPENRAG_API_KEY not set`

The pipeline will refuse to run without an API key. Copy the example env file and fill in your key:

```bash
cp .env.example .env
# Edit .env and set OPENRAG_API_KEY=your_key_here
```

---

### `faster-whisper is not installed` or import errors

If you see warnings about faster-whisper not being available, or import errors for mlx-whisper, this usually means you're running the script with the system Python instead of the virtual environment Python.

**Solution:**

1. Ensure your virtual environment is activated:
   ```bash
   source .venv/bin/activate  # macOS / Linux
   # .venv\Scripts\activate   # Windows
   ```

2. Verify you're using the correct Python:
   ```bash
   which python  # Should show: /path/to/audio_to_openrag/.venv/bin/python
   ```

3. If the venv is activated but imports still fail, reinstall dependencies:
   ```bash
   pip install -e .
   ```

---

### Diarization logs `segments=None`

Docling ASR did not return structured word segments for this audio file, and the pipeline fell back to a simpler segment extraction strategy. The transcript will still be generated, but speaker boundaries may be less precise. Check the logs at `DEBUG` level for more detail:

```bash
LOG_LEVEL=DEBUG python main.py ingest "https://www.youtube.com/watch?v=VIDEO_ID"
```

---

## License

MIT © The Flow