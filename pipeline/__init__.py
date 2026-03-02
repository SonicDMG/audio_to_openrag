"""
The Flow Pipeline — podcast ingestion pipeline.

Stages:
  1. acquire   — Download audio from YouTube via yt-dlp
  2. transcribe — Transcribe audio using Docling ASR (Whisper Turbo)
  3. diarize   — Identify speakers using pyannote.audio
  4. document  — Build a DoclingDocument with diarized transcript
  5. ingest    — Upload transcript to OpenRAG via openrag-sdk
  6. state     — Track ingested episodes for idempotency
"""

