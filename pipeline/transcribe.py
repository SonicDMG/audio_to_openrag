"""
pipeline/transcribe.py — ASR Transcription Stage

Transcribes a local audio file using ONLY Docling's DocumentConverter with
AsrPipeline, following the pattern from:
https://github.com/TejasQ/example-docling-media/blob/main/transcribe.py

Returns a DoclingDocument and exports to Markdown. No segments are returned
since diarization is skipped in this refactored version.

Security (OWASP):
  - audio_path is resolved and existence-checked before any model call
  - No shell=True invocations; all model calls use Python APIs
  - No secrets or credentials handled in this module
"""

from __future__ import annotations

import logging
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class TranscriptResult:
    """Result of the ASR transcription stage.

    Attributes:
        document: The raw DoclingDocument produced by Docling's AsrPipeline.
                  Contains the full transcript text without speaker labels.
        markdown: The transcript exported to Markdown format.
        model_info: String describing the transcription backend used.
    """

    document: object  # DoclingDocument — typed as object to avoid hard import at module level
    markdown: str
    model_info: str = "Docling AsrPipeline - whisper-turbo"


# ---------------------------------------------------------------------------
# Progress tracking helper
# ---------------------------------------------------------------------------


class _ProgressTracker:
    """Thread-safe progress tracker for simulating transcription progress.

    Since Docling's DocumentConverter doesn't provide progress callbacks,
    this class simulates smooth progress updates based on elapsed time and
    estimated audio duration.
    """

    def __init__(self, callback: Callable[[float], None] | None, audio_path: Path):
        self.callback = callback
        self.audio_path = audio_path
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.current_progress = 0.0
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start the progress tracking thread."""
        if not self.callback:
            return

        # Signal start
        self.callback(0.0)

        # Start background thread for progress updates
        self.thread = threading.Thread(target=self._update_progress, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        """Stop the progress tracking thread and signal completion."""
        if self.thread:
            self.stop_event.set()
            self.thread.join(timeout=1.0)

        if self.callback:
            self.callback(1.0)

    def _get_audio_duration(self) -> float:
        """Estimate audio duration in seconds using ffprobe if available."""
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(self.audio_path)],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip())
        except (FileNotFoundError, ValueError, subprocess.TimeoutExpired):
            pass

        # Fallback: estimate based on file size (rough approximation)
        # Assume ~1MB per minute for typical audio files
        try:
            file_size_mb = self.audio_path.stat().st_size / (1024 * 1024)
            return file_size_mb * 60  # seconds
        except (FileNotFoundError, OSError):
            # If file doesn't exist or can't be accessed, use a default estimate
            return 300.0  # 5 minutes default

    def _update_progress(self) -> None:
        """Background thread that updates progress based on elapsed time."""
        start_time = time.time()
        estimated_duration = self._get_audio_duration()

        # Transcription typically takes 0.1-0.3x of audio duration with whisper-turbo
        # Use a conservative estimate of 0.25x (4x real-time speed)
        estimated_transcription_time = estimated_duration * 0.25

        # Cap minimum time at 5 seconds to avoid too-fast progress
        estimated_transcription_time = max(estimated_transcription_time, 5.0)

        while not self.stop_event.is_set():
            elapsed = time.time() - start_time

            # Use a sigmoid-like curve for smooth progress
            # Progress accelerates initially, then slows as it approaches completion
            raw_progress = elapsed / estimated_transcription_time

            # Apply easing function: fast start, slow finish (stays under 95% until done)
            if raw_progress < 0.5:
                # First half: accelerate to 50%
                progress = raw_progress * 1.0
            else:
                # Second half: slow down, asymptotically approach 95%
                progress = 0.5 + (raw_progress - 0.5) * 0.9

            # Cap at 95% until actual completion
            progress = min(progress, 0.95)

            with self._lock:
                self.current_progress = progress

            if self.callback:
                self.callback(progress)

            # Update every 0.5 seconds for smooth visual feedback
            self.stop_event.wait(0.5)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def transcribe_audio(
    audio_path: Path,
    progress_callback: Callable[[float], None] | None = None,
) -> TranscriptResult:
    """Transcribe an audio file using Docling's DocumentConverter with AsrPipeline.

    This follows the exact pattern from the TejasQ reference implementation:
    https://github.com/TejasQ/example-docling-media/blob/main/transcribe.py

    Args:
        audio_path: Path to the local .mp3 (or other audio) file to transcribe.
        progress_callback: Optional callback function that receives progress (0.0-1.0)
                          during transcription. Progress is estimated based on
                          audio duration and typical transcription speed.

    Returns:
        A :class:`TranscriptResult` containing the DoclingDocument,
        the markdown export, and model information string.

    Raises:
        FileNotFoundError: If *audio_path* does not exist.
        RuntimeError:      If transcription fails.
    """
    # --- Input validation (OWASP: validate all inputs before processing) ---
    audio_path = audio_path.resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    logger.info(
        "Starting Docling transcription for '%s'",
        audio_path.name,
    )

    # Initialize progress tracker
    progress_tracker = _ProgressTracker(progress_callback, audio_path)
    progress_tracker.start()

    try:
        # pylint: disable=import-outside-toplevel
        from docling.datamodel import asr_model_specs
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import AsrPipelineOptions
        from docling.document_converter import AudioFormatOption, DocumentConverter
        from docling.pipeline.asr_pipeline import AsrPipeline
    except ImportError as exc:
        raise RuntimeError(
            "Docling is not installed or not importable in the current Python environment.\n"
            "Install with: pip install 'docling[asr]'\n"
            f"Original error: {exc}"
        ) from exc

    # Configure Docling pipeline with WHISPER_TURBO model
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

    logger.info("Running Docling DocumentConverter with AsrPipeline (whisper-turbo)…")

    try:
        # Pass a Path object, NOT a str. When a str is passed, Docling's
        # _DocumentConversionInput.docs() calls resolve_source_to_stream()
        # which converts it to a BytesIO DocumentStream, losing the directory
        # component. Passing a Path bypasses that branch and preserves the
        # full absolute path through to the MLX/Whisper ffmpeg call.
        result = converter.convert(audio_path)
        document = result.document

        # Export to markdown
        markdown = document.export_to_markdown()

        logger.info("Docling transcription complete for '%s'.", audio_path.name)

        # Stop progress tracker and signal completion
        progress_tracker.stop()

        return TranscriptResult(
            document=document,
            markdown=markdown,
            model_info="Docling AsrPipeline - whisper-turbo"
        )

    except Exception as exc:
        # Ensure progress tracker is stopped on error
        progress_tracker.stop()

        logger.error(
            "Docling transcription failed for '%s': %s",
            audio_path.name,
            exc,
        )
        raise RuntimeError(f"Docling transcription failed: {exc}") from exc


