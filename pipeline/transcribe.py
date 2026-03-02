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
                          during transcription. Note: Docling doesn't provide
                          granular progress, so this will be called with 0.0 at
                          start and 1.0 at completion.

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

    # Signal start of transcription
    if progress_callback:
        progress_callback(0.0)

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

        # Signal completion
        if progress_callback:
            progress_callback(1.0)

        return TranscriptResult(
            document=document,
            markdown=markdown,
            model_info="Docling AsrPipeline - whisper-turbo"
        )

    except Exception as exc:
        logger.error(
            "Docling transcription failed for '%s': %s",
            audio_path.name,
            exc,
        )
        raise RuntimeError(f"Docling transcription failed: {exc}") from exc


