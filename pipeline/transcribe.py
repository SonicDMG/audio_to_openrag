"""
pipeline/transcribe.py — ASR Transcription Stage

Transcribes a local audio file using Docling's ASR pipeline backed by
OpenAI Whisper Turbo. Returns both the raw DoclingDocument and a list of
timestamped segments for downstream diarization alignment.

NOTE — Unresolved API question:
    The exact attribute path for extracting timestamped segments from a
    DoclingDocument produced by AsrPipeline is not fully documented as of
    Docling 2.x. This module attempts to extract segments from the document's
    internal representation. If Docling does not expose them, a fallback using
    faster-whisper is provided. See _extract_segments() for details.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

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
        segments: List of timestamped text segments for diarization alignment.
                  Each dict has keys: ``start`` (float), ``end`` (float),
                  ``text`` (str). May be ``None`` if Docling does not expose
                  segment-level timestamps for the installed version.
    """

    document: object  # DoclingDocument — typed as object to avoid hard import at module level
    segments: list[dict] | None  # {"start": float, "end": float, "text": str}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_segments_from_docling(document: object) -> list[dict] | None:
    """Attempt to extract timestamped segments from a Docling ASR DoclingDocument.

    Docling's AsrPipeline stores ASR output in the document body. The exact
    internal structure depends on the installed Docling version. This function
    tries several known attribute paths in order of preference.

    # TODO — UNRESOLVED API QUESTION:
    # The Docling ASR pipeline (as of 2.x) stores segments in the document's
    # ``texts`` list as TextItem objects. Each TextItem may carry provenance
    # information (prov) with timing data. The exact schema must be verified
    # against the installed Docling version. If this extraction fails, the
    # fallback _extract_segments_via_faster_whisper() is used instead.

    Args:
        document: A DoclingDocument returned by DocumentConverter.convert().

    Returns:
        List of segment dicts, or None if extraction is not possible.
    """
    segments: list[dict] = []

    # Attempt 1: iterate document.texts (TextItem list in docling-core)
    try:
        texts = getattr(document, "texts", None)
        if texts:
            for item in texts:
                # TextItem may have a prov attribute with page/bbox info,
                # but for ASR output it may carry timing metadata instead.
                # Try common attribute names for start/end times.
                start = None
                end = None
                text = getattr(item, "text", None) or ""

                # Try direct timing attributes
                for start_attr in ("start", "start_time", "t_start"):
                    val = getattr(item, start_attr, None)
                    if val is not None:
                        start = float(val)
                        break

                for end_attr in ("end", "end_time", "t_end"):
                    val = getattr(item, end_attr, None)
                    if val is not None:
                        end = float(val)
                        break

                # Try prov-based timing (docling-core provenance objects)
                if start is None or end is None:
                    prov_list = getattr(item, "prov", None)
                    if prov_list:
                        prov = prov_list[0] if isinstance(prov_list, list) else prov_list
                        if start is None:
                            for attr in ("start", "start_time", "t_start"):
                                val = getattr(prov, attr, None)
                                if val is not None:
                                    start = float(val)
                                    break
                        if end is None:
                            for attr in ("end", "end_time", "t_end"):
                                val = getattr(prov, attr, None)
                                if val is not None:
                                    end = float(val)
                                    break

                if text and start is not None and end is not None:
                    segments.append({"start": start, "end": end, "text": text})

            if segments:
                logger.debug(
                    "Extracted %d segments from document.texts.", len(segments)
                )
                return segments
    except Exception as exc:
        logger.debug("document.texts extraction failed: %s", exc)

    # Attempt 2: iterate document body children
    try:
        body = getattr(document, "body", None)
        if body:
            children = getattr(body, "children", None) or []
            for child_ref in children:
                # Resolve reference if needed
                child = getattr(child_ref, "obj", child_ref)
                text = getattr(child, "text", None) or ""
                start = getattr(child, "start", None)
                end = getattr(child, "end", None)
                if text and start is not None and end is not None:
                    segments.append(
                        {"start": float(start), "end": float(end), "text": text}
                    )
            if segments:
                logger.debug(
                    "Extracted %d segments from document.body.children.", len(segments)
                )
                return segments
    except Exception as exc:
        logger.debug("document.body.children extraction failed: %s", exc)

    logger.warning(
        "Could not extract timestamped segments from DoclingDocument. "
        "Falling back to faster-whisper for segment extraction."
    )
    return None


def _extract_segments_via_faster_whisper(audio_path: Path) -> list[dict]:
    """Fallback: extract timestamped segments using faster-whisper directly.

    This is used when Docling's AsrPipeline does not expose segment-level
    timestamps in the DoclingDocument. faster-whisper is a transitive
    dependency of Docling, so it should always be available.

    Args:
        audio_path: Path to the audio file to transcribe.

    Returns:
        List of segment dicts with keys: start, end, text.

    Raises:
        RuntimeError: If faster-whisper is not available and Docling did not
                      expose segments — diarization cannot proceed without
                      timestamped segments.
    """
    try:
        from faster_whisper import WhisperModel  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(
            "Neither Docling nor faster-whisper could provide timestamped segments. "
            "Install faster-whisper: pip install faster-whisper\n"
            f"Original error: {exc}"
        ) from exc

    logger.info(
        "Running faster-whisper fallback transcription on: %s", audio_path.name
    )
    # Use the same model size as Docling's WHISPER_TURBO spec
    model = WhisperModel("turbo", device="auto", compute_type="default")
    raw_segments, _ = model.transcribe(str(audio_path), beam_size=5)

    segments: list[dict] = []
    for seg in raw_segments:
        segments.append(
            {"start": float(seg.start), "end": float(seg.end), "text": seg.text}
        )

    logger.info(
        "faster-whisper fallback produced %d segments.", len(segments)
    )
    return segments


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def transcribe_audio(audio_path: Path) -> TranscriptResult:
    """Transcribe an audio file using Docling's ASR pipeline (Whisper Turbo).

    Runs the full Docling DocumentConverter with AsrPipeline and
    WHISPER_TURBO model spec. The resulting DoclingDocument is returned
    alongside a list of timestamped segments for diarization alignment.

    If Docling does not expose segment-level timestamps in the returned
    DoclingDocument, this function falls back to running faster-whisper
    directly to obtain the segment list.

    Args:
        audio_path: Path to the local .mp3 (or other audio) file to transcribe.

    Returns:
        A :class:`TranscriptResult` containing the DoclingDocument and
        the list of timestamped segments (or None if unavailable).

    Raises:
        FileNotFoundError: If *audio_path* does not exist.
        RuntimeError:      If transcription fails.
    """
    audio_path = audio_path.resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    logger.info("Starting Docling ASR transcription: %s", audio_path.name)

    # Import Docling here to keep module-level imports clean and allow the
    # module to be imported even if Docling is not yet installed (e.g., during
    # testing with mocks).
    try:
        from docling.datamodel import asr_model_specs  # type: ignore[import]
        from docling.datamodel.base_models import InputFormat  # type: ignore[import]
        from docling.datamodel.pipeline_options import AsrPipelineOptions  # type: ignore[import]
        from docling.document_converter import (  # type: ignore[import]
            AudioFormatOption,
            DocumentConverter,
        )
        from docling.pipeline.asr_pipeline import AsrPipeline  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(
            f"Docling is not installed or could not be imported: {exc}\n"
            "Install with: pip install 'docling[asr]>=2.74.0'"
        ) from exc

    try:
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

        logger.info("Running Docling converter (this may take several minutes)…")
        # Pass a Path object, NOT a str. When a str is passed, Docling's
        # _DocumentConversionInput.docs() calls resolve_source_to_stream()
        # which converts it to a BytesIO DocumentStream, losing the directory
        # component. Passing a Path bypasses that branch and preserves the
        # full absolute path through to the MLX/Whisper ffmpeg call.
        result = converter.convert(audio_path)
        document = result.document

    except Exception as exc:
        raise RuntimeError(
            f"Docling ASR transcription failed for '{audio_path.name}': {exc}"
        ) from exc

    logger.info("Docling transcription complete. Extracting segments…")

    # Attempt to extract timestamped segments from the DoclingDocument
    segments = _extract_segments_from_docling(document)

    if segments is None:
        # Fallback: re-run with faster-whisper to get timestamped segments
        try:
            segments = _extract_segments_via_faster_whisper(audio_path)
        except RuntimeError as exc:
            logger.warning(
                "Segment extraction fallback also failed: %s. "
                "Diarization will not be possible for this episode.",
                exc,
            )
            segments = None

    segment_count = len(segments) if segments else 0
    logger.info(
        "Transcription stage complete: %d segments extracted.", segment_count
    )

    return TranscriptResult(document=document, segments=segments)

# Made with Bob
