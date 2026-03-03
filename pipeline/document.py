"""
pipeline/document.py — Document Construction & Export Stage

Takes a list of DiarizedSegment objects, episode metadata, and a DoclingDocument,
then exports the document to both DocTags (for OpenRAG ingestion) and Markdown
(for human readability) formats.

The DocTags format preserves document structure for better RAG performance.
The Markdown file provides human-readable output.

Security controls (OWASP):
- Filenames are sanitized (alphanumeric + underscores, max 60 chars)
- No secrets are logged or embedded in output files
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from pipeline.acquire import EpisodeAudio
from pipeline.config import get_transcripts_dir

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class DiarizedSegment:
    """A single speaker-attributed transcript segment.

    Attributes:
        speaker_label: Human-readable label, e.g. "Speaker 1", "Speaker 2".
                      Empty string for transcripts without speaker labels.
        start:         Segment start time in seconds.
        end:           Segment end time in seconds.
        text:          Transcribed text for this segment.
    """

    speaker_label: str
    start: float
    end: float
    text: str

# Maximum characters for the title portion of the output filename
_MAX_TITLE_CHARS = 60


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sanitize_title_for_filename(title: str) -> str:
    """Convert an episode title to a filesystem-safe string.

    Transforms the title to lowercase, replaces spaces and hyphens with
    underscores, strips all non-alphanumeric/underscore characters, and
    truncates to _MAX_TITLE_CHARS.

    Args:
        title: Raw episode title string.

    Returns:
        Sanitized, lowercase, underscore-separated string safe for use in
        a filename.
    """
    sanitized = title.lower()
    sanitized = re.sub(r"[\s\-]+", "_", sanitized)
    sanitized = re.sub(r"[^a-z0-9_]", "", sanitized)
    sanitized = sanitized[:_MAX_TITLE_CHARS]
    sanitized = sanitized.strip("_")
    return sanitized or "episode"


def _format_upload_date(upload_date: str) -> str:
    """Format a YYYYMMDD date string to a human-readable form.

    Args:
        upload_date: Date string in YYYYMMDD format (from yt-dlp).

    Returns:
        Formatted date string (e.g. "2024-01-15"), or the original string
        if it cannot be parsed.
    """
    if len(upload_date) == 8 and upload_date.isdigit():
        return f"{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:]}"
    return upload_date


def _build_markdown(
    episode: EpisodeAudio,
    segments: list[DiarizedSegment],
) -> str:
    """Build the full Markdown transcript string.

    Produces a document with a metadata header followed by the diarized
    transcript. Each speaker block is rendered with timestamps and text.

    Args:
        episode:  Episode metadata (title, channel, date, URL).
        segments: Consolidated, speaker-labeled transcript segments with timestamps.

    Returns:
        Complete Markdown string ready to write to disk.
    """
    formatted_date = _format_upload_date(episode.upload_date)

    lines: list[str] = [
        f"# {episode.title}",
        "",
        f"**Channel:** {episode.channel}",
        f"**Date:** {formatted_date}",
        f"**YouTube:** {episode.webpage_url}",
        "",
        "---",
        "",
        "## Transcript",
        "",
    ]

    if not segments:
        lines.append("*No transcript segments available.*")
    else:
        for seg in segments:
            # Format timestamp as [MM:SS] or [H:MM:SS]
            timestamp = _format_timestamp(seg.start)

            if seg.speaker_label:
                # Format with speaker label and timestamp: **[Speaker 1] [0:05]:** text
                lines.append(f"**[{seg.speaker_label}] [{timestamp}]:** {seg.text}")
            else:
                # No speaker label (diarization skipped) — timestamp + paragraph
                lines.append(f"**[{timestamp}]** {seg.text}")
            lines.append("")  # blank line between blocks

    return "\n".join(lines)


def _format_timestamp(seconds: float) -> str:
    """Format a timestamp in seconds to MM:SS or H:MM:SS format.

    Args:
        seconds: Timestamp in seconds (can be float).

    Returns:
        Formatted timestamp string (e.g., "5:23" or "1:32:45").
    """
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_transcript_document(
    episode: EpisodeAudio,
    segments: list[DiarizedSegment],
    docling_document: object,
    transcripts_dir: Path | None = None,
) -> tuple[Path, Path, object]:
    """Export a DoclingDocument to DocTags and Markdown formats.

    Exports the provided DoclingDocument to both DocTags format (optimized for
    OpenRAG ingestion with preserved structure) and Markdown format (for human
    readability). Both files are written to *transcripts_dir*.

    Filename format: ``{video_id}_{sanitized_title}.{doctags|md}``

    Args:
        episode:          Episode metadata from the acquisition stage.
        segments:         Consolidated diarized segments (used for logging only).
        docling_document: The DoclingDocument to export (from transcription stage).
        transcripts_dir:  Directory to write the output files. Defaults to the
                          ``TRANSCRIPTS_DIR`` environment variable, or
                          ``./transcripts`` if unset.

    Returns:
        A tuple of ``(doctags_path, md_path, docling_document)`` where:
        - *doctags_path* is the absolute path to the DocTags file
        - *md_path* is the absolute path to the Markdown file
        - *docling_document* is the original document (passed through)

    Raises:
        RuntimeError: If either export operation or file write fails.
    """
    if transcripts_dir is None:
        transcripts_dir = get_transcripts_dir()

    transcripts_dir.mkdir(parents=True, exist_ok=True)

    # Build base filename: {video_id}_{sanitized_title}
    sanitized_title = _sanitize_title_for_filename(episode.title)
    base_filename = f"{episode.video_id}_{sanitized_title}"

    logger.info(
        "Exporting transcript document for '%s' → %s.*",
        episode.title[:60],
        base_filename,
    )

    # Export to DocTags format (RAG-optimized with structure preservation)
    doctags_path = transcripts_dir / f"{base_filename}.doctags"
    try:
        logger.info("Exporting to DocTags format for OpenRAG...")
        # Type is 'object' to avoid hard dependency on docling types
        # AttributeError is caught below if method doesn't exist
        doctags_content = docling_document.export_to_doctags()  # type: ignore[attr-defined]
        doctags_path.write_text(doctags_content, encoding="utf-8")
        logger.info(
            "DocTags file written: %s (%d bytes)",
            doctags_path.name,
            len(doctags_content),
        )
    except AttributeError as exc:
        raise RuntimeError(
            f"DoclingDocument does not support export_to_doctags(): {exc}"
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            f"Failed to write DocTags file '{doctags_path.name}': {exc}"
        ) from exc
    except Exception as exc:  # pylint: disable=broad-exception-caught
        raise RuntimeError(
            f"Unexpected error during DocTags export: {exc}"
        ) from exc

    # Export to Markdown format (human-readable with timestamps)
    md_path = transcripts_dir / f"{base_filename}.md"
    try:
        logger.info("Exporting to Markdown format with timestamps...")
        # Use our custom markdown builder that includes timestamps from segments
        markdown_content = _build_markdown(episode, segments)
        md_path.write_text(markdown_content, encoding="utf-8")
        logger.info(
            "Markdown file written: %s (%d bytes)",
            md_path.name,
            len(markdown_content),
        )
    except OSError as exc:
        raise RuntimeError(
            f"Failed to write Markdown file '{md_path.name}': {exc}"
        ) from exc
    except Exception as exc:  # pylint: disable=broad-exception-caught
        raise RuntimeError(
            f"Unexpected error during Markdown export: {exc}"
        ) from exc

    logger.info(
        "Document export complete: %d segments → DocTags + Markdown",
        len(segments),
    )

    return doctags_path, md_path, docling_document


# DEPRECATED: This function is no longer needed since we now receive the
# DoclingDocument directly from the transcription stage instead of rebuilding
# it from Markdown. Kept here temporarily for reference during migration.
#
# def _convert_markdown_to_docling(md_path: Path) -> object:
#     """Convert a Markdown file to a DoclingDocument using Docling.
#
#     Uses Docling's DocumentConverter with InputFormat.MD to produce a
#     DoclingDocument from the written Markdown transcript file.
#
#     Args:
#         md_path: Path to the Markdown file to convert.
#
#     Returns:
#         A DoclingDocument instance, or a lightweight fallback object if
#         Docling is not available.
#     """
#     try:
#         from docling.datamodel.base_models import InputFormat  # type: ignore[import]  # pylint: disable=import-outside-toplevel
#         from docling.document_converter import DocumentConverter  # type: ignore[import]  # pylint: disable=import-outside-toplevel
#     except ImportError as exc:
#         logger.warning(
#             "Docling not available for Markdown conversion: %s. "
#             "Returning a stub document object.",
#             exc,
#         )
#         return _StubDocument(md_path)
#
#     try:
#         # Log Docling usage - Markdown processing doesn't require heavy ML models
#         # (unlike PDF processing which uses OCR, layout analysis, table detection)
#         logger.info(
#             "Using Docling DocumentConverter for Markdown processing "
#             "(lightweight parser, no ML models required)"
#         )
#
#         converter = DocumentConverter(
#             allowed_formats=[InputFormat.MD],
#         )
#         result = converter.convert(str(md_path))
#         logger.debug("Docling Markdown conversion complete for: %s", md_path.name)
#         return result.document
#     except Exception as exc:  # pylint: disable=broad-exception-caught
#         logger.warning(
#             "Docling failed to convert Markdown '%s': %s. "
#             "Returning a stub document object.",
#             md_path.name,
#             exc,
#         )
#         return _StubDocument(md_path)


class _StubDocument:
    """Minimal fallback when Docling is unavailable or conversion fails.

    Stores the Markdown path so downstream code can still access the content.
    This is intentionally minimal — the Markdown file is the primary artifact.
    
    NOTE: This class is now deprecated since we receive DoclingDocument directly
    from the transcription stage. Kept for backward compatibility during migration.
    """

    def __init__(self, md_path: Path) -> None:
        self.md_path = md_path
        self.export_to_markdown = lambda: md_path.read_text(encoding="utf-8")
        # Add export_to_doctags for compatibility with new interface
        self.export_to_doctags = lambda: md_path.read_text(encoding="utf-8")

    def __repr__(self) -> str:
        return f"_StubDocument(md_path={self.md_path!r})"
