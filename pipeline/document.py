"""
pipeline/document.py — Document Construction & Export Stage

Takes a list of DiarizedSegment objects and episode metadata, builds a
well-structured Markdown transcript, writes it to disk, and returns both
the file path and a DoclingDocument built from the Markdown.

The Markdown file is the primary artifact ingested into OpenRAG. The
DoclingDocument is returned for any downstream programmatic use.

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
    transcript. Each speaker block is rendered as a bold label followed by
    the spoken text.

    Args:
        episode:  Episode metadata (title, channel, date, URL).
        segments: Consolidated, speaker-labeled transcript segments.

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
            if seg.speaker_label:
                # Format with speaker label: **[Speaker 1]:** text content
                lines.append(f"**[{seg.speaker_label}]:** {seg.text}")
            else:
                # No speaker label (diarization skipped) — plain paragraph
                lines.append(seg.text)
            lines.append("")  # blank line between blocks

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_transcript_document(
    episode: EpisodeAudio,
    segments: list[DiarizedSegment],
    transcripts_dir: Path | None = None,
) -> tuple[Path, object]:
    """Build a Markdown transcript and return its path plus a DoclingDocument.

    Writes the diarized transcript to a ``.md`` file in *transcripts_dir*,
    then re-converts the Markdown file using Docling's DocumentConverter to
    produce a DoclingDocument for any downstream programmatic use.

    Filename format: ``{video_id}_{sanitized_title}.md``

    Args:
        episode:         Episode metadata from the acquisition stage.
        segments:        Consolidated diarized segments from the diarization
                         stage.
        transcripts_dir: Directory to write the ``.md`` file. Defaults to the
                         ``TRANSCRIPTS_DIR`` environment variable, or
                         ``./transcripts`` if unset.

    Returns:
        A tuple of ``(md_path, docling_document)`` where *md_path* is the
        absolute path to the written Markdown file and *docling_document* is
        the DoclingDocument produced by re-converting the Markdown.

    Raises:
        RuntimeError: If the Markdown file cannot be written or if Docling
                      fails to convert the Markdown.
    """
    if transcripts_dir is None:
        transcripts_dir = get_transcripts_dir()

    transcripts_dir.mkdir(parents=True, exist_ok=True)

    # Build filename: {video_id}_{sanitized_title}.md
    sanitized_title = _sanitize_title_for_filename(episode.title)
    filename = f"{episode.video_id}_{sanitized_title}.md"
    md_path = transcripts_dir / filename

    logger.info(
        "Building transcript document for '%s' → %s",
        episode.title[:60],
        filename,
    )

    # Build Markdown content
    markdown_content = _build_markdown(episode, segments)

    # Write Markdown to disk
    try:
        md_path.write_text(markdown_content, encoding="utf-8")
        logger.info("Transcript written: %s (%d bytes)", filename, len(markdown_content))
    except OSError as exc:
        raise RuntimeError(
            f"Failed to write transcript file '{filename}': {exc}"
        ) from exc

    # Convert the Markdown file to a DoclingDocument using Docling
    # Option A: Write .md → re-convert with DocumentConverter (confirmed to work)
    docling_document = _convert_markdown_to_docling(md_path)

    return md_path, docling_document


def _convert_markdown_to_docling(md_path: Path) -> object:
    """Convert a Markdown file to a DoclingDocument using Docling.

    Uses Docling's DocumentConverter with InputFormat.MD to produce a
    DoclingDocument from the written Markdown transcript file.

    Args:
        md_path: Path to the Markdown file to convert.

    Returns:
        A DoclingDocument instance, or a lightweight fallback object if
        Docling is not available.
    """
    try:
        from docling.datamodel.base_models import InputFormat  # type: ignore[import]  # pylint: disable=import-outside-toplevel
        from docling.document_converter import DocumentConverter  # type: ignore[import]  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        logger.warning(
            "Docling not available for Markdown conversion: %s. "
            "Returning a stub document object.",
            exc,
        )
        return _StubDocument(md_path)

    try:
        # Log Docling usage - Markdown processing doesn't require heavy ML models
        # (unlike PDF processing which uses OCR, layout analysis, table detection)
        logger.info(
            "Using Docling DocumentConverter for Markdown processing "
            "(lightweight parser, no ML models required)"
        )

        converter = DocumentConverter(
            allowed_formats=[InputFormat.MD],
        )
        result = converter.convert(str(md_path))
        logger.debug("Docling Markdown conversion complete for: %s", md_path.name)
        return result.document
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.warning(
            "Docling failed to convert Markdown '%s': %s. "
            "Returning a stub document object.",
            md_path.name,
            exc,
        )
        return _StubDocument(md_path)


class _StubDocument:
    """Minimal fallback when Docling is unavailable or conversion fails.

    Stores the Markdown path so downstream code can still access the content.
    This is intentionally minimal — the Markdown file is the primary artifact.
    """

    def __init__(self, md_path: Path) -> None:
        self.md_path = md_path
        self.export_to_markdown = lambda: md_path.read_text(encoding="utf-8")

    def __repr__(self) -> str:
        return f"_StubDocument(md_path={self.md_path!r})"

