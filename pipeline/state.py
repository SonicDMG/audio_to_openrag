"""
pipeline/state.py — Idempotency State Tracking

Reads and writes state.json to track which episodes have been ingested,
preventing duplicate processing across pipeline runs.

Uses atomic writes (write to .tmp then os.replace) to prevent corruption
if the process is interrupted mid-write.

state.json schema:
{
  "schema_version": "1.0",
  "last_updated": "2026-02-27T17:00:00Z",
  "episodes": {
    "VIDEO_ID": {
      "video_id": "VIDEO_ID",
      "title": "Episode Title",
      "channel": "The Flow",
      "upload_date": "20240115",
      "webpage_url": "https://youtube.com/watch?v=VIDEO_ID",
      "transcript_path": "./transcripts/VIDEO_ID_title.md",
      "openrag_document_id": "task_abc123",
      "ingested_at": "2026-02-27T17:00:00Z",
      "pipeline_version": "0.1.0"
    }
  }
}

Security controls (OWASP):
- video_id validated to [a-zA-Z0-9_-] before any write (input validation)
- No secrets stored in state.json
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from pipeline.acquire import EpisodeAudio

logger = logging.getLogger(__name__)

# Pipeline version stamped into each state entry for future migration support
_PIPELINE_VERSION = "0.1.0"

# Regex for validating video IDs before writing to state
_SAFE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_\-]+$")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_video_id(video_id: str) -> str:
    """Validate that *video_id* contains only filesystem/JSON-safe characters.

    Args:
        video_id: YouTube video ID to validate.

    Returns:
        The validated video ID.

    Raises:
        ValueError: If the video ID contains unexpected characters.
    """
    if not _SAFE_ID_PATTERN.match(video_id):
        raise ValueError(
            f"Invalid video_id for state write: {video_id!r}. "
            "Only alphanumeric characters, hyphens, and underscores are allowed."
        )
    return video_id


def _load_state(state_file: Path) -> dict:
    """Load and parse state.json, returning an empty state dict on any error.

    Args:
        state_file: Path to the state.json file.

    Returns:
        Parsed state dict, or a fresh empty state if the file does not exist
        or is corrupt.
    """
    if not state_file.exists():
        return _empty_state()

    try:
        raw = state_file.read_text(encoding="utf-8")
        data = json.loads(raw)
        if not isinstance(data, dict) or "episodes" not in data:
            logger.warning(
                "state.json has unexpected structure — treating as empty state."
            )
            return _empty_state()
        return data
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(
            "state.json could not be read (%s) — treating as empty state.", exc
        )
        return _empty_state()


def _empty_state() -> dict:
    """Return a fresh, empty state dictionary."""
    return {
        "schema_version": "1.0",
        "last_updated": "",
        "episodes": {},
    }


def _atomic_write(state_file: Path, data: dict) -> None:
    """Write *data* as JSON to *state_file* atomically.

    Writes to a temporary file in the same directory, then uses os.replace()
    for an atomic rename. This prevents partial writes from corrupting the
    state file if the process is interrupted.

    Args:
        state_file: Destination path for the state file.
        data:       Dict to serialise as JSON.

    Raises:
        RuntimeError: If the write or rename fails.
    """
    state_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=state_file.parent,
            suffix=".tmp",
            delete=False,
            encoding="utf-8",
        ) as tmp_f:
            json.dump(data, tmp_f, indent=2, ensure_ascii=False)
            tmp_path = tmp_f.name

        os.replace(tmp_path, state_file)
        logger.debug("state.json written atomically to: %s", state_file.name)

    except OSError as exc:
        raise RuntimeError(
            f"Failed to write state file '{state_file}': {exc}\n"
            "The episode may have been ingested but state was not saved. "
            "Check the logs for the document_id and add it manually if needed."
        ) from exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def is_ingested(video_id: str, state_file: Path | None = None) -> bool:
    """Check whether an episode has already been ingested.

    Args:
        video_id:   YouTube video ID to look up.
        state_file: Path to state.json. Defaults to the ``STATE_FILE``
                    environment variable, or ``./state.json`` if unset.

    Returns:
        True if the video ID is present in state.json, False otherwise.
    """
    if state_file is None:
        state_file = Path(os.environ.get("STATE_FILE", "./state.json"))

    state = _load_state(state_file)
    return video_id in state.get("episodes", {})


def mark_ingested(
    episode: EpisodeAudio,
    transcript_path: Path,
    openrag_document_id: str | None,
    state_file: Path | None = None,
) -> None:
    """Record a successfully ingested episode in state.json.

    Writes an entry keyed by the episode's video_id. If an entry already
    exists for this video_id, it is overwritten (supports --force re-ingest).

    Args:
        episode:             Episode metadata from the acquisition stage.
        transcript_path:     Path to the written Markdown transcript file.
        openrag_document_id: The task_id / document ID returned by OpenRAG,
                             or None if ingestion was skipped (dry-run).
        state_file:          Path to state.json. Defaults to the ``STATE_FILE``
                             environment variable, or ``./state.json`` if unset.

    Raises:
        ValueError:   If the episode's video_id contains unsafe characters.
        RuntimeError: If the state file cannot be written.
    """
    if state_file is None:
        state_file = Path(os.environ.get("STATE_FILE", "./state.json"))

    video_id = _validate_video_id(episode.video_id)

    state = _load_state(state_file)
    now_iso = datetime.now(timezone.utc).isoformat()

    state["episodes"][video_id] = {
        "video_id": video_id,
        "title": episode.title,
        "channel": episode.channel,
        "upload_date": episode.upload_date,
        "webpage_url": episode.webpage_url,
        "transcript_path": str(transcript_path),
        "openrag_document_id": openrag_document_id,
        "ingested_at": now_iso,
        "pipeline_version": _PIPELINE_VERSION,
    }
    state["last_updated"] = now_iso

    _atomic_write(state_file, state)
    logger.info(
        "State updated: video_id=%s openrag_document_id=%s",
        video_id,
        openrag_document_id,
    )


def get_all(state_file: Path | None = None) -> dict:
    """Return the full contents of state.json as a dict.

    Args:
        state_file: Path to state.json. Defaults to the ``STATE_FILE``
                    environment variable, or ``./state.json`` if unset.

    Returns:
        The parsed state dict. Returns an empty state structure if the file
        does not exist or is corrupt.
    """
    if state_file is None:
        state_file = Path(os.environ.get("STATE_FILE", "./state.json"))

    return _load_state(state_file)

# Made with Bob
