"""
pipeline/ingest.py — OpenRAG Ingestion Stage

Uploads a Markdown transcript file to a self-hosted OpenRAG instance using
the OpenRAGClient wrapper. The wrapper handles all async/sync bridging,
retry logic, and knowledge filter management.

Security controls (OWASP):
- OPENRAG_API_KEY loaded exclusively from environment variables; never logged
- AuthenticationError raised with a helpful message that does not expose the key
- Only the filename (not the full path) is logged
"""

from __future__ import annotations

import logging
from pathlib import Path

from pipeline.config import get_openrag_api_key, get_openrag_url
from pipeline.openrag_client import IngestResult, OpenRAGClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ingest_transcript(
    transcript_path: Path,
    force: bool = False,
    filter_name: str = "Videos",
) -> IngestResult:
    """Upload a Markdown transcript to OpenRAG and return the result.

    Uses the OpenRAGClient wrapper which handles async/sync bridging,
    retry logic with exponential backoff, and knowledge filter management.

    When *force* is True, the function first attempts to delete any existing
    document with the same filename from OpenRAG before re-ingesting, to
    prevent duplicate chunks in the knowledge base.

    The ``document_id`` field in the returned :class:`IngestResult` contains
    the OpenRAG ``task_id`` from the ingestion response, which serves as the
    stable reference stored in ``state.json``.

    Args:
        transcript_path: Path to the ``.md`` transcript file to upload.
        force:           If True, delete the existing OpenRAG document (if any)
                         before re-ingesting. Default: False.
        filter_name:     Name of the OpenRAG knowledge filter to create/update.
                         Default: "Videos".

    Returns:
        An :class:`IngestResult` with status "success", "already_exists",
        or "failed".

    Raises:
        FileNotFoundError: If *transcript_path* does not exist.
        RuntimeError:      If openrag-sdk is not installed, OPENRAG_API_KEY is
                           missing, or authentication fails.
    """
    if not transcript_path.exists():
        raise FileNotFoundError(
            f"Transcript file not found: {transcript_path.name}"
        )

    logger.info(
        "Starting OpenRAG ingestion for: %s (force=%s, filter=%s)",
        transcript_path.name,
        force,
        filter_name,
    )

    # Get configuration
    api_key = get_openrag_api_key()
    if not api_key:
        raise RuntimeError(
            "OPENRAG_API_KEY environment variable is not set. "
            "Add it to your .env file."
        )

    base_url = get_openrag_url() or "http://localhost:3000"

    # Initialize OpenRAG client wrapper
    client = OpenRAGClient(api_key=api_key, base_url=base_url)

    # Force delete if requested
    if force:
        client.delete_document(transcript_path.name)

    # Ingest document with automatic retry logic
    result = client.ingest_document(transcript_path, wait=True, filter_name=filter_name)

    # Log final result
    if result.status == "success":
        logger.info(
            "✅ Ingestion complete: document_id=%s filter_id=%s",
            result.document_id,
            result.filter_id,
        )
    elif result.status == "already_exists":
        logger.info("ℹ️  Document already exists in OpenRAG (skipped)")
    else:
        logger.error(
            "❌ Ingestion failed: %s",
            result.error or "Unknown error",
        )

    return result
