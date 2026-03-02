"""
pipeline/ingest.py — OpenRAG Ingestion Stage

Uploads a Markdown transcript file to a self-hosted OpenRAG instance using
the async openrag-sdk. Bridges the async SDK into the synchronous pipeline
via asyncio.run().

SDK response schema (verified against installed openrag-sdk):
  - ingest() with wait=True  → IngestTaskStatus(task_id, status, ...)
  - ingest() with wait=False → IngestResponse(task_id, status, filename)
  - delete(filename)         → DeleteDocumentResponse(success, deleted_chunks)

Includes exponential-backoff retry logic (3 attempts: 1s, 2s, 4s delays)
for transient network errors against the local Docker container.

Security controls (OWASP):
- OPENRAG_API_KEY loaded exclusively from environment variables; never logged
- AuthenticationError raised with a helpful message that does not expose the key
- Only the filename (not the full path) is logged
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pipeline.config import get_openrag_api_key, get_openrag_url

logger = logging.getLogger(__name__)

# Retry configuration
_MAX_RETRIES = 3
_RETRY_DELAYS = (1.0, 2.0, 4.0)  # seconds between attempts


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class IngestResult:
    """Result of the OpenRAG ingestion stage.

    Attributes:
        document_id:  The OpenRAG task_id returned by the ingest call, used as
                      the stable reference for this document in state.json.
                      None if ingestion failed.
        filename:     Basename of the uploaded transcript file.
        status:       One of "success", "already_exists", or "failed".
        error:        Human-readable error message if status is "failed".
    """

    document_id: str | None
    filename: str
    status: str  # "success" | "already_exists" | "failed"
    error: str | None = field(default=None)
    filter_id: str | None = field(default=None)


# ---------------------------------------------------------------------------
# Knowledge filter helpers
# ---------------------------------------------------------------------------

_PODCAST_FILTER_NAME = "Podcast"


async def _ensure_podcast_filter(client: Any, filename: str) -> str | None:
    """
    Idempotently create or update the 'Podcast' knowledge filter,
    appending ``filename`` to its ``data_sources`` list.

    This is a query-time filter — it scopes searches to podcast transcripts
    by tracking their filenames. It is non-fatal: callers must catch exceptions.

    Returns the filter ID on success, or None if creation failed.
    """
    # pylint: disable=import-outside-toplevel
    from openrag_sdk.models import (  # type: ignore[import]
        CreateKnowledgeFilterOptions,
        KnowledgeFilterQueryData,
        UpdateKnowledgeFilterOptions,
    )

    existing: list = await client.knowledge_filters.search(
        _PODCAST_FILTER_NAME, limit=20
    )
    podcast_filter = next(
        (f for f in existing if f.name == _PODCAST_FILTER_NAME), None
    )

    if podcast_filter is None:
        # Create the filter with this file as the first data_source
        options = CreateKnowledgeFilterOptions(
            name=_PODCAST_FILTER_NAME,
            description="Auto-created filter for podcast transcripts",
            queryData=KnowledgeFilterQueryData(
                query="",
                filters={"data_sources": [filename], "document_types": ["*"]},
                limit=10,
                scoreThreshold=0.0,
                color="blue",
                icon="book",
            ),
        )
        result = await client.knowledge_filters.create(options)
        return result.id if result.success else None
    else:
        # Append filename to existing filter's data_sources if not already present
        current_sources: list[str] = []
        if podcast_filter.query_data and podcast_filter.query_data.filters:
            current_sources = list(
                podcast_filter.query_data.filters.get("data_sources", [])
            )

        if filename in current_sources:
            return podcast_filter.id

        updated_sources = current_sources + [filename]
        existing_query_data = podcast_filter.query_data
        update_options = UpdateKnowledgeFilterOptions(
            queryData=KnowledgeFilterQueryData(
                query=existing_query_data.query if existing_query_data else "",
                filters={
                    "data_sources": updated_sources,
                    "document_types": (
                        existing_query_data.filters.get("document_types", ["*"])
                        if existing_query_data and existing_query_data.filters
                        else ["*"]
                    ),
                },
                limit=existing_query_data.limit if existing_query_data else 10,
                scoreThreshold=(
                    existing_query_data.score_threshold
                    if existing_query_data
                    else 0.0
                ),
                color=existing_query_data.color if existing_query_data else "blue",
                icon=existing_query_data.icon if existing_query_data else "book",
            ),
        )
        await client.knowledge_filters.update(podcast_filter.id, update_options)
        return podcast_filter.id


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _delete_if_exists(client: object, filename: str) -> None:
    """Attempt to delete a document from OpenRAG by filename.

    Silently ignores errors where the document does not exist.

    Args:
        client:   An OpenRAGClient instance.
        filename: Basename of the document to delete.
    """
    # pylint: disable=import-outside-toplevel
    from openrag_sdk import OpenRAGClient  # type: ignore[import]
    from openrag_sdk.exceptions import NotFoundError  # type: ignore[import]

    assert isinstance(client, OpenRAGClient)
    try:
        result = await client.documents.delete(filename)
        logger.info(
            "Deleted existing document '%s' from OpenRAG "
            "(chunks removed: %d).",
            filename,
            result.deleted_chunks,
        )
    except NotFoundError:
        logger.debug(
            "Document '%s' not found in OpenRAG (nothing to delete).", filename
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.warning(
            "Could not delete document '%s' from OpenRAG: %s. "
            "Proceeding with re-ingest anyway.",
            filename,
            exc,
        )


async def _ingest_async(transcript_path: Path, force: bool) -> IngestResult:
    """Async implementation of the ingestion logic.

    Args:
        transcript_path: Path to the Markdown transcript file.
        force:           If True, attempt to delete the existing document from
                         OpenRAG before re-ingesting.

    Returns:
        An :class:`IngestResult` describing the outcome.
    """
    try:
        # pylint: disable=import-outside-toplevel
        from openrag_sdk import OpenRAGClient  # type: ignore[import]
        from openrag_sdk.exceptions import AuthenticationError  # type: ignore[import]
        from openrag_sdk.models import IngestTaskStatus  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(
            f"openrag-sdk is not installed: {exc}\n"
            "Install with: pip install 'openrag-sdk>=0.1.3'"
        ) from exc

    api_key = get_openrag_api_key()
    if not api_key:
        raise RuntimeError(
            "OPENRAG_API_KEY environment variable is not set. "
            "Add it to your .env file."
        )

    base_url = get_openrag_url() or "http://localhost:3000"
    filename = transcript_path.name

    # Instantiate client — key is used but never logged
    client = OpenRAGClient(api_key=api_key, base_url=base_url)

    # --force: attempt to delete the existing document before re-ingesting
    if force:
        await _delete_if_exists(client, filename)

    # Ingest with retry
    last_error: str | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            logger.info(
                "📤 Sending document to OpenRAG: '%s' → %s (attempt %d/%d)",
                filename,
                base_url,
                attempt,
                _MAX_RETRIES,
            )
            # wait=True polls until the task completes → returns IngestTaskStatus
            result = await client.documents.ingest(
                file_path=str(transcript_path),
                wait=True,
            )

            # result is IngestTaskStatus when wait=True
            task_id: str = result.task_id
            task_status: str = (
                result.status if isinstance(result, IngestTaskStatus) else "completed"
            )

            if isinstance(result, IngestTaskStatus) and result.status == "failed":
                failed_files = result.failed_files
                raise RuntimeError(
                    f"OpenRAG ingestion task failed (task_id={task_id}, "
                    f"failed_files={failed_files})."
                )

            logger.info(
                "✅ Document successfully sent to OpenRAG: "
                "filename='%s' task_id=%s status=%s url=%s",
                filename,
                task_id,
                task_status,
                base_url,
            )

            # Ensure the "Podcast" knowledge filter includes this file
            filter_id: str | None = None
            try:
                filter_id = await _ensure_podcast_filter(client, filename)
                if filter_id:
                    logger.info(
                        "Podcast filter updated: filter_id=%s filename=%s",
                        filter_id,
                        filename,
                    )
                else:
                    logger.warning(
                        "Podcast filter update returned no ID for filename=%s",
                        filename,
                    )
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.warning("Could not update podcast filter (non-fatal): %s", exc)

            return IngestResult(
                document_id=task_id,
                filename=filename,
                status="success",
                filter_id=filter_id,
            )

        except AuthenticationError as exc:
            # Never log the key value
            raise RuntimeError(
                "OpenRAG authentication failed. "
                "Check that OPENRAG_API_KEY is correct. "
                "(Key value not shown for security.)"
            ) from exc

        except Exception as exc:  # pylint: disable=broad-exception-caught
            err_str = str(exc)
            err_lower = err_str.lower()

            # Check for "already exists" — treat as non-fatal
            if "already" in err_lower or "duplicate" in err_lower or "409" in err_str:
                logger.info(
                    "Document '%s' already exists in OpenRAG (skipping).", filename
                )
                return IngestResult(
                    document_id=None,
                    filename=filename,
                    status="already_exists",
                )

            last_error = err_str
            logger.warning(
                "Ingestion attempt %d/%d failed for '%s': %s",
                attempt,
                _MAX_RETRIES,
                filename,
                err_str,
            )

            if attempt < _MAX_RETRIES:
                delay = _RETRY_DELAYS[attempt - 1]
                logger.info("Retrying in %.1f seconds…", delay)
                await asyncio.sleep(delay)

    # All retries exhausted
    logger.error(
        "All %d ingestion attempts failed for '%s'. Last error: %s",
        _MAX_RETRIES,
        filename,
        last_error,
    )
    return IngestResult(
        document_id=None,
        filename=filename,
        status="failed",
        error=last_error,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ingest_transcript(
    transcript_path: Path,
    force: bool = False,
) -> IngestResult:
    """Upload a Markdown transcript to OpenRAG and return the result.

    Bridges the async openrag-sdk into the synchronous pipeline using
    ``asyncio.run()``. Retries up to 3 times with exponential backoff on
    transient errors.

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
        "Starting OpenRAG ingestion for: %s (force=%s)",
        transcript_path.name,
        force,
    )

    return asyncio.run(_ingest_async(transcript_path, force))

