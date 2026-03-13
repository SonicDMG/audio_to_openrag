"""
pipeline/openrag_client.py — OpenRAG SDK Wrapper

Provides a clean, synchronous wrapper around the async openrag-sdk with built-in
retry logic and helper methods. Simplifies OpenRAG integration by hiding async
complexity and providing idempotent operations.

Key Features:
- Synchronous API (uses asyncio.run() internally)
- Configurable retry logic with exponential backoff
- Idempotent knowledge filter management
- Graceful error handling with helpful messages

Security (OWASP):
- API key never logged or exposed in error messages
- Authentication errors provide helpful guidance without leaking credentials
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default retry configuration
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_RETRY_DELAYS = (1.0, 2.0, 4.0)  # seconds between attempts


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class IngestResult:
    """Result of an OpenRAG document ingestion operation.

    Attributes:
        document_id:  The OpenRAG task_id returned by the ingest call.
                      None if ingestion failed.
        filename:     Basename of the uploaded document file.
        status:       One of "success", "already_exists", or "failed".
        error:        Human-readable error message if status is "failed".
        filter_id:    ID of the knowledge filter if one was created/updated.
    """

    document_id: str | None
    filename: str
    status: str  # "success" | "already_exists" | "failed"
    error: str | None = field(default=None)
    filter_id: str | None = field(default=None)


# ---------------------------------------------------------------------------
# OpenRAG Client Wrapper
# ---------------------------------------------------------------------------


class OpenRAGClient:
    """Synchronous wrapper around the async openrag-sdk.

    Provides a clean, reusable interface for OpenRAG operations with built-in
    retry logic and helper methods. All async complexity is hidden from callers.

    Example:
        >>> client = OpenRAGClient(api_key="your-key")
        >>> result = client.ingest_document(Path("transcript.md"))
        >>> if result.status == "success":
        ...     print(f"Ingested: {result.document_id}")
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "http://localhost:3000",
        max_retries: int = _DEFAULT_MAX_RETRIES,
    ):
        """Initialize the OpenRAG client wrapper.

        Args:
            api_key:      OpenRAG API key for authentication.
            base_url:     Base URL of the OpenRAG instance.
                          Default: "http://localhost:3000"
            max_retries:  Maximum number of retry attempts for transient errors.
                          Default: 3
        """
        self.api_key = api_key
        self.base_url = base_url
        self.max_retries = max_retries
        self.retry_delays = _DEFAULT_RETRY_DELAYS[: max_retries]

    def ingest_document(
        self, file_path: Path, wait: bool = True, filter_name: str | None = None
    ) -> IngestResult:
        """Upload a document to OpenRAG with retry logic.

        Includes exponential backoff retry logic (1s, 2s, 4s delays by default)
        for transient network errors. Optionally creates/updates a knowledge
        filter to include the document.

        Args:
            file_path:    Path to the document file to upload.
            wait:         If True, wait for ingestion to complete before returning.
                          Default: True
            filter_name:  Optional name of knowledge filter to create/update.
                          If provided, the document will be added to this filter.

        Returns:
            IngestResult with status "success", "already_exists", or "failed".

        Raises:
            FileNotFoundError: If file_path does not exist.
            RuntimeError:      If openrag-sdk is not installed or API key is invalid.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        return asyncio.run(
            self._ingest_document_async(file_path, wait, filter_name)
        )

    def delete_document(self, filename: str) -> bool:
        """Delete a document from OpenRAG by filename.

        Gracefully handles the case where the document does not exist.

        Args:
            filename: Basename of the document to delete.

        Returns:
            True if document was deleted, False if not found.
        """
        return asyncio.run(self._delete_document_async(filename))

    def ensure_filter(
        self, filter_name: str, filename: str, description: str = ""
    ) -> str | None:
        """Idempotently ensure a knowledge filter exists and includes a file.

        This method is idempotent: it will search for an existing filter by name,
        create it if missing, or update it if it exists. The filename will be
        added to the filter's data_sources list if not already present.

        Args:
            filter_name:  Name of the knowledge filter.
            filename:     Document filename to add to the filter's data_sources.
            description:  Optional description for the filter (used on creation).

        Returns:
            The filter ID on success, or None if the operation failed.
        """
        return asyncio.run(
            self._ensure_filter_async(filter_name, filename, description)
        )

    def search_filters(self, query: str, limit: int = 20) -> list[Any]:
        """Search for knowledge filters by name.

        Args:
            query: Search query string (filter name).
            limit: Maximum number of results to return. Default: 20

        Returns:
            List of matching filter objects.
        """
        return asyncio.run(self._search_filters_async(query, limit))

    def create_filter(
        self, name: str, data_sources: list[str], description: str = ""
    ) -> str:
        """Create a new knowledge filter.

        Args:
            name:         Name of the filter.
            data_sources: List of document filenames to include in the filter.
            description:  Optional description of the filter.

        Returns:
            The created filter's ID.

        Raises:
            RuntimeError: If filter creation fails.
        """
        return asyncio.run(
            self._create_filter_async(name, data_sources, description)
        )

    def update_filter(self, filter_id: str, data_sources: list[str]) -> None:
        """Update an existing knowledge filter's data_sources.

        Args:
            filter_id:    ID of the filter to update.
            data_sources: New list of document filenames for the filter.
        """
        asyncio.run(self._update_filter_async(filter_id, data_sources))

    # -----------------------------------------------------------------------
    # Internal async implementations
    # -----------------------------------------------------------------------

    async def _ingest_document_async(
        self, file_path: Path, wait: bool, filter_name: str | None
    ) -> IngestResult:
        """Async implementation of document ingestion with retry logic."""
        try:
            # pylint: disable=import-outside-toplevel
            from openrag_sdk import OpenRAGClient as AsyncOpenRAGClient
            from openrag_sdk.exceptions import AuthenticationError
            from openrag_sdk.models import IngestTaskStatus
        except ImportError as exc:
            raise RuntimeError(
                f"openrag-sdk is not installed: {exc}\n"
                "Install with: pip install 'openrag-sdk>=0.1.3'"
            ) from exc

        filename = file_path.name
        client = AsyncOpenRAGClient(api_key=self.api_key, base_url=self.base_url)

        # Retry loop with exponential backoff
        last_error: str | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                print(
                    f"📤 Uploading to OpenRAG: {filename} "
                    f"(attempt {attempt}/{self.max_retries})"
                )

                # Ingest with wait=True to poll until completion
                result = await client.documents.ingest(
                    file_path=str(file_path),
                    wait=wait,
                )

                # Extract task_id and status
                task_id: str = result.task_id
                task_status: str = (
                    result.status
                    if isinstance(result, IngestTaskStatus)
                    else "completed"
                )

                # Validate ingestion status
                if isinstance(result, IngestTaskStatus):
                    if result.status == "failed":
                        raise RuntimeError(
                            f"Ingestion task failed (task_id={task_id}, "
                            f"failed_files={result.failed_files})"
                        )
                    elif result.status != "completed":
                        raise RuntimeError(
                            f"Ingestion task did not complete successfully "
                            f"(task_id={task_id}, status={result.status})"
                        )
                    if not result.processed_files:
                        raise RuntimeError(
                            f"Ingestion completed but no files were processed "
                            f"(task_id={task_id})"
                        )

                print(f"✅ Document ingested: {filename} (task_id={task_id})")

                # Optionally create/update knowledge filter
                filter_id: str | None = None
                if filter_name:
                    try:
                        filter_id = await self._ensure_filter_async(
                            filter_name, filename, "Auto-created knowledge filter"
                        )
                        if filter_id:
                            print(
                                f"📋 Knowledge filter '{filter_name}' updated "
                                f"(filter_id={filter_id})"
                            )
                    except Exception as exc:  # pylint: disable=broad-exception-caught
                        print(
                            f"⚠️  Could not update knowledge filter '{filter_name}': {exc}"
                        )

                return IngestResult(
                    document_id=task_id,
                    filename=filename,
                    status="success" if task_status == "completed" else "failed",
                    filter_id=filter_id,
                )

            except AuthenticationError as exc:
                # Never log the key value
                raise RuntimeError(
                    "OpenRAG authentication failed. "
                    "Check that your API key is correct. "
                    "(Key value not shown for security.)"
                ) from exc

            except Exception as exc:  # pylint: disable=broad-exception-caught
                err_str = str(exc)
                err_lower = err_str.lower()

                # Check for "already exists" — treat as non-fatal
                if (
                    "already" in err_lower
                    or "duplicate" in err_lower
                    or "409" in err_str
                ):
                    print(f"ℹ️  Document '{filename}' already exists in OpenRAG")
                    return IngestResult(
                        document_id=None,
                        filename=filename,
                        status="already_exists",
                    )

                last_error = err_str
                print(
                    f"⚠️  Attempt {attempt}/{self.max_retries} failed: {err_str}"
                )

                if attempt < self.max_retries:
                    delay = self.retry_delays[attempt - 1]
                    print(f"   Retrying in {delay:.1f} seconds...")
                    await asyncio.sleep(delay)

        # All retries exhausted
        print(f"❌ All {self.max_retries} attempts failed for '{filename}'")
        return IngestResult(
            document_id=None,
            filename=filename,
            status="failed",
            error=last_error,
        )

    async def _delete_document_async(self, filename: str) -> bool:
        """Async implementation of document deletion."""
        try:
            # pylint: disable=import-outside-toplevel
            from openrag_sdk import OpenRAGClient as AsyncOpenRAGClient
            from openrag_sdk.exceptions import NotFoundError
        except ImportError as exc:
            raise RuntimeError(
                f"openrag-sdk is not installed: {exc}\n"
                "Install with: pip install 'openrag-sdk>=0.1.3'"
            ) from exc

        client = AsyncOpenRAGClient(api_key=self.api_key, base_url=self.base_url)

        try:
            result = await client.documents.delete(filename)
            print(
                f"🗑️  Deleted document '{filename}' "
                f"(chunks removed: {result.deleted_chunks})"
            )
            return True
        except NotFoundError:
            print(f"ℹ️  Document '{filename}' not found (nothing to delete)")
            return False
        except Exception as exc:  # pylint: disable=broad-exception-caught
            print(f"⚠️  Could not delete document '{filename}': {exc}")
            return False

    async def _ensure_filter_async(
        self, filter_name: str, filename: str, description: str
    ) -> str | None:
        """Async implementation of idempotent filter management."""
        try:
            # pylint: disable=import-outside-toplevel
            from openrag_sdk import OpenRAGClient as AsyncOpenRAGClient
            from openrag_sdk.models import (
                CreateKnowledgeFilterOptions,
                KnowledgeFilterQueryData,
                UpdateKnowledgeFilterOptions,
            )
        except ImportError as exc:
            raise RuntimeError(
                f"openrag-sdk is not installed: {exc}\n"
                "Install with: pip install 'openrag-sdk>=0.1.3'"
            ) from exc

        client = AsyncOpenRAGClient(api_key=self.api_key, base_url=self.base_url)

        # Search for existing filter
        existing = await client.knowledge_filters.search(filter_name, limit=20)
        podcast_filter = next((f for f in existing if f.name == filter_name), None)

        if podcast_filter is None:
            # Create new filter
            options = CreateKnowledgeFilterOptions(
                name=filter_name,
                description=description,
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
            # Update existing filter
            current_sources: list[str] = []
            if podcast_filter.query_data and podcast_filter.query_data.filters:
                current_sources = list(
                    podcast_filter.query_data.filters.get("data_sources", [])
                )

            # Skip update if filename already present
            if filename in current_sources:
                return podcast_filter.id

            # Append filename to data_sources
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

    async def _search_filters_async(self, query: str, limit: int) -> list[Any]:
        """Async implementation of filter search."""
        try:
            # pylint: disable=import-outside-toplevel
            from openrag_sdk import OpenRAGClient as AsyncOpenRAGClient
        except ImportError as exc:
            raise RuntimeError(
                f"openrag-sdk is not installed: {exc}\n"
                "Install with: pip install 'openrag-sdk>=0.1.3'"
            ) from exc

        client = AsyncOpenRAGClient(api_key=self.api_key, base_url=self.base_url)
        return await client.knowledge_filters.search(query, limit=limit)

    async def _create_filter_async(
        self, name: str, data_sources: list[str], description: str
    ) -> str:
        """Async implementation of filter creation."""
        try:
            # pylint: disable=import-outside-toplevel
            from openrag_sdk import OpenRAGClient as AsyncOpenRAGClient
            from openrag_sdk.models import (
                CreateKnowledgeFilterOptions,
                KnowledgeFilterQueryData,
            )
        except ImportError as exc:
            raise RuntimeError(
                f"openrag-sdk is not installed: {exc}\n"
                "Install with: pip install 'openrag-sdk>=0.1.3'"
            ) from exc

        client = AsyncOpenRAGClient(api_key=self.api_key, base_url=self.base_url)

        options = CreateKnowledgeFilterOptions(
            name=name,
            description=description,
            queryData=KnowledgeFilterQueryData(
                query="",
                filters={"data_sources": data_sources, "document_types": ["*"]},
                limit=10,
                scoreThreshold=0.0,
                color="blue",
                icon="book",
            ),
        )
        result = await client.knowledge_filters.create(options)
        if not result.success or not result.id:
            raise RuntimeError(f"Failed to create knowledge filter '{name}'")
        return result.id

    async def _update_filter_async(
        self, filter_id: str, data_sources: list[str]
    ) -> None:
        """Async implementation of filter update."""
        try:
            # pylint: disable=import-outside-toplevel
            from openrag_sdk import OpenRAGClient as AsyncOpenRAGClient
            from openrag_sdk.models import (
                KnowledgeFilterQueryData,
                UpdateKnowledgeFilterOptions,
            )
        except ImportError as exc:
            raise RuntimeError(
                f"openrag-sdk is not installed: {exc}\n"
                "Install with: pip install 'openrag-sdk>=0.1.3'"
            ) from exc

        client = AsyncOpenRAGClient(api_key=self.api_key, base_url=self.base_url)

        update_options = UpdateKnowledgeFilterOptions(
            queryData=KnowledgeFilterQueryData(
                query="",
                filters={"data_sources": data_sources, "document_types": ["*"]},
                limit=10,
                scoreThreshold=0.0,
                color="blue",
                icon="book",
            ),
        )
        await client.knowledge_filters.update(filter_id, update_options)

# Made with Bob
