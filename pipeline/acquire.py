"""
pipeline/acquire.py — Audio Acquisition Stage

Downloads podcast audio from YouTube using yt-dlp and extracts structured
episode metadata. Supports single video URLs, playlist URLs, and channel URLs.

Security controls (OWASP):
- YouTube URL validated by structural URL parsing (scheme, host, path, query)
  before any network call; extra tracking parameters (e.g. &si=, &pp=) are
  ignored without weakening the check
- video_id sanitized to alphanumeric + hyphen + underscore before filesystem use
- No secrets accepted or logged
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import yt_dlp

from pipeline.config import get_audio_dir
from pipeline.constants import SAFE_ID_PATTERN
from pipeline.utils import validate_video_id

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Allowed YouTube hostnames
_YOUTUBE_HOSTS = {"www.youtube.com", "youtube.com", "youtu.be"}

# Path prefixes that identify channel/playlist pages (no required query param)
_CHANNEL_PATH_PATTERN = re.compile(
    r"^/(@[\w\-]+(/videos)?|channel/[\w\-]+|c/[\w\-]+)$"
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class EpisodeAudio:
    """Metadata and local media path for a single downloaded podcast episode.

    Attributes:
        video_id:    YouTube video ID (idempotency key, filesystem-safe).
        title:       Episode title as reported by YouTube.
        upload_date: Upload date in YYYYMMDD format (from yt-dlp).
        description: Full episode description text.
        webpage_url: Canonical YouTube watch URL.
        channel:     Channel name.
        duration:    Video duration in seconds (0 if unavailable).
        audio_path:  Absolute path to the downloaded media file on disk.
                     Note: Despite the name, this can be a video file (.mp4, .webm, etc.)
                     since Docling's ASR pipeline can extract audio from video directly.
    """

    video_id: str
    title: str
    upload_date: str
    description: str
    webpage_url: str
    channel: str
    duration: int
    audio_path: Path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_youtube_url(url: str) -> str:
    """Validate that *url* is a recognised YouTube URL.

    Validation is performed by parsing the URL structure with ``urllib.parse``
    rather than matching the full raw string, so extra tracking parameters
    (e.g. ``&si=``, ``&pp=``, ``&feature=``) appended by YouTube's share UI
    are ignored without weakening the security check.

    Accepted forms:
      - https://www.youtube.com/watch?v=VIDEO_ID   (+ any extra query params)
      - https://youtu.be/VIDEO_ID                  (+ any extra query params)
      - https://www.youtube.com/playlist?list=ID   (+ any extra query params)
      - https://www.youtube.com/@ChannelHandle[/videos]
      - https://www.youtube.com/channel/CHANNEL_ID
      - https://www.youtube.com/c/ChannelSlug

    Args:
        url: Raw URL string provided by the caller.

    Returns:
        The stripped, validated URL.

    Raises:
        ValueError: If the URL does not match any recognised YouTube pattern.
    """
    url = url.strip()

    try:
        parsed = urlparse(url)
    except (ValueError, TypeError):
        parsed = None  # urlparse rarely raises, but be defensive

    def _invalid() -> ValueError:
        return ValueError(
            f"Invalid YouTube URL: {url!r}\n"
            "Expected formats:\n"
            "  https://www.youtube.com/watch?v=VIDEO_ID\n"
            "  https://youtu.be/VIDEO_ID\n"
            "  https://www.youtube.com/playlist?list=PLAYLIST_ID\n"
            "  https://www.youtube.com/@ChannelHandle\n"
            "  https://www.youtube.com/channel/CHANNEL_ID"
        )

    if parsed is None or parsed.scheme not in ("http", "https"):
        raise _invalid()

    host = parsed.netloc.lower()
    if host not in _YOUTUBE_HOSTS:
        raise _invalid()

    path = parsed.path  # e.g. "/watch", "/playlist", "/@Handle", "/VIDEO_ID"
    qs = parse_qs(parsed.query)  # dict of {param: [value, ...]}

    # --- youtu.be/VIDEO_ID ---
    if host == "youtu.be":
        video_id = path.lstrip("/")
        if video_id and SAFE_ID_PATTERN.match(video_id):
            return url
        raise _invalid()

    # --- www.youtube.com / youtube.com paths ---
    if path == "/watch":
        # Must have a non-empty v= parameter
        v_values = qs.get("v", [])
        if v_values and SAFE_ID_PATTERN.match(v_values[0]):
            return url
        raise _invalid()

    if path == "/playlist":
        # Must have a non-empty list= parameter
        list_values = qs.get("list", [])
        if list_values and re.match(r"^[\w\-]+$", list_values[0]):
            return url
        raise _invalid()

    # Channel / handle paths require no query params — just a valid path shape
    if _CHANNEL_PATH_PATTERN.match(path):
        return url

    raise _invalid()


def _build_ydl_opts(audio_dir: Path) -> dict:
    """Build the yt-dlp options dictionary.

    Args:
        audio_dir: Directory where downloaded video files will be saved.

    Returns:
        Dictionary of yt-dlp options.
    
    Note:
        Docling's ASR pipeline can process video files directly, so we don't
        need FFmpeg to extract audio separately. This simplifies the pipeline
        and reduces processing time by eliminating the audio extraction step.
    """
    return {
        # Download full video - Docling extracts audio internally
        "format": "best",
        # Name files by video ID to keep paths predictable and filesystem-safe
        "outtmpl": str(audio_dir / "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "extract_flat": False,
        # Prevent yt-dlp from printing progress to stdout
        "noprogress": True,
        # Persist a download archive so yt-dlp skips videos it has already
        # fetched on subsequent runs.
        "download_archive": str(audio_dir / ".yt_dlp_archive"),
    }


def _load_metadata_cache(audio_dir: Path) -> dict[str, dict]:
    """Load cached video metadata from disk.

    Args:
        audio_dir: Directory containing the cache file.

    Returns:
        Dictionary mapping video_id to metadata dict. Empty dict if cache
        doesn't exist or is corrupted.
    """
    cache_file = audio_dir / ".video_metadata.json"
    if cache_file.exists():
        try:
            with open(cache_file, encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load metadata cache, will rebuild: %s", exc)
            return {}
    return {}


def _save_metadata_cache(audio_dir: Path, cache: dict[str, dict]) -> None:
    """Save video metadata cache to disk atomically.

    Writes to a temporary file first, then renames to avoid corruption if
    the process is interrupted.

    Args:
        audio_dir: Directory to save the cache file.
        cache: Dictionary mapping video_id to metadata dict.
    """
    cache_file = audio_dir / ".video_metadata.json"
    temp_file = audio_dir / ".video_metadata.json.tmp"

    try:
        # Write to temp file first
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2)

        # Atomic rename (on POSIX systems)
        temp_file.replace(cache_file)
    except OSError as exc:
        logger.warning("Failed to save metadata cache: %s", exc)
        # Clean up temp file if it exists
        if temp_file.exists():
            try:
                temp_file.unlink()
            except OSError:
                pass


def _extract_metadata_from_info(info: dict) -> dict:
    """Extract relevant metadata from yt-dlp info dict for caching.

    Args:
        info: yt-dlp info dictionary for a video.

    Returns:
        Dictionary containing cached metadata fields.
    """
    return {
        "title": info.get("title", "Unknown"),
        "duration": info.get("duration", 0),
        "upload_date": info.get("upload_date"),
        "uploader": info.get("uploader"),
        "channel": info.get("channel", info.get("uploader")),
        "description": info.get("description", ""),
        "url": info.get("webpage_url", info.get("original_url", "")),
        "cached_at": datetime.now(UTC).isoformat()
    }


def _info_to_episode(info: dict, audio_dir: Path) -> EpisodeAudio:
    """Convert a yt-dlp info dictionary to an :class:`EpisodeAudio` instance.

    Args:
        info:      yt-dlp info dict for a single video entry.
        audio_dir: Directory where the media file was saved.

    Returns:
        Populated :class:`EpisodeAudio` dataclass.

    Raises:
        ValueError: If the video ID is missing or contains unsafe characters.
        RuntimeError: If the expected media file does not exist on disk.
    """
    raw_id = info.get("id", "")
    if not raw_id:
        raise ValueError("yt-dlp returned an entry with no video ID.")

    video_id = validate_video_id(raw_id)
    
    # Find the actual downloaded file - could be .mp4, .webm, .mkv, etc.
    # yt-dlp names files as {video_id}.{ext} where ext depends on the format
    media_files = list(audio_dir.glob(f"{video_id}.*"))
    
    # Filter out metadata files and archives
    media_files = [
        f for f in media_files
        if f.suffix.lower() not in {'.json', '.txt', '.part', '.ytdl', '.tmp'}
    ]
    
    if not media_files:
        raise RuntimeError(
            f"Expected media file not found after download: {video_id}.* "
            f"No video or audio file found in {audio_dir}"
        )
    
    if len(media_files) > 1:
        # Multiple files found - prefer video formats, then audio formats
        video_exts = {'.mp4', '.webm', '.mkv', '.avi', '.mov'}
        audio_exts = {'.mp3', '.m4a', '.opus', '.ogg', '.wav'}
        
        video_files = [f for f in media_files if f.suffix.lower() in video_exts]
        audio_files = [f for f in media_files if f.suffix.lower() in audio_exts]
        
        if video_files:
            media_path = video_files[0]
        elif audio_files:
            media_path = audio_files[0]
        else:
            media_path = media_files[0]
        
        logger.debug(
            "Multiple files found for video_id %s, selected: %s",
            video_id,
            media_path.name
        )
    else:
        media_path = media_files[0]

    return EpisodeAudio(
        video_id=video_id,
        title=info.get("title", ""),
        upload_date=info.get("upload_date", ""),
        description=info.get("description", ""),
        webpage_url=info.get("webpage_url", info.get("original_url", "")),
        channel=info.get("channel", info.get("uploader", "")),
        duration=info.get("duration", 0) or 0,
        audio_path=media_path,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def download_episode(
    url: str,
    audio_dir: Path | None = None,
    progress_callback: Callable[[float, int, int, int], None] | None = None,
) -> list[EpisodeAudio]:
    """Download media from a YouTube URL and return episode metadata.

    Accepts single video URLs, playlist URLs, and channel URLs. For playlists
    and channels, all available videos are downloaded and returned.

    The function uses the yt-dlp Python API (not subprocess) to download video
    files in their best available format. Files are named ``{video_id}.{ext}``
    to keep paths predictable and filesystem-safe. Docling's ASR pipeline can
    extract audio from video files directly, so no separate audio extraction
    is needed.

    Args:
        url:              YouTube video, playlist, or channel URL.
        audio_dir:        Directory to save downloaded media files. Defaults to the
                          ``AUDIO_DIR`` environment variable, or ``./audio`` if unset.
        progress_callback: Optional callback function that receives download progress
                          as (percentage: float, current_video: int, total_videos: int,
                          duration_seconds: int). Percentage is between 0.0 and 1.0.
                          Duration is the length of the current video in seconds (0 if unavailable).

    Returns:
        A list of :class:`EpisodeAudio` instances — one per downloaded video.
        Always returns a list (single-video URLs return a one-element list).

    Raises:
        ValueError:   If *url* is not a valid YouTube URL.
        RuntimeError: If the download fails or the output file is missing.
    """
    validated_url = _validate_youtube_url(url)

    if audio_dir is None:
        audio_dir = get_audio_dir()

    audio_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata cache
    metadata_cache = _load_metadata_cache(audio_dir)
    cache_modified = False

    # Check if this is a single video URL and if the file already exists
    # Extract video ID for single video URLs to check for existing file
    parsed = urlparse(validated_url)
    video_id = None

    if parsed.netloc.lower() == "youtu.be":
        # youtu.be/VIDEO_ID format
        video_id = parsed.path.lstrip("/")
    elif parsed.path == "/watch":
        # youtube.com/watch?v=VIDEO_ID format
        qs = parse_qs(parsed.query)
        v_values = qs.get("v", [])
        if v_values:
            video_id = v_values[0]

    # If we have a video ID and the file exists, skip download
    if video_id:
        try:
            video_id = validate_video_id(video_id)
            
            # Check if any media file exists for this video_id
            existing_files = list(audio_dir.glob(f"{video_id}.*"))
            existing_files = [
                f for f in existing_files
                if f.suffix.lower() not in {'.json', '.txt', '.part', '.ytdl', '.tmp'}
            ]
            
            if existing_files:
                media_path = existing_files[0]
                logger.info(
                    "File already exists for video ID %s, skipping download: %s",
                    video_id,
                    media_path.name
                )

                # Try to use cached metadata first
                if video_id in metadata_cache:
                    logger.info("Using cached metadata for video ID %s", video_id)
                    cached = metadata_cache[video_id]
                    episode = EpisodeAudio(
                        video_id=video_id,
                        title=cached.get("title", "Unknown"),
                        upload_date=cached.get("upload_date", ""),
                        description=cached.get("description", ""),
                        webpage_url=cached.get("url", validated_url),
                        channel=cached.get("channel", cached.get("uploader", "")),
                        duration=cached.get("duration", 0) or 0,
                        audio_path=media_path,
                    )
                    logger.info(
                        "Episode ready (cached metadata): video_id=%s title=%r duration=%ds",
                        episode.video_id,
                        episode.title[:60],
                        episode.duration,
                    )
                    return [episode]

                # Fetch metadata without downloading to build EpisodeAudio
                logger.info(
                    "Metadata not in cache, fetching from YouTube for video ID %s",
                    video_id
                )
                ydl_opts_meta = {
                    "quiet": True,
                    "no_warnings": True,
                    "extract_flat": False,
                    "skip_download": True,
                }

                try:
                    with yt_dlp.YoutubeDL(ydl_opts_meta) as ydl:
                        info = ydl.extract_info(validated_url, download=False)

                    if info:
                        # Cache the metadata for future use
                        metadata_cache[video_id] = _extract_metadata_from_info(info)
                        _save_metadata_cache(audio_dir, metadata_cache)

                        episode = _info_to_episode(info, audio_dir)
                        logger.info(
                            "Episode ready (existing file): video_id=%s title=%r",
                            episode.video_id,
                            episode.title[:60],
                        )
                        return [episode]
                except (yt_dlp.utils.DownloadError, yt_dlp.utils.ExtractorError) as exc:
                    logger.warning(
                        "Failed to fetch metadata for existing file, will re-download: %s",
                        exc
                    )
        except (ValueError, RuntimeError):
            # Invalid video ID or other issue, proceed with normal download
            pass

    logger.info("Downloading audio from: %s → %s", validated_url, audio_dir)

    ydl_opts = _build_ydl_opts(audio_dir)

    # Collect info dicts for all downloaded entries
    downloaded_infos: list[dict] = []

    # Custom logger to suppress yt-dlp output while still capturing errors
    class _YdlLogger:
        """Custom logger for yt-dlp to redirect output to Python logging."""

        def debug(self, msg: str) -> None:
            """Log debug messages from yt-dlp."""
            logger.debug("[yt-dlp] %s", msg)

        def info(self, msg: str) -> None:
            """Log info messages from yt-dlp as debug."""
            logger.debug("[yt-dlp] %s", msg)

        def warning(self, msg: str) -> None:
            """Log warning messages from yt-dlp."""
            logger.warning("[yt-dlp] %s", msg)

        def error(self, msg: str) -> None:
            """Log error messages from yt-dlp."""
            logger.error("[yt-dlp] %s", msg)

    ydl_opts["logger"] = _YdlLogger()

    # Track video count and durations for progress reporting
    current_video_num = 0
    total_videos = 1  # Default to 1 for single videos
    video_durations: dict[int, int] = {}  # Map video number to duration in seconds

    def _progress_hook(d: dict) -> None:
        nonlocal current_video_num, total_videos
        status = d.get("status")
        if status == "downloading":
            downloaded = d.get("downloaded_bytes", 0)
            total = d.get("total_bytes") or d.get("total_bytes_estimate", 0)
            if total > 0 and progress_callback:
                # Get duration for current video, default to 0 if not available
                duration = video_durations.get(current_video_num, 0)
                progress_callback(downloaded / total, current_video_num, total_videos, duration)
        elif status == "finished":
            if progress_callback:
                duration = video_durations.get(current_video_num, 0)
                progress_callback(1.0, current_video_num, total_videos, duration)
            filename = d.get("filename", "")
            # Log only the basename to avoid leaking full paths
            logger.info("Download finished: %s", Path(filename).name)
            # Increment video counter when a video finishes downloading
            current_video_num += 1

    ydl_opts["progress_hooks"] = [_progress_hook]

    # First, extract info to determine total video count and durations before downloading
    # Use options WITHOUT download_archive so we see all videos, not just undownloaded ones
    try:
        # Create metadata-only options without download archive
        meta_opts = {k: v for k, v in ydl_opts.items() if k != "download_archive"}
        meta_opts["logger"] = ydl_opts["logger"]

        with yt_dlp.YoutubeDL(meta_opts) as ydl_meta:
            # Extract info first to get playlist size and durations (all videos, not just new ones)
            info_extract = ydl_meta.extract_info(validated_url, download=False)
            if info_extract:
                entries = info_extract.get("entries")
                if entries is not None:
                    # Playlist or channel - extract durations for each video
                    # First, convert entries to a list to get the total count
                    entries_list = [e for e in entries if e is not None]
                    total_entries = len(entries_list)

                    video_num = 1
                    for entry in entries_list:
                        # Extract duration (in seconds), default to 0 if not available
                        duration = entry.get("duration", 0) or 0
                        video_durations[video_num] = duration

                        # Call progress callback during metadata extraction
                        if progress_callback:
                            progress_callback(
                                video_num / total_entries, video_num, total_entries, duration
                            )

                        # Cache metadata for all videos in the playlist
                        entry_id = entry.get("id")
                        if entry_id and entry_id not in metadata_cache:
                            try:
                                validated_entry_id = validate_video_id(entry_id)
                                metadata_cache[validated_entry_id] = (
                                    _extract_metadata_from_info(entry)
                                )
                                cache_modified = True
                            except ValueError:
                                pass  # Skip invalid video IDs

                        video_num += 1
                    total_videos = len(video_durations)
                    # Log total videos immediately after extraction
                    logger.info("Playlist/channel detected: %d total videos.", total_videos)
                else:
                    # Single video - extract its duration
                    duration = info_extract.get("duration", 0) or 0
                    video_durations[1] = duration

                    # Call progress callback for single video metadata extraction
                    if progress_callback:
                        progress_callback(0.0, 1, 1, duration)

                    # Cache metadata for single video
                    entry_id = info_extract.get("id")
                    if entry_id and entry_id not in metadata_cache:
                        try:
                            validated_entry_id = validate_video_id(entry_id)
                            metadata_cache[validated_entry_id] = (
                                _extract_metadata_from_info(info_extract)
                            )
                            cache_modified = True
                        except ValueError:
                            pass

        # Now perform the actual download with the full options (including download_archive)
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            current_video_num = 1  # Start counting from 1
            info = ydl.extract_info(validated_url, download=True)
    except yt_dlp.utils.DownloadError as exc:
        raise RuntimeError(
            f"yt-dlp failed to download '{validated_url}': {exc}"
        ) from exc
    except (yt_dlp.utils.ExtractorError, OSError, ValueError) as exc:
        raise RuntimeError(
            f"Unexpected error during download of '{validated_url}': {exc}"
        ) from exc

    if info is None:
        raise RuntimeError(f"yt-dlp returned no info for URL: {validated_url!r}")

    # Handle playlists / channels (entries key present) vs single videos
    entries = info.get("entries")
    if entries is not None:
        # Playlist or channel — iterate all entries
        raw_entries = list(entries)  # may be a generator
        num_downloaded = len(raw_entries)
        num_skipped = total_videos - num_downloaded
        if num_skipped > 0:
            logger.info(
                "Downloaded %d new video(s), skipped %d already-downloaded video(s) (total: %d).",
                num_downloaded,
                num_skipped,
                total_videos
            )
        else:
            logger.info("Downloaded %d video(s).", num_downloaded)
        for entry in raw_entries:
            if entry is None:
                continue
            downloaded_infos.append(entry)

        # If yt-dlp returned 0 entries it means every video was skipped via the
        # download archive (all .mp3 files already exist on disk). Re-fetch the
        # playlist metadata without downloading so we can build EpisodeAudio
        # objects for the files that are already present.
        if not downloaded_infos:
            logger.info(
                "All entries skipped by download archive — checking metadata cache "
                "to build episode list from existing files."
            )

            # Try to build episode list from cache first
            cache_hits = 0
            cache_misses = []

            # Get list of all media files in audio directory (excluding metadata files)
            all_files = list(audio_dir.glob("*.*"))
            media_files = [
                f for f in all_files
                if f.suffix.lower() not in {'.json', '.txt', '.part', '.ytdl', '.tmp'}
            ]
            total_cached_videos = len(media_files)

            for idx, media_file in enumerate(media_files, 1):
                video_id = media_file.stem  # filename without extension

                if video_id in metadata_cache:
                    # Use cached metadata
                    cached = metadata_cache[video_id]
                    duration = cached.get("duration", 0) or 0

                    # Call progress callback when loading from cache
                    if progress_callback:
                        progress_callback(
                            idx / total_cached_videos, idx, total_cached_videos, duration
                        )

                    # Create a minimal info dict that _info_to_episode can process
                    info_dict = {
                        "id": video_id,
                        "title": cached.get("title", "Unknown"),
                        "upload_date": cached.get("upload_date", ""),
                        "description": cached.get("description", ""),
                        "webpage_url": cached.get("url", ""),
                        "channel": cached.get("channel", cached.get("uploader", "")),
                        "uploader": cached.get("uploader", ""),
                    }
                    downloaded_infos.append(info_dict)
                    cache_hits += 1
                else:
                    cache_misses.append(video_id)

            if cache_hits > 0:
                logger.info("Using cached metadata for %d video(s).", cache_hits)

            # If we have cache misses, fetch only those from YouTube
            if cache_misses:
                logger.info(
                    "Metadata not in cache for %d video(s), fetching from YouTube.",
                    len(cache_misses)
                )
                meta_opts = {k: v for k, v in ydl_opts.items() if k != "download_archive"}
                meta_opts["logger"] = ydl_opts["logger"]
                try:
                    with yt_dlp.YoutubeDL(meta_opts) as ydl_meta:
                        meta_info = ydl_meta.extract_info(validated_url, download=False)
                    if meta_info:
                        for entry in (meta_info.get("entries") or []):
                            if entry is not None:
                                entry_id = entry.get("id")
                                if entry_id in cache_misses:
                                    downloaded_infos.append(entry)
                                    # Cache the newly fetched metadata
                                    try:
                                        validated_entry_id = validate_video_id(entry_id)
                                        metadata_cache[validated_entry_id] = (
                                            _extract_metadata_from_info(entry)
                                        )
                                        cache_modified = True
                                    except ValueError:
                                        pass
                except (yt_dlp.utils.DownloadError, yt_dlp.utils.ExtractorError) as exc:
                    logger.warning("Metadata-only fetch failed: %s", exc)
    else:
        # Single video
        downloaded_infos.append(info)

    episodes: list[EpisodeAudio] = []
    errors: list[str] = []

    for entry_info in downloaded_infos:
        try:
            episode = _info_to_episode(entry_info, audio_dir)
            episodes.append(episode)
            logger.info(
                "Episode ready: video_id=%s title=%r",
                episode.video_id,
                episode.title[:60],
            )
        except (ValueError, RuntimeError) as exc:
            errors.append(str(exc))
            logger.error("Failed to process entry: %s", exc)

    if not episodes and errors:
        raise RuntimeError(
            "All downloads failed. Errors:\n" + "\n".join(errors)
        )

    # Save metadata cache if it was modified
    if cache_modified:
        _save_metadata_cache(audio_dir, metadata_cache)
        logger.debug("Metadata cache updated with new entries.")

    logger.info(
        "Acquisition complete: %d episode(s) ready, %d error(s).",
        len(episodes),
        len(errors),
    )
    return episodes

# Made with Bob
