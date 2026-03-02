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

import logging
import re
from dataclasses import dataclass
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
    """Metadata and local audio path for a single downloaded podcast episode.

    Attributes:
        video_id:    YouTube video ID (idempotency key, filesystem-safe).
        title:       Episode title as reported by YouTube.
        upload_date: Upload date in YYYYMMDD format (from yt-dlp).
        description: Full episode description text.
        webpage_url: Canonical YouTube watch URL.
        channel:     Channel name.
        audio_path:  Absolute path to the downloaded .mp3 file on disk.
    """

    video_id: str
    title: str
    upload_date: str
    description: str
    webpage_url: str
    channel: str
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
    except Exception:
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
        audio_dir: Directory where downloaded audio files will be saved.

    Returns:
        Dictionary of yt-dlp options.
    """
    return {
        "format": "bestaudio/best",
        # Name files by video ID to keep paths predictable and filesystem-safe
        "outtmpl": str(audio_dir / "%(id)s.%(ext)s"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
        "no_warnings": True,
        "extract_flat": False,
        # Prevent yt-dlp from printing progress to stdout
        "noprogress": True,
        # Persist a download archive so yt-dlp skips videos it has already
        # fetched on subsequent runs. yt-dlp's built-in "file exists" check
        # looks for the pre-conversion container (e.g. .webm); after
        # FFmpegExtractAudio converts it to .mp3 the container is deleted, so
        # without an archive every re-run would re-download all videos.
        "download_archive": str(audio_dir / ".yt_dlp_archive"),
    }


def _info_to_episode(info: dict, audio_dir: Path) -> EpisodeAudio:
    """Convert a yt-dlp info dictionary to an :class:`EpisodeAudio` instance.

    Args:
        info:      yt-dlp info dict for a single video entry.
        audio_dir: Directory where the audio file was saved.

    Returns:
        Populated :class:`EpisodeAudio` dataclass.

    Raises:
        ValueError: If the video ID is missing or contains unsafe characters.
        RuntimeError: If the expected .mp3 file does not exist on disk.
    """
    raw_id = info.get("id", "")
    if not raw_id:
        raise ValueError("yt-dlp returned an entry with no video ID.")

    video_id = validate_video_id(raw_id)
    audio_path = audio_dir / f"{video_id}.mp3"

    if not audio_path.exists():
        raise RuntimeError(
            f"Expected audio file not found after download: {audio_path.name}. "
            "Ensure ffmpeg is installed and available on PATH."
        )

    return EpisodeAudio(
        video_id=video_id,
        title=info.get("title", ""),
        upload_date=info.get("upload_date", ""),
        description=info.get("description", ""),
        webpage_url=info.get("webpage_url", info.get("original_url", "")),
        channel=info.get("channel", info.get("uploader", "")),
        audio_path=audio_path,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def download_episode(url: str, audio_dir: Path | None = None) -> list[EpisodeAudio]:
    """Download audio from a YouTube URL and return episode metadata.

    Accepts single video URLs, playlist URLs, and channel URLs. For playlists
    and channels, all available videos are downloaded and returned.

    The function uses the yt-dlp Python API (not subprocess) to download audio
    as 192 kbps MP3 files. Files are named ``{video_id}.mp3`` to keep paths
    predictable and filesystem-safe.

    Args:
        url:       YouTube video, playlist, or channel URL.
        audio_dir: Directory to save downloaded audio files. Defaults to the
                   ``AUDIO_DIR`` environment variable, or ``./audio`` if unset.

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
    logger.info("Downloading audio from: %s → %s", validated_url, audio_dir)

    ydl_opts = _build_ydl_opts(audio_dir)

    # Collect info dicts for all downloaded entries
    downloaded_infos: list[dict] = []

    # Custom logger to suppress yt-dlp output while still capturing errors
    class _YdlLogger:
        def debug(self, msg: str) -> None:
            logger.debug("[yt-dlp] %s", msg)

        def info(self, msg: str) -> None:
            logger.debug("[yt-dlp] %s", msg)

        def warning(self, msg: str) -> None:
            logger.warning("[yt-dlp] %s", msg)

        def error(self, msg: str) -> None:
            logger.error("[yt-dlp] %s", msg)

    ydl_opts["logger"] = _YdlLogger()

    def _progress_hook(d: dict) -> None:
        if d.get("status") == "finished":
            filename = d.get("filename", "")
            # Log only the basename to avoid leaking full paths
            logger.info("Download finished: %s", Path(filename).name)

    ydl_opts["progress_hooks"] = [_progress_hook]

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # extract_info returns the full info dict; download=True triggers download
            info = ydl.extract_info(validated_url, download=True)
    except yt_dlp.utils.DownloadError as exc:
        raise RuntimeError(
            f"yt-dlp failed to download '{validated_url}': {exc}"
        ) from exc
    except Exception as exc:
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
        logger.info("Playlist/channel detected: %d entries found.", len(raw_entries))
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
                "All entries skipped by download archive — re-fetching metadata "
                "without downloading to build episode list from existing files."
            )
            meta_opts = {k: v for k, v in ydl_opts.items() if k != "download_archive"}
            meta_opts["logger"] = ydl_opts["logger"]
            try:
                with yt_dlp.YoutubeDL(meta_opts) as ydl_meta:
                    meta_info = ydl_meta.extract_info(validated_url, download=False)
                if meta_info:
                    for entry in (meta_info.get("entries") or []):
                        if entry is not None:
                            downloaded_infos.append(entry)
            except Exception as exc:
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
            f"All downloads failed. Errors:\n" + "\n".join(errors)
        )

    logger.info(
        "Acquisition complete: %d episode(s) ready, %d error(s).",
        len(episodes),
        len(errors),
    )
    return episodes

