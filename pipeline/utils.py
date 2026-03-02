"""
pipeline/utils.py — Shared pipeline utilities

Security controls (OWASP):
- No shell=True invocations; shutil.which() is used for binary resolution
- No user-controlled input in path construction; probe list is a hardcoded
  constant defined in this module
- PATH is modified only by prepending a known, validated directory
"""

from __future__ import annotations

import logging
import os
import shutil

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known ffmpeg installation locations, probed in priority order.
# Only absolute paths to directories that ship ffmpeg on macOS/Linux are
# listed here.  No user-supplied or environment-derived paths are included
# (OWASP: avoid injection via environment manipulation).
# ---------------------------------------------------------------------------
_FFMPEG_PROBE_DIRS: tuple[str, ...] = (
    "/opt/homebrew/bin",   # Apple Silicon Homebrew (primary target)
    "/usr/local/bin",      # Intel Homebrew / manual installs
    "/usr/bin",            # System-provided ffmpeg (rare on macOS)
)


def ensure_ffmpeg_on_path() -> str | None:
    """Ensure ``ffmpeg`` is resolvable by all subprocess calls in this process.

    Many libraries used by this pipeline (``yt-dlp``, ``faster-whisper``,
    ``mlx-whisper``, ``docling``) invoke ``ffmpeg`` by name via subprocess.
    In non-interactive shells (cron, launchd, VS Code integrated terminal with
    a clean environment) ``/opt/homebrew/bin`` is often absent from ``PATH``,
    causing those calls to raise ``FileNotFoundError`` or fail silently.

    This function:

    1. Checks whether ``ffmpeg`` is already resolvable via :func:`shutil.which`.
    2. If not, probes a hardcoded list of known macOS/Linux installation
       directories in priority order.
    3. If found, **prepends** the parent directory to ``os.environ["PATH"]``
       so that all subsequent subprocess calls in this process (including those
       made by third-party libraries) can locate ``ffmpeg``.
    4. If not found anywhere, logs a ``WARNING`` and returns ``None`` — the
       pipeline degrades gracefully rather than raising.

    Security notes (OWASP):
    - Uses :func:`shutil.which` instead of ``subprocess.run(["which", ...])``
      to avoid spawning a shell.
    - The probe list is a hardcoded constant; no user input is incorporated
      into path construction.
    - ``shell=False`` is implicit — no subprocess is spawned at all.

    Returns:
        The resolved absolute path to the ``ffmpeg`` binary as a ``str``,
        or ``None`` if ``ffmpeg`` could not be found.
    """
    # Fast path: ffmpeg is already on PATH (interactive shell, Docker image
    # with ffmpeg in /usr/bin, etc.)
    existing = shutil.which("ffmpeg")
    if existing:
        logger.debug("ffmpeg already on PATH: %s", existing)
        return existing

    # Slow path: probe known locations
    for probe_dir in _FFMPEG_PROBE_DIRS:
        candidate = os.path.join(probe_dir, "ffmpeg")
        # os.access with X_OK checks execute permission without spawning a
        # process — safe and cross-platform.
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            logger.info(
                "ffmpeg not on PATH; found at %s — prepending %s to PATH.",
                candidate,
                probe_dir,
            )
            # Prepend so our directory wins over any later entries that might
            # contain a different (possibly older) ffmpeg.
            os.environ["PATH"] = probe_dir + os.pathsep + os.environ.get("PATH", "")
            return candidate

    logger.warning(
        "ffmpeg not found on PATH or in known locations. "
        "Audio download and transcription may fail. "
        "Install with: brew install ffmpeg"
    )
    return None

# Made with Bob
