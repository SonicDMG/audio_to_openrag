"""
main.py — CLI Entrypoint for The Flow Pipeline

Two commands:
  ingest  <url> [--force] [--dry-run] [--num-speakers N] [--skip-diarization]
  status

Loads .env via python-dotenv, validates required environment variables,
then orchestrates all pipeline stages with rich progress output.

Security controls (OWASP):
- Secrets loaded from .env only; never accepted as CLI arguments
- YouTube URL validated before any network call (in acquire.py)
- Required env vars checked at startup before any processing begins
- HF_TOKEN and OPENRAG_API_KEY values are never logged
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pipeline.acquire import EpisodeAudio

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

# Load .env before anything else so env vars are available to all imports
load_dotenv()

console = Console()


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def _configure_logging() -> None:
    """Configure Python logging with Rich handler and level from env."""
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )
    # Suppress noisy third-party loggers at INFO level
    for noisy in ("httpx", "httpcore", "urllib3", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------


def _check_required_env_vars(dry_run: bool, skip_diarization: bool) -> None:
    """Validate that required environment variables are set.

    Args:
        dry_run:           If True, OPENRAG_API_KEY is not required (no upload).
        skip_diarization:  If True, HF_TOKEN is not required (pyannote not used).

    Raises:
        click.ClickException: If any required variable is missing.
    """
    required: list[str] = []
    if not skip_diarization:
        required.append("HF_TOKEN")
    if not dry_run:
        required.append("OPENRAG_API_KEY")

    missing = [var for var in required if not os.environ.get(var)]
    if missing:
        raise click.ClickException(
            f"Required environment variable(s) not set: {', '.join(missing)}\n"
            "Copy .env.example to .env and fill in your values."
        )


def _check_ffmpeg() -> None:
    """Verify ffmpeg is available on PATH.

    Raises:
        click.ClickException: If ffmpeg is not found.
    """
    if not shutil.which("ffmpeg"):
        raise click.ClickException(
            "ffmpeg is not installed or not on PATH.\n"
            "Install with: brew install ffmpeg"
        )


# ---------------------------------------------------------------------------
# Plain-transcript helper (used when diarization is skipped)
# ---------------------------------------------------------------------------


def _build_plain_segments(
    transcript_result: object,
    diarized_segment_cls: type,
    log: logging.Logger,
) -> list:
    """Build a list of DiarizedSegment objects from raw Whisper segments.

    Used when diarization is skipped (``--skip-diarization``) or when
    pyannote returned no output. Each Whisper segment is wrapped in a
    DiarizedSegment with ``speaker_label=""`` so that ``document.py``
    renders the text without a speaker prefix.

    Falls back to extracting plain text from the Docling document if
    Whisper segments are not available.

    Args:
        transcript_result:     TranscriptResult from the transcribe stage.
        diarized_segment_cls:  The DiarizedSegment dataclass.
        log:                   Logger instance.

    Returns:
        List of DiarizedSegment objects (may be empty if no text is found).
    """
    segments = getattr(transcript_result, "segments", None)

    if segments:
        return [
            diarized_segment_cls(
                speaker_label="",
                start=s["start"],
                end=s["end"],
                text=s.get("text", "").strip(),
            )
            for s in segments
            if s.get("text", "").strip()
        ]

    # No Whisper segments — try extracting text from the Docling document
    log.warning(
        "No Whisper segments available; falling back to Docling document text."
    )
    document = getattr(transcript_result, "document", None)
    fallback_text = ""
    if document is not None:
        try:
            fallback_text = document.export_to_markdown()
        except Exception:
            try:
                fallback_text = " ".join(
                    getattr(item, "text", "")
                    for item in getattr(document, "texts", [])
                    if getattr(item, "text", "")
                )
            except Exception:
                fallback_text = ""

    if fallback_text.strip():
        return [
            diarized_segment_cls(
                speaker_label="",
                start=0.0,
                end=0.0,
                text=fallback_text.strip(),
            )
        ]

    log.warning(
        "Could not extract any text. Transcript file will note no content is available."
    )
    return []


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
def cli() -> None:
    """The Flow Pipeline — ingest podcast episodes from YouTube into OpenRAG."""
    _configure_logging()


# ---------------------------------------------------------------------------
# ingest command
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("url")
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Re-ingest even if the episode is already in state.json.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Download, transcribe, and diarize but do NOT upload to OpenRAG.",
)
@click.option(
    "--num-speakers",
    type=int,
    default=2,
    show_default=True,
    help="Expected number of speakers for diarization.",
)
@click.option(
    "--skip-diarization",
    is_flag=True,
    default=False,
    help=(
        "Skip speaker diarization and produce transcripts without speaker labels. "
        "HF_TOKEN is not required when this flag is set."
    ),
)
def ingest(url: str, force: bool, dry_run: bool, num_speakers: int, skip_diarization: bool) -> None:
    """Download, transcribe, diarize, and ingest a YouTube episode or playlist.

    URL can be a single video, a playlist, or a channel URL.
    Pass --skip-diarization to skip speaker labelling (no HF_TOKEN required).
    """
    log = logging.getLogger(__name__)

    # Preflight checks
    _check_required_env_vars(dry_run=dry_run, skip_diarization=skip_diarization)
    _check_ffmpeg()

    # Resolve directories from env vars
    audio_dir = Path(os.environ.get("AUDIO_DIR", "./audio"))
    transcripts_dir = Path(os.environ.get("TRANSCRIPTS_DIR", "./transcripts"))
    state_file = Path(os.environ.get("STATE_FILE", "./state.json"))

    # Import pipeline modules (deferred to keep startup fast)
    from pipeline import acquire, diarize, document, ingest as ingest_mod, state, transcribe

    # -----------------------------------------------------------------------
    # Acquire: download all episodes from the URL
    # -----------------------------------------------------------------------
    console.rule("[bold blue]Stage 1 — Acquire")
    try:
        episodes = acquire.download_episode(url, audio_dir=audio_dir)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    except RuntimeError as exc:
        raise click.ClickException(f"Download failed: {exc}") from exc

    total = len(episodes)
    console.print(f"[green]Found {total} episode(s) to process.[/green]")

    # -----------------------------------------------------------------------
    # Per-episode processing
    # -----------------------------------------------------------------------
    succeeded = 0
    skipped = 0
    failed = 0

    for idx, episode in enumerate(episodes, start=1):
        console.rule(
            f"[bold]Episode {idx}/{total}: {episode.title[:70]}"
        )

        # State check
        if not force and state.is_ingested(episode.video_id, state_file):
            console.print(
                f"[yellow]Skipping[/yellow] {episode.video_id} — already ingested. "
                "Use --force to re-ingest."
            )
            skipped += 1
            continue

        try:
            _process_episode(
                episode=episode,
                transcripts_dir=transcripts_dir,
                state_file=state_file,
                force=force,
                dry_run=dry_run,
                num_speakers=num_speakers,
                skip_diarization=skip_diarization,
                transcribe_mod=transcribe,
                diarize_mod=diarize,
                document_mod=document,
                ingest_mod=ingest_mod,
                state_mod=state,
            )
            succeeded += 1

        except Exception as exc:  # noqa: BLE001
            log.error(
                "Episode %s failed: %s",
                episode.video_id,
                exc,
                exc_info=True,
            )
            console.print(f"[red]FAILED[/red] {episode.video_id}: {exc}")
            failed += 1

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    console.rule("[bold]Pipeline Complete")
    console.print(
        f"[bold green]{succeeded}[/bold green] ingested  "
        f"[bold yellow]{skipped}[/bold yellow] skipped  "
        f"[bold red]{failed}[/bold red] failed"
        f"  (of {total} total)"
    )

    if failed > 0:
        sys.exit(1)


def _process_episode(
    *,
    episode: "EpisodeAudio",
    transcripts_dir: Path,
    state_file: Path,
    force: bool,
    dry_run: bool,
    num_speakers: int,
    skip_diarization: bool,
    transcribe_mod: object,
    diarize_mod: object,
    document_mod: object,
    ingest_mod: object,
    state_mod: object,
) -> None:
    """Run all pipeline stages for a single episode.

    Separated from the main ingest() command to keep error handling clean.
    Each stage is wrapped in a rich Progress spinner for user feedback.

    Args:
        episode:           EpisodeAudio from the acquisition stage.
        transcripts_dir:   Directory for Markdown output files.
        state_file:        Path to state.json.
        force:             Whether to re-ingest if already present.
        dry_run:           If True, skip the OpenRAG upload.
        num_speakers:      Number of speakers for diarization.
        skip_diarization:  If True, skip pyannote diarization entirely and
                           produce a plain transcript without speaker labels.
        *_mod:             Imported pipeline module references.
    """
    log = logging.getLogger(__name__)
    from pipeline.diarize import DiarizedSegment  # type: ignore[import]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:

        # Stage 2: Transcribe
        task = progress.add_task("Transcribing audio…", total=None)
        transcript_result = transcribe_mod.transcribe_audio(episode.audio_path)  # type: ignore[union-attr]
        progress.update(task, description="[green]Transcription complete")
        progress.stop_task(task)

        # Stage 3: Diarize (or skip)
        if skip_diarization:
            progress.add_task("[yellow]Diarization skipped (--skip-diarization)", total=None)
            log.info("Diarization skipped — building plain transcript without speaker labels.")
            diarized_segments = _build_plain_segments(transcript_result, DiarizedSegment, log)
        else:
            task = progress.add_task("Diarizing speakers…", total=None)
            diarized_segments = diarize_mod.diarize_audio(  # type: ignore[union-attr]
                audio_path=episode.audio_path,
                segments=transcript_result.segments,
                num_speakers=num_speakers,
            )
            progress.update(task, description="[green]Diarization complete")
            progress.stop_task(task)

            # If diarization produced nothing because segments were None (no timestamps),
            # fall back to extracting plain text from the Docling document so the
            # transcript file still contains the transcribed content.
            if not diarized_segments and transcript_result.segments is None:
                diarized_segments = _build_plain_segments(transcript_result, DiarizedSegment, log)

        # Stage 4: Build document
        task = progress.add_task("Building transcript document…", total=None)
        md_path, _docling_doc = document_mod.build_transcript_document(  # type: ignore[union-attr]
            episode=episode,
            segments=diarized_segments,
            transcripts_dir=transcripts_dir,
        )
        progress.update(task, description="[green]Document built")
        progress.stop_task(task)

    console.print(f"  [dim]Transcript:[/dim] {md_path.name}")

    # Stage 5: Ingest (unless --dry-run)
    openrag_document_id: str | None = None

    if dry_run:
        console.print("  [yellow]--dry-run:[/yellow] skipping OpenRAG upload.")
    else:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Uploading to OpenRAG…", total=None)
            ingest_result = ingest_mod.ingest_transcript(  # type: ignore[union-attr]
                transcript_path=md_path,
                force=force,
            )
            progress.stop_task(task)

        if ingest_result.status == "success":
            openrag_document_id = ingest_result.document_id
            console.print(
                f"  [green]Ingested:[/green] task_id={openrag_document_id}"
            )
        elif ingest_result.status == "already_exists":
            console.print("  [yellow]Already exists in OpenRAG[/yellow] (skipped upload).")
        else:
            raise RuntimeError(
                f"OpenRAG ingestion failed: {ingest_result.error}"
            )

    # Stage 6: Update state
    state_mod.mark_ingested(  # type: ignore[union-attr]
        episode=episode,
        transcript_path=md_path,
        openrag_document_id=openrag_document_id,
        state_file=state_file,
    )
    console.print(f"  [green]State updated[/green] for video_id={episode.video_id}")


# ---------------------------------------------------------------------------
# status command
# ---------------------------------------------------------------------------


@cli.command()
def status() -> None:
    """Show all ingested episodes from state.json."""
    from pipeline import state

    state_file = Path(os.environ.get("STATE_FILE", "./state.json"))
    all_state = state.get_all(state_file)
    episodes = all_state.get("episodes", {})

    if not episodes:
        console.print("[yellow]No episodes ingested yet.[/yellow]")
        console.print(f"State file: {state_file}")
        return

    table = Table(
        title=f"Ingested Episodes ({len(episodes)} total)",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Video ID", style="cyan", no_wrap=True)
    table.add_column("Title", max_width=50)
    table.add_column("Channel", max_width=20)
    table.add_column("Date", style="dim", no_wrap=True)
    table.add_column("OpenRAG Task ID", style="dim", max_width=30)
    table.add_column("Ingested At", style="dim", no_wrap=True)

    for entry in episodes.values():
        ingested_at = entry.get("ingested_at", "")
        # Trim to date+time without microseconds for display
        if "T" in ingested_at:
            ingested_at = ingested_at[:19].replace("T", " ")

        table.add_row(
            entry.get("video_id", ""),
            entry.get("title", ""),
            entry.get("channel", ""),
            entry.get("upload_date", ""),
            entry.get("openrag_document_id") or "[dim]—[/dim]",
            ingested_at,
        )

    console.print(table)
    console.print(f"\n[dim]State file: {state_file}[/dim]")
    last_updated = all_state.get("last_updated", "")
    if last_updated:
        console.print(f"[dim]Last updated: {last_updated[:19].replace('T', ' ')}[/dim]")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    cli()

# Made with Bob
