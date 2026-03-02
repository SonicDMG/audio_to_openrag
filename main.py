"""
main.py — CLI Entrypoint for The Flow Pipeline

Two commands:
  ingest  <url> [--force] [--dry-run]
  status

Loads .env via python-dotenv, validates required environment variables,
then orchestrates all pipeline stages with rich progress output.

Transcription uses Docling's AsrPipeline with whisper-turbo.
Diarization has been removed - transcripts are produced without speaker labels.

Security controls (OWASP):
- Secrets loaded from .env only; never accepted as CLI arguments
- YouTube URL validated before any network call (in acquire.py)
- Required env vars checked at startup before any processing begins
- OPENRAG_API_KEY values are never logged
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from pipeline import acquire, document, state, transcribe
from pipeline import ingest as ingest_mod
from pipeline.document import DiarizedSegment
from pipeline.utils import ensure_ffmpeg_on_path

if TYPE_CHECKING:
    from pipeline.acquire import EpisodeAudio

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


def _check_required_env_vars(dry_run: bool) -> None:
    """Validate that required environment variables are set.

    Args:
        dry_run: If True, OPENRAG_API_KEY is not required (no upload).

    Raises:
        click.ClickException: If any required variable is missing.
    """
    required: list[str] = []
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


def _build_plain_segments_from_markdown(
    markdown: str,
    diarized_segment_cls: type,
) -> list:
    """Build a list of DiarizedSegment objects from markdown text.

    Used when diarization is skipped. The entire markdown text is wrapped
    in a single DiarizedSegment with ``speaker_label=""`` so that
    ``document.py`` renders the text without a speaker prefix.

    Args:
        markdown:              Markdown text from transcription.
        diarized_segment_cls:  The DiarizedSegment dataclass.

    Returns:
        List containing a single DiarizedSegment (or empty if no text).
    """
    if markdown.strip():
        return [
            diarized_segment_cls(
                speaker_label="",
                start=0.0,
                end=0.0,
                text=markdown.strip(),
            )
        ]
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
    help="Download and transcribe but do NOT upload to OpenRAG.",
)
def ingest(url: str, force: bool, dry_run: bool) -> None:
    """Download, transcribe, and ingest a YouTube episode or playlist.

    URL can be a single video, a playlist, or a channel URL.
    
    Transcripts are produced without speaker labels using Docling's AsrPipeline.
    """
    log = logging.getLogger(__name__)

    # Inject ffmpeg onto PATH before any preflight check or pipeline stage runs.
    # This covers non-interactive shells (cron, launchd, VS Code clean env)
    # where /opt/homebrew/bin is absent from PATH.  Must run before
    # _check_ffmpeg() so that check can succeed after PATH injection.
    ensure_ffmpeg_on_path()

    # Preflight checks
    _check_required_env_vars(dry_run=dry_run)
    _check_ffmpeg()

    # Resolve directories from env vars
    audio_dir = Path(os.environ.get("AUDIO_DIR", "./audio"))
    transcripts_dir = Path(os.environ.get("TRANSCRIPTS_DIR", "./transcripts"))
    state_file = Path(os.environ.get("STATE_FILE", "./state.json"))

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
    # Per-episode processing with global progress bar
    # -----------------------------------------------------------------------
    succeeded = 0
    skipped = 0
    failed = 0

    # Create a global progress bar for the entire batch
    with Progress(
        TextColumn("[bold blue]Overall Progress:"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total} episodes)"),
        console=console,
    ) as global_progress:
        overall_task = global_progress.add_task("Processing episodes", total=total)

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
                global_progress.update(overall_task, advance=1)
                continue

            try:
                _process_episode(
                    episode=episode,
                    transcripts_dir=transcripts_dir,
                    state_file=state_file,
                    force=force,
                    dry_run=dry_run,
                    transcribe_mod=transcribe,
                    document_mod=document,
                    ingest_module=ingest_mod,
                    state_mod=state,
                )
                succeeded += 1

            except (OSError, RuntimeError, ValueError, click.ClickException) as exc:
                log.error(
                    "Episode %s failed: %s",
                    episode.video_id,
                    exc,
                    exc_info=True,
                )
                console.print(f"[red]FAILED[/red] {episode.video_id}: {exc}")
                failed += 1

            finally:
                # Update global progress after each episode (success or failure)
                global_progress.update(overall_task, advance=1)

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
    transcribe_mod: object,
    document_mod: object,
    ingest_module: object,
    state_mod: object,
) -> None:
    """Run all pipeline stages for a single episode.

    Separated from the main ingest() command to keep error handling clean.
    Each stage is wrapped in a rich Progress spinner for user feedback.

    Transcripts are produced without speaker labels using Docling's AsrPipeline.

    Args:
        episode:         EpisodeAudio from the acquisition stage.
        transcripts_dir: Directory for Markdown output files.
        state_file:      Path to state.json.
        force:           Whether to re-ingest if already present.
        dry_run:         If True, skip the OpenRAG upload.
        *_mod:           Imported pipeline module references.
    """
    log = logging.getLogger(__name__)

    # Stage 2: Transcription only (diarization is skipped)
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        transcribe_task = progress.add_task("Transcribing audio…", total=100)

        def update_transcribe_progress(pct: float) -> None:
            progress.update(transcribe_task, completed=pct * 100)

        transcript_result = transcribe_mod.transcribe_audio(  # type: ignore[union-attr]
            episode.audio_path,
            progress_callback=update_transcribe_progress,
        )

        # Update description with model info
        model_info = getattr(transcript_result, "model_info", "")
        progress.update(
            transcribe_task,
            description=f"[green]✓ Transcription complete[/green] ({model_info})",
            completed=100,
        )

    log.info("Diarization skipped — building plain transcript without speaker labels.")

    # Build plain segments from markdown
    markdown = getattr(transcript_result, "markdown", "")
    diarized_segments = _build_plain_segments_from_markdown(markdown, DiarizedSegment)

    # Stage 4: Build document
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            "Building transcript document… (Docling Markdown parser)",
            total=None
        )
        md_path, _docling_doc = document_mod.build_transcript_document(  # type: ignore[union-attr]
            episode=episode,
            segments=diarized_segments,
            transcripts_dir=transcripts_dir,
        )
        progress.update(
            task,
            description="[green]✓ Document built[/green] (Docling Markdown parser)",
            completed=True
        )

    console.print(f"  [dim]Transcript:[/dim] {md_path.name}")

    # Stage 5: Ingest (unless --dry-run)
    openrag_document_id: str | None = None
    openrag_url = os.environ.get("OPENRAG_URL", "http://localhost:3000")

    if dry_run:
        console.print("  [yellow]--dry-run:[/yellow] skipping OpenRAG upload.")
    else:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"Uploading to OpenRAG ({openrag_url})…",
                total=None
            )
            ingest_result = ingest_module.ingest_transcript(  # type: ignore[union-attr]
                transcript_path=md_path,
                force=force,
            )
            progress.update(
                task,
                description=f"[green]✓ Upload complete[/green] → OpenRAG ({openrag_url})",
                completed=True
            )

        if ingest_result.status == "success":
            openrag_document_id = ingest_result.document_id
            console.print(
                f"  [bold green]✓ Document sent to OpenRAG[/bold green]\n"
                f"    [dim]URL:[/dim] {openrag_url}\n"
                f"    [dim]Task ID:[/dim] {openrag_document_id}\n"
                f"    [dim]File:[/dim] {md_path.name}"
            )
        elif ingest_result.status == "already_exists":
            console.print(
                f"  [yellow]Document already exists in OpenRAG[/yellow]\n"
                f"    [dim]URL:[/dim] {openrag_url}\n"
                f"    [dim]File:[/dim] {md_path.name}"
            )
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
