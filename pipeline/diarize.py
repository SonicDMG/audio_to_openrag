"""
pipeline/diarize.py — Speaker Diarization Stage

Identifies speaker boundaries in an audio file using pyannote.audio
(speaker-diarization-3.1), then merges the diarization output with
Whisper timestamped segments via maximum-overlap assignment.

Raw pyannote speaker labels (SPEAKER_00, SPEAKER_01, …) are mapped to
human-readable labels (Speaker 1, Speaker 2, …) based on total speaking
time — the dominant speaker (most cumulative speech) becomes "Speaker 1"
(assumed to be the host for The Flow podcast).

Security controls (OWASP):
- HF_TOKEN loaded exclusively from environment variables; never logged
- Model loading wrapped in try/except with helpful error messages
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Gap threshold (seconds) for consolidating consecutive same-speaker segments
CONSOLIDATION_GAP_SECONDS: float = 2.0


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class DiarizedSegment:
    """A single speaker-attributed transcript segment.

    Attributes:
        speaker_label: Human-readable label, e.g. "Speaker 1", "Speaker 2".
        start:         Segment start time in seconds.
        end:           Segment end time in seconds.
        text:          Transcribed text for this segment.
    """

    speaker_label: str
    start: float
    end: float
    text: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _select_device() -> "torch.device":  # type: ignore[name-defined]
    """Select the best available compute device for pyannote inference.

    Priority order:
    1. Apple Silicon MPS (fastest on macOS with M-series chips)
    2. NVIDIA CUDA
    3. CPU (fallback)

    Returns:
        A ``torch.device`` instance.
    """
    import torch  # type: ignore[import]

    if torch.backends.mps.is_available():
        logger.info("Using Apple Silicon MPS device for diarization.")
        return torch.device("mps")
    if torch.cuda.is_available():
        logger.info("Using CUDA device for diarization.")
        return torch.device("cuda")
    logger.info("Using CPU device for diarization (no GPU detected).")
    return torch.device("cpu")


def _load_diarization_pipeline() -> Any:
    """Load the pyannote speaker-diarization-3.1 pipeline.

    Reads ``HF_TOKEN`` from the environment. The token value is never logged.

    Returns:
        A loaded pyannote ``Pipeline`` instance, moved to the best device.

    Raises:
        RuntimeError: If ``HF_TOKEN`` is missing, the model cannot be loaded,
                      or the Hugging Face Hub license has not been accepted.
    """
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN environment variable is not set. "
            "Obtain a token at https://huggingface.co/settings/tokens and add it to .env. "
            "You must also accept the model license at: "
            "https://huggingface.co/pyannote/speaker-diarization-3.1"
        )

    try:
        from pyannote.audio import Pipeline  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(
            f"pyannote.audio is not installed: {exc}\n"
            "Install with: pip install 'pyannote.audio>=3.1.0'"
        ) from exc

    logger.info("Loading pyannote/speaker-diarization-3.1 model…")
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token,
        )
    except Exception as exc:
        # Check for GatedRepoError first (valid token, but license not accepted).
        # Import lazily — huggingface_hub is a transitive dep of pyannote.audio.
        try:
            from huggingface_hub.errors import GatedRepoError  # type: ignore[import]
            _gated_type: type = GatedRepoError
        except ImportError:
            _gated_type = type(None)  # sentinel — will never match

        if isinstance(exc, _gated_type) or (
            hasattr(exc, "__cause__") and isinstance(exc.__cause__, _gated_type)
        ):
            raise RuntimeError(
                "Access to pyannote/speaker-diarization-3.1 is restricted.\n"
                "Your HF_TOKEN is valid, but your Hugging Face account has not been\n"
                "granted access to this gated model. To fix this:\n"
                "  1. Visit https://huggingface.co/pyannote/speaker-diarization-3.1\n"
                "  2. Click 'Agree and access repository' to accept the license terms\n"
                "  3. Re-run the pipeline (no code changes needed)"
            ) from exc

        err_str = str(exc).lower()
        if "401" in err_str or "unauthorized" in err_str:
            raise RuntimeError(
                "Hugging Face token authentication failed for pyannote/speaker-diarization-3.1.\n"
                "Your HF_TOKEN is invalid or expired.\n"
                "  1. Regenerate your token at https://huggingface.co/settings/tokens\n"
                "  2. Update HF_TOKEN in your .env file"
            ) from exc
        raise RuntimeError(
            f"Failed to load pyannote diarization model: {exc}"
        ) from exc

    if pipeline is None:
        raise RuntimeError(
            "pyannote Pipeline.from_pretrained returned None for "
            "pyannote/speaker-diarization-3.1. "
            "This is unexpected — check that the model ID is correct."
        )

    device = _select_device()
    pipeline = pipeline.to(device)
    logger.info("pyannote pipeline loaded and moved to device: %s", device)
    return pipeline


def _build_speaker_map(raw_segments: list[dict]) -> dict[str, str]:
    """Map raw pyannote speaker labels to human-readable labels.

    The speaker with the most total speaking time is mapped to "Speaker 1"
    (assumed host), the next to "Speaker 2" (assumed guest), and so on.

    Args:
        raw_segments: List of dicts with keys: start, end, speaker (raw label).

    Returns:
        Dict mapping raw label (e.g. "SPEAKER_00") to readable label
        (e.g. "Speaker 1").
    """
    speaking_time: dict[str, float] = defaultdict(float)
    for seg in raw_segments:
        speaking_time[seg["speaker"]] += seg["end"] - seg["start"]

    sorted_speakers = sorted(
        speaking_time, key=lambda s: speaking_time[s], reverse=True
    )

    speaker_map: dict[str, str] = {}
    for idx, raw_label in enumerate(sorted_speakers, start=1):
        speaker_map[raw_label] = f"Speaker {idx}"

    logger.debug("Speaker map: %s", speaker_map)
    return speaker_map


def _merge_segments(
    whisper_segments: list[dict],
    diarization_segments: list[dict],
    speaker_map: dict[str, str],
) -> list[DiarizedSegment]:
    """Assign speaker labels to Whisper segments via maximum-overlap alignment.

    For each Whisper segment, finds the diarization segment with the greatest
    temporal overlap and assigns that speaker's label. If no overlap is found,
    the segment is labelled "Unknown".

    Complexity: O(W × D) — negligible for typical podcast lengths.

    Args:
        whisper_segments:     List of dicts: {start, end, text}.
        diarization_segments: List of dicts: {start, end, speaker}.
        speaker_map:          Mapping from raw pyannote label to readable label.

    Returns:
        List of :class:`DiarizedSegment` objects with readable speaker labels.
    """
    labeled: list[DiarizedSegment] = []

    for w in whisper_segments:
        w_start: float = w["start"]
        w_end: float = w["end"]
        w_text: str = w.get("text", "").strip()

        best_speaker: str = "Unknown"
        best_overlap: float = 0.0

        for d in diarization_segments:
            overlap = max(0.0, min(w_end, d["end"]) - max(w_start, d["start"]))
            if overlap > best_overlap:
                best_overlap = overlap
                raw_label: str = d["speaker"]
                best_speaker = speaker_map.get(raw_label) or raw_label

        labeled.append(
            DiarizedSegment(
                speaker_label=best_speaker,
                start=w_start,
                end=w_end,
                text=w_text,
            )
        )

    return labeled


def _consolidate_segments(segments: list[DiarizedSegment]) -> list[DiarizedSegment]:
    """Merge consecutive same-speaker segments separated by a small gap.

    Consecutive segments from the same speaker are merged into a single
    paragraph when the gap between them is ≤ CONSOLIDATION_GAP_SECONDS.
    This improves readability by reducing fragmentation.

    Args:
        segments: List of :class:`DiarizedSegment` objects (speaker-labeled).

    Returns:
        Consolidated list of :class:`DiarizedSegment` objects.
    """
    if not segments:
        return []

    consolidated: list[DiarizedSegment] = [segments[0]]

    for seg in segments[1:]:
        prev = consolidated[-1]
        gap = seg.start - prev.end

        if seg.speaker_label == prev.speaker_label and gap <= CONSOLIDATION_GAP_SECONDS:
            # Merge into the previous segment
            consolidated[-1] = DiarizedSegment(
                speaker_label=prev.speaker_label,
                start=prev.start,
                end=seg.end,
                text=prev.text.rstrip() + " " + seg.text.lstrip(),
            )
        else:
            consolidated.append(seg)

    return consolidated


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def diarize_audio(
    audio_path: Path,
    segments: list[dict] | None,
    num_speakers: int = 2,
) -> list[DiarizedSegment]:
    """Run speaker diarization and merge results with Whisper segments.

    Loads the pyannote/speaker-diarization-3.1 model (downloading on first
    use), runs diarization on the audio file, then merges the speaker
    boundaries with the provided Whisper timestamped segments using
    maximum-overlap assignment.

    Speaker labels are mapped from raw pyannote identifiers (SPEAKER_00, …)
    to human-readable labels (Speaker 1, Speaker 2, …) based on total
    speaking time.

    Args:
        audio_path:   Path to the local audio file (.mp3 or similar).
        segments:     Whisper timestamped segments from
                      :class:`~pipeline.transcribe.TranscriptResult`.
                      Each dict must have keys: ``start``, ``end``, ``text``.
        num_speakers: Expected number of distinct speakers. Constraining this
                      value improves diarization accuracy. Default: 2.

    Returns:
        List of :class:`DiarizedSegment` objects, consolidated by speaker.

    Raises:
        FileNotFoundError: If *audio_path* does not exist.
        RuntimeError:      If *segments* is None (Docling did not expose
                           timestamps and the fallback also failed), or if
                           the pyannote model cannot be loaded.
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if segments is None:
        logger.warning(
            "No timestamped segments available for diarization (Docling did not expose "
            "segment-level timestamps and the faster-whisper fallback also failed). "
            "Skipping diarization — transcript will be written without speaker labels. "
            "To enable diarization, install faster-whisper: pip install faster-whisper"
        )
        return []

    if not segments:
        logger.warning(
            "Whisper segments list is empty — diarization will produce no output."
        )
        return []

    # Load pyannote pipeline (raises RuntimeError with helpful message on failure)
    diar_pipeline = _load_diarization_pipeline()

    logger.info(
        "Running pyannote diarization on '%s' (num_speakers=%d)…",
        audio_path.name,
        num_speakers,
    )

    try:
        diarization = diar_pipeline(str(audio_path), num_speakers=num_speakers)
    except Exception as exc:
        raise RuntimeError(
            f"pyannote diarization failed for '{audio_path.name}': {exc}"
        ) from exc

    # Extract raw diarization segments from pyannote output
    raw_diar_segments: list[dict] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        raw_diar_segments.append(
            {
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker,
            }
        )

    logger.info(
        "pyannote produced %d raw diarization segments.", len(raw_diar_segments)
    )

    if not raw_diar_segments:
        logger.warning(
            "pyannote returned no diarization segments. "
            "All Whisper segments will be labelled 'Unknown'."
        )
        return [
            DiarizedSegment(
                speaker_label="Unknown",
                start=s["start"],
                end=s["end"],
                text=s.get("text", "").strip(),
            )
            for s in segments
        ]

    # Build speaker label mapping (dominant speaker → Speaker 1, etc.)
    speaker_map = _build_speaker_map(raw_diar_segments)

    # Merge Whisper segments with diarization via max-overlap assignment
    labeled_segments = _merge_segments(segments, raw_diar_segments, speaker_map)

    # Consolidate consecutive same-speaker segments
    consolidated = _consolidate_segments(labeled_segments)

    logger.info(
        "Diarization complete: %d segments → %d consolidated segments.",
        len(labeled_segments),
        len(consolidated),
    )

    return consolidated

# Made with Bob
