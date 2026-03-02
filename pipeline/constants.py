"""Shared constants for the audio_to_openrag pipeline."""
import re

# Video ID validation pattern
SAFE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_\-]+$")

# Pipeline version
PIPELINE_VERSION = "1.0.0"

# Title truncation
MAX_TITLE_CHARS = 100

