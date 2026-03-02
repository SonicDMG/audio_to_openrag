"""Configuration management for the audio_to_openrag pipeline."""
from pathlib import Path
import os

def get_audio_dir() -> Path:
    """Get the audio directory from environment or use default."""
    return Path(os.environ.get("AUDIO_DIR", "./audio"))

def get_transcripts_dir() -> Path:
    """Get the transcripts directory from environment or use default."""
    return Path(os.environ.get("TRANSCRIPTS_DIR", "./transcripts"))

def get_state_file() -> Path:
    """Get the state file path from environment or use default."""
    return Path(os.environ.get("STATE_FILE", "./pipeline_state.json"))

def get_openrag_url() -> str:
    """Get OpenRAG URL from environment."""
    return os.environ.get("OPENRAG_URL", "")

def get_openrag_api_key() -> str:
    """Get OpenRAG API key from environment."""
    return os.environ.get("OPENRAG_API_KEY", "")

