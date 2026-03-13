#!/usr/bin/env python3
"""
Minimal Docling transcription demo.

Shows how to transcribe audio/video files using Docling with Whisper Turbo.
This is a simplified example - production code includes error handling and logging.
"""

import sys
from pathlib import Path

from docling.datamodel import asr_model_specs
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import AsrPipelineOptions
from docling.document_converter import AudioFormatOption, DocumentConverter
from docling.pipeline.asr_pipeline import AsrPipeline


def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio file to markdown using Docling."""

    # Configure Whisper Turbo model for fast, accurate transcription
    pipeline_options = AsrPipelineOptions()
    pipeline_options.asr_options = asr_model_specs.WHISPER_TURBO

    # Create converter with audio pipeline
    converter = DocumentConverter(
        format_options={
            InputFormat.AUDIO: AudioFormatOption(
                pipeline_cls=AsrPipeline,
                pipeline_options=pipeline_options,
            )
        }
    )

    # Convert audio to document and export as markdown
    print(f"📝 Transcribing: {audio_path}")
    result = converter.convert(audio_path)
    markdown = result.document.export_to_markdown()

    print("✅ Transcription complete!")
    return markdown


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python demo_docling.py <audio_file>")
        print("Example: python demo_docling.py sample_files/test.mp3")
        sys.exit(1)

    audio_file = sys.argv[1]
    if not Path(audio_file).exists():
        print(f"❌ File not found: {audio_file}")
        sys.exit(1)

    # Transcribe and print result
    transcript = transcribe_audio(audio_file)
    print("\n" + "="*50)
    print("TRANSCRIPT:")
    print("="*50)
    print(transcript)

# Made with Bob
