#!/usr/bin/env python3
"""
Complete audio-to-RAG pipeline demo.

Shows the full workflow: transcribe audio with Docling, save transcript,
and upload to OpenRAG. This is a simplified example - production code includes
comprehensive error handling, logging, and state management.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from docling.document_converter import DocumentConverter, AudioFormatOption
from docling.datamodel.pipeline_options import AsrPipelineOptions
from docling.datamodel import asr_model_specs
from docling.pipeline.asr_pipeline import AsrPipeline
from docling.datamodel.base_models import InputFormat

# Add parent directory to path to import pipeline modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.openrag_client import OpenRAGClient


def run_full_pipeline(audio_path: str, transcript_dir: str = "transcripts"):
    """Run complete audio-to-RAG pipeline."""

    audio_file_path = Path(audio_path)
    if not audio_file_path.exists():
        print(f"❌ File not found: {audio_path}")
        sys.exit(1)

    # Create output directory
    output_path = Path(transcript_dir)
    output_path.mkdir(exist_ok=True)

    # Step 1: Transcribe with Docling
    print("=" * 60)
    print("STEP 1: Transcribing audio with Docling")
    print("=" * 60)

    pipeline_options = AsrPipelineOptions()
    pipeline_options.asr_options = asr_model_specs.WHISPER_TURBO

    converter = DocumentConverter(
        format_options={
            InputFormat.AUDIO: AudioFormatOption(
                pipeline_cls=AsrPipeline,
                pipeline_options=pipeline_options,
            )
        }
    )

    print(f"📝 Transcribing: {audio_file_path.name}")
    result = converter.convert(str(audio_file_path))
    markdown = result.document.export_to_markdown()
    print("✅ Transcription complete!")

    # Step 2: Save transcript
    print("\n" + "=" * 60)
    print("STEP 2: Saving transcript")
    print("=" * 60)

    transcript_filename = audio_file_path.stem + ".md"
    transcript_path = output_path / transcript_filename
    transcript_path.write_text(markdown, encoding="utf-8")
    print(f"💾 Saved to: {transcript_path}")

    # Step 3: Upload to OpenRAG
    print("\n" + "=" * 60)
    print("STEP 3: Uploading to OpenRAG")
    print("=" * 60)

    load_dotenv()
    api_key = os.getenv("OPENRAG_API_KEY")
    if not api_key:
        print("⚠️  OPENRAG_API_KEY not set - skipping upload")
        print("   Set the API key in .env to enable OpenRAG upload")
        return

    client = OpenRAGClient(
        api_key=api_key,
        base_url=os.getenv("OPENRAG_URL", "http://localhost:3000")
    )

    upload_result = client.ingest_document(transcript_path, wait=True)

    if upload_result.status == "success":
        print(f"✅ Upload complete! Document ID: {upload_result.document_id}")
    else:
        print(f"❌ Upload failed: {upload_result.error}")

    # Summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"📄 Transcript: {transcript_path}")
    if upload_result.status == "success":
        print(f"🔗 OpenRAG ID: {upload_result.document_id}")
    print("\n✨ Your audio is now searchable in OpenRAG!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python demo_full_pipeline.py <audio_file> [output_dir]")
        print("Example: python demo_full_pipeline.py sample_files/test.mp3")
        sys.exit(1)

    audio_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "transcripts"

    run_full_pipeline(audio_file, output_dir)

# Made with Bob
