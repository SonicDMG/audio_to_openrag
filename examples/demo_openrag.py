#!/usr/bin/env python3
"""
Minimal OpenRAG upload demo.

Shows how to upload documents to OpenRAG using the client wrapper.
This is a simplified example - production code includes retry logic and validation.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path to import pipeline modules
sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.openrag_client import OpenRAGClient


def upload_document(document_path: str) -> None:
    """Upload a document to OpenRAG."""

    # Load environment variables
    load_dotenv()

    # Initialize OpenRAG client with credentials
    api_key = os.getenv("OPENRAG_API_KEY")
    if not api_key:
        print("❌ OPENRAG_API_KEY not set in environment")
        sys.exit(1)

    client = OpenRAGClient(
        api_key=api_key,
        base_url=os.getenv("OPENRAG_URL", "http://localhost:3000")
    )

    # Ingest document and wait for completion
    print(f"📤 Uploading: {document_path}")
    result = client.ingest_document(Path(document_path), wait=True)

    if result.status == "success":
        print(f"✅ Upload complete! Document ID: {result.document_id}")
    else:
        print(f"❌ Upload failed: {result.error}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python demo_openrag.py <file_path>")
        print("Example: python demo_openrag.py sample_files/transcript.md")
        sys.exit(1)

    file_path = sys.argv[1]
    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        sys.exit(1)

    upload_document(file_path)

# Made with Bob
