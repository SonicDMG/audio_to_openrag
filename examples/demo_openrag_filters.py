#!/usr/bin/env python3
"""
OpenRAG knowledge filter demo.

Shows how to create and manage knowledge filters for organizing content.
Filters help you group related documents for targeted retrieval.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add parent directory to path to import pipeline modules
sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.openrag_client import OpenRAGClient


def demo_filters():
    """Demonstrate knowledge filter creation and management."""

    # Load environment variables
    load_dotenv()

    api_key = os.getenv("OPENRAG_API_KEY")
    if not api_key:
        print("❌ OPENRAG_API_KEY not set in environment")
        sys.exit(1)

    # Initialize OpenRAG client
    client = OpenRAGClient(
        api_key=api_key,
        base_url=os.getenv("OPENRAG_URL", "http://localhost:3000")
    )

    # Example 1: Create a new filter
    print("📁 Creating knowledge filter...")
    filter_id = client.create_filter(
        name="Tech Podcasts",
        data_sources=["episode1.md", "episode2.md"],
        description="Filter for tech podcast transcripts"
    )
    print(f"✅ Filter created! ID: {filter_id}")

    # Example 2: Use ensure_filter for idempotent operations
    # This will reuse existing filter or create new one
    print("\n📁 Ensuring filter exists...")
    filter_id = client.ensure_filter(
        filter_name="Tech Podcasts",
        filename="episode3.md",
        description="Filter for tech podcast transcripts"
    )
    print(f"✅ Filter ready! ID: {filter_id}")

    print("\n💡 Tip: Filters help organize content by topic, source, or date")
    print("   Use them to create focused knowledge bases for specific queries")


if __name__ == "__main__":
    demo_filters()

# Made with Bob
