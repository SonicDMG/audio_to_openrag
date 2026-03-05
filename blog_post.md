# 🎥 → 🔍 From YouTube Chaos to Searchable Knowledge: Building a Local-First Video RAG Pipeline

## The Problem Every Developer Knows Too Well

You're deep in a debugging session at 2 AM. You *know* you watched a conference talk last month that explained exactly this problem. You remember the speaker had a great analogy about distributed systems... or was it microservices? Was it that 45-minute keynote or the lightning talk?

You spend 20 minutes scrubbing through video timelines, trying to find that one golden nugget of wisdom. Sound familiar?

Here's the thing: **video content is amazing for learning, but terrible for reference**. We've got thousands of hours of conference talks, tutorials, and technical deep-dives locked away in linear video format, completely unsearchable beyond basic title/description metadata.

What if you could ask questions like "What did the speaker say about connection pooling?" and get the exact timestamp with context? 

Enter: **Audio to OpenRAG** 🎯

## What Is This Magic?

Audio to OpenRAG is a Python pipeline that transforms YouTube videos into semantically searchable knowledge bases. It's like giving your video library a photographic memory.

Here's what happens under the hood:

1. **Download** videos from YouTube (single videos, playlists, or entire channels)
2. **Transcribe** them locally using Whisper Turbo (via Docling's ASR wrapper) with timestamp preservation
3. **Ingest** transcripts into OpenRAG for semantic search and RAG applications

The best part? **Everything runs locally**. No API keys for transcription, no sending your data to third-party services, no usage limits. Your conference recordings stay on your machine.

## Show Me the Code

Getting started is ridiculously simple:

```bash
# Install dependencies (using uv for blazing-fast installs)
uv sync

# Ingest a single video
uv run python main.py ingest "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# Ingest an entire channel (because why not?)
uv run python main.py ingest "https://www.youtube.com/@YourFavoriteDevChannel"

# Ingest a playlist of conference talks
uv run python main.py ingest "https://www.youtube.com/playlist?list=..."
```

That's it. Seriously. The tool handles everything else:

- ✅ Downloads videos (audio-only to save space)
- ✅ Transcribes with timestamps
- ✅ Exports in multiple formats (DocTags + Markdown)
- ✅ Ingests into OpenRAG
- ✅ Tracks state to avoid duplicate work
- ✅ Shows beautiful progress bars because we're not savages

## The Technical Goodies 🛠️

### 1. Privacy-First Transcription

Unlike cloud-based solutions, Audio to OpenRAG uses **OpenAI's Whisper Turbo** (accessed via Docling's ASR wrapper) running entirely on your machine. No audio leaves your computer. No API quotas. No surprise bills.

```python
# The transcription happens locally with full timestamp preservation
# Whisper runs via Docling's convenient wrapper
transcriber = DoclingTranscriber()
doc = transcriber.transcribe(audio_path)
# Returns a DoclingDocument with TrackSource objects containing timestamps
```

### 2. Smart State Management

The pipeline is **idempotent** - run it multiple times on the same content and it won't duplicate work. It maintains a state file tracking what's been processed:

```python
# Already transcribed this video? Skip it.
# Already ingested this transcript? Skip it.
# New video in the channel? Process only that one.
```

This is *huge* for batch processing channels with hundreds of videos.

### 3. Timestamp Preservation

Here's where it gets interesting. The tool doesn't just dump raw text - it preserves the **temporal structure** of the content:

```markdown
[00:00:15] Welcome everyone to this talk about distributed systems...
[00:02:30] Let's dive into the CAP theorem...
[00:15:45] Here's where things get interesting with eventual consistency...
```

When you search later, you get the exact moment in the video where the topic was discussed. No more scrubbing!

### 4. Dual Export Format

The pipeline exports transcripts in two formats:

- **DocTags**: Preserves the full DoclingDocument structure for programmatic access
- **Markdown**: Human-readable format with timestamps for easy browsing

Both get ingested into OpenRAG, giving you flexibility in how you query the data.

### 5. Async-Sync Bridge

The OpenRAG SDK is async, but the pipeline needs to coordinate multiple sync operations (video download, transcription). The tool includes a clean async-sync bridge:

```python
async def ingest_async(doc: DoclingDocument, metadata: dict):
    # Async OpenRAG operations
    await openrag_client.ingest(doc, metadata)

# Called from sync context
asyncio.run(ingest_async(doc, metadata))
```

### 6. Exponential Backoff

Network hiccups? No problem. The pipeline includes retry logic with exponential backoff for resilient operation:

```python
@retry(max_attempts=3, backoff_factor=2)
def download_video(url: str):
    # Handles transient failures gracefully
```

## Real-World Use Cases

### 📚 Conference Talk Library

Ingest every talk from your favorite conference (PyCon, JSConf, KubeCon) and build a searchable knowledge base:

```bash
uv run python main.py ingest "https://www.youtube.com/@PyConUS"
```

Later, ask your RAG system: *"What are the best practices for async Python mentioned in PyCon talks?"*

### 🎓 Course Material Search

Taking an online course with 50+ video lessons? Make it searchable:

```bash
uv run python main.py ingest "https://www.youtube.com/playlist?list=COURSE_PLAYLIST"
```

Now you can find that specific explanation about closures without rewatching 10 hours of content.

### 🏢 Internal Training Videos

Got company training videos or recorded tech talks? Keep them private and searchable:

```bash
# Works with unlisted videos too
uv run python main.py ingest "https://www.youtube.com/watch?v=UNLISTED_VIDEO"
```

### 🎙️ Podcast Archives

Many podcasts publish to YouTube. Make your favorite tech podcast searchable:

```bash
uv run python main.py ingest "https://www.youtube.com/@YourFavoriteTechPodcast"
```

## The Tech Stack

Built on solid foundations:

- **yt-dlp**: The Swiss Army knife of video downloading
- **Whisper Turbo**: OpenAI's state-of-the-art speech recognition model
- **Docling**: IBM Research's document processing framework (provides ASR wrapper)
- **OpenRAG SDK**: Semantic search and RAG capabilities
- **Rich**: Beautiful terminal output (because UX matters in CLI tools too)
- **Python 3.12+**: Modern Python with all the goodies

## What Makes This Different?

**Local-First**: Your data never leaves your machine during transcription. In an era of cloud-everything, this is refreshing.

**Batch-Friendly**: Process entire channels or playlists without babysitting the script.

**Developer-Focused**: Clean code, proper error handling, progress indicators, and state management. Built by developers, for developers.

**No Vendor Lock-In**: Uses open standards (Markdown, DocTags) and open-source tools throughout.

**Timestamp Awareness**: Unlike simple transcription tools, this preserves temporal context - crucial for video content.

## Performance Notes

Transcription speed depends on your hardware:

- **CPU**: ~0.5-1x realtime (a 10-minute video takes 10-20 minutes)
- **GPU**: ~5-10x realtime (a 10-minute video takes 1-2 minutes)

The tool automatically uses GPU acceleration if available (CUDA, MPS on Apple Silicon).

## Attribution & Building Blocks

This project builds upon [Tejas Kumar's example-docling-media](https://github.com/TejasQ/example-docling-media), which demonstrates using Docling's ASR wrapper for media transcription. We've extended it with batch processing, state management, and OpenRAG integration.

**Core technologies:**
- [OpenAI Whisper](https://github.com/openai/whisper) - The transcription engine
- [Docling](https://github.com/DS4SD/docling) - Document processing framework and ASR wrapper
- [OpenRAG](https://github.com/langflow-ai/openrag) - Semantic search and RAG platform

Standing on the shoulders of giants! 🙏

## Get Started Today

```bash
# Clone the repo
git clone https://github.com/yourusername/audio_to_openrag
cd audio_to_openrag

# Install dependencies
uv sync

# Start ingesting
uv run python main.py ingest "YOUR_YOUTUBE_URL"
```

Check out the [README](README.md) for detailed configuration options and advanced usage.

## What's Next?

Some ideas for extending this pipeline:

- 🔊 Support for local video files (not just YouTube)
- 🌐 Multi-language transcription support
- 📊 Analytics on your video library (most discussed topics, etc.)
- 🔗 Integration with note-taking apps (Obsidian, Notion)
- 🎯 Speaker diarization (who said what)

## The Bottom Line

Video content is incredible for learning but frustrating for reference. Audio to OpenRAG bridges that gap, turning your video library into a searchable, queryable knowledge base - all running locally on your machine.

No more scrubbing through timelines at 2 AM. No more "I know I saw this somewhere" moments. Just ask your question and get the answer with timestamps.

Your future self will thank you. 🚀

---

**Ready to make your video library searchable?** Star the repo, try it out, and let me know what you build with it!

*Have questions or ideas? Open an issue or PR. This is a community project - contributions welcome!*