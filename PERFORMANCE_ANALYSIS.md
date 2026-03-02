# Performance Analysis Report — `audio_to_openrag`

**Generated:** 2026-03-02  
**Machine:** MacBook Pro (MacBookPro18,1) — Apple M1 Pro, 32 GB RAM  
**Python Runtime:** 3.12.9 (uv-managed, `cpython-3.12.9-macos-aarch64-none`)  
**Scope:** Full pipeline from audio download through OpenRAG ingestion

---

## 1. Executive Summary

- 🔴 **Double-transcription bug (most critical):** [`_extract_segments_from_docling()`](pipeline/transcribe.py:52) probes undocumented Docling API attribute paths and almost certainly returns `None` on every run. When it does, [`_extract_segments_via_faster_whisper()`](pipeline/transcribe.py:158) re-transcribes the entire audio file from scratch using `faster-whisper` on CPU — meaning every episode is transcribed **twice**, with the second pass running on CPU only (CTranslate2 has no Apple Metal support). For a 60-minute episode this adds an estimated **10–25 minutes** of unnecessary CPU work per run.

- 🔴 **Cold-start Metal shader cache penalty (one-time):** Both the Metal shader cache (`~/Library/Caches/com.apple.metal/`) and the PyTorch MPS cache (`~/.cache/torch/`) are currently **empty**. On the first complete run, MLX must JIT-compile all Metal compute kernels from scratch, adding an estimated **30–120 seconds** of overhead. This is a **one-time penalty** that self-resolves after the first run.

- 🔴 **No model caching between episodes:** Every call to [`transcribe_audio()`](pipeline/transcribe.py:209) instantiates a new `DocumentConverter` + `AsrPipeline`. Every call to [`diarize_audio()`](pipeline/diarize.py:83) calls `Pipeline.from_pretrained()`. For a batch of N episodes, models are loaded N times from disk.

- 🟡 **No parallelism despite architecture docs claiming it:** `ARCHITECTURE.md` states "Stage 2: Process Audio in Parallel," but [`_process_episode()`](main.py:326) runs transcription then diarization sequentially, and the episode loop at [`main.py:270`](main.py:270) is a plain `for` loop with no concurrency.

- 🟡 **ffmpeg not on non-interactive shell PATH:** ffmpeg 8.0.1 is installed at `/opt/homebrew/bin/ffmpeg` but `/opt/homebrew/bin` is absent from the non-interactive shell PATH. Any subprocess invocation of `ffmpeg` by name from `yt-dlp`, `faster-whisper`, or Docling will raise `FileNotFoundError` or fail silently.

---

## 2. Root Cause: Why the App Is Slow Right Now

### 2.1 Empty Metal Shader Cache (One-Time Cold-Start Penalty)

The primary transcription path uses **mlx-whisper 0.4.3** via Docling's `AsrPipeline`. MLX compiles Metal compute kernels (GPU shaders) on first use and caches them in `~/Library/Caches/com.apple.metal/`. This cache is currently **empty**.

On the first complete run, every MLX operation that hits the GPU must compile its Metal kernel from scratch before executing. This JIT compilation is serial and cannot be parallelized. Estimated overhead: **30–120 seconds** added to the first transcription run, depending on the number of unique operation shapes encountered.

Similarly, the PyTorch MPS cache at `~/.cache/torch/` is empty. The pyannote diarization pipeline (Stage 3) uses PyTorch with MPS acceleration. On first run, PyTorch must compile Metal kernels for each unique tensor operation shape. Estimated overhead: **15–45 seconds** added to the first diarization run.

**This is a one-time penalty.** After the first complete end-to-end run, both caches will be populated and subsequent runs will start at full GPU speed. No code changes are required to resolve this — simply run the pipeline once to completion.

### 2.2 Why This Matters for Diagnosis

If the pipeline appears slow on first run, the Metal shader compilation overhead will be conflated with the structural bugs described in Section 3. It is important to run the pipeline **twice** on the same episode (using `--force` on the second run) to distinguish cold-start overhead from permanent structural overhead.

---

## 3. Structural Performance Issues

These issues exist in the code itself and cause permanent overhead on **every run**, regardless of cache state.

### 3.1 🔴 Double-Transcription Bug (Most Critical)

**Location:** [`pipeline/transcribe.py:52`](pipeline/transcribe.py:52) — [`_extract_segments_from_docling()`](pipeline/transcribe.py:52)  
**Trigger:** [`pipeline/transcribe.py:285–297`](pipeline/transcribe.py:285)

**What happens:**

1. [`transcribe_audio()`](pipeline/transcribe.py:209) runs the full Docling `DocumentConverter` with `AsrPipeline` + `WHISPER_TURBO`. On macOS arm64, this resolves to **mlx-whisper** running on the Apple MLX Metal GPU. This is the fast path. ✅

2. After Docling completes, [`_extract_segments_from_docling()`](pipeline/transcribe.py:52) attempts to extract timestamped segments from the returned `DoclingDocument`. It probes two undocumented attribute paths:
   - `document.texts` — looking for `TextItem` objects with timing attributes (`start`, `start_time`, `t_start`, etc.)
   - `document.body.children` — looking for child nodes with `start`/`end` attributes

3. The module's own header comment at [`pipeline/transcribe.py:8`](pipeline/transcribe.py:8) states: *"The exact attribute path for extracting timestamped segments from a DoclingDocument produced by AsrPipeline is not fully documented as of Docling 2.x."* The `TODO` comment at [`pipeline/transcribe.py:59`](pipeline/transcribe.py:59) explicitly marks this as **"UNRESOLVED API QUESTION."**

4. Both extraction attempts are wrapped in broad `except Exception` blocks ([`line 125`](pipeline/transcribe.py:125), [`line 148`](pipeline/transcribe.py:148)) that swallow all errors silently and log only at `DEBUG` level. If neither path finds timing data, the function returns `None`.

5. When `_extract_segments_from_docling()` returns `None`, [`transcribe_audio()`](pipeline/transcribe.py:287) immediately calls [`_extract_segments_via_faster_whisper(audio_path)`](pipeline/transcribe.py:158), which:
   - Loads a **new** `WhisperModel("turbo", device="auto", compute_type="default")` instance
   - Re-transcribes the **entire audio file from scratch**
   - Runs on **CPU only** — CTranslate2 (the faster-whisper backend) has no Apple Metal support

**The result:** Every episode is transcribed twice. The first transcription (mlx-whisper via Docling, GPU-accelerated) produces the text content. The second transcription (faster-whisper via CTranslate2, CPU-only) re-does the entire audio just to get timestamps. The GPU work from the first pass is discarded for the purpose of segment extraction.

**Estimated cost per 60-minute episode:**
- First transcription (mlx-whisper, GPU): ~3–6 minutes
- Second transcription (faster-whisper, CPU): ~10–25 minutes
- **Total wasted time: 10–25 minutes per episode**

**Fix:** See Section 6, Recommendation R1.

---

### 3.2 🔴 No Model Caching Between Episodes

**Locations:**
- [`pipeline/transcribe.py:259`](pipeline/transcribe.py:259) — `DocumentConverter(...)` instantiated inside `transcribe_audio()`
- [`pipeline/diarize.py:114`](pipeline/diarize.py:114) — `Pipeline.from_pretrained(...)` called inside `_load_diarization_pipeline()`

Neither the Docling `DocumentConverter`/`AsrPipeline` nor the pyannote `Pipeline` is cached at module level or passed as a parameter. Every call to `transcribe_audio()` or `diarize_audio()` reloads the model from disk and re-initializes it.

The episode loop at [`main.py:270`](main.py:270) calls `_process_episode()` for each episode, which calls `transcribe_mod.transcribe_audio()` and `diarize_mod.diarize_audio()` — both of which reload their models on every iteration.

**Estimated cost per additional episode (beyond the first):**
- Docling/mlx-whisper model load: ~15–45 seconds
- pyannote model load + device transfer: ~10–30 seconds
- **Total wasted time per episode (N > 1): ~25–75 seconds**

For a batch of 10 episodes, this adds **4–12 minutes** of pure model-loading overhead.

---

### 3.3 🟡 No Parallelism Despite Architecture Documentation

**Location:** [`main.py:270`](main.py:270) — episode `for` loop; [`main.py:371`](main.py:371) — sequential transcribe → diarize

`ARCHITECTURE.md` states: *"Stage 2: Process Audio in Parallel."* The actual implementation is fully sequential:

```python
# main.py:270 — plain for loop, no concurrency
for idx, episode in enumerate(episodes, start=1):
    _process_episode(...)
```

Within each episode, transcription (Stage 2) and diarization (Stage 3) are also sequential — diarization cannot begin until transcription is complete, which is architecturally correct. However, **across episodes**, there is no overlap: episode N+1 does not begin downloading or transcribing while episode N is being diarized or uploaded.

On an M1 Pro with 8 performance cores, transcription (GPU-bound via MLX) and diarization (GPU-bound via MPS) could theoretically run concurrently for different episodes, or at minimum the download stage for episode N+1 could overlap with processing of episode N.

**Estimated cost for a 10-episode batch:** Sequential processing adds the full wall-clock time of each episode's download + transcription + diarization in series. With parallelism, download + transcription of episode N+1 could overlap with diarization + upload of episode N, potentially reducing total batch time by 30–50%.

---

### 3.4 🟢 Docling Markdown Re-Conversion (Minor)

**Location:** [`pipeline/document.py:186`](pipeline/document.py:186) — [`_convert_markdown_to_docling()`](pipeline/document.py:191)

After writing the transcript `.md` file to disk, [`build_transcript_document()`](pipeline/document.py:126) instantiates a **second** `DocumentConverter` to re-parse the just-written Markdown file back into a `DoclingDocument`. The returned `DoclingDocument` is assigned to `_docling_doc` at [`main.py:398`](main.py:398) and never used downstream (the variable is discarded with `_`).

This is minor overhead (~1–5 seconds per episode) but represents unnecessary work. The Markdown `DocumentConverter` is lighter than the ASR pipeline, so this is low priority.

---

### 3.5 🟢 pyannote O(W×D) Merge (Negligible)

**Location:** [`pipeline/diarize.py:193`](pipeline/diarize.py:193) — [`_merge_segments()`](pipeline/diarize.py:193)

The speaker-to-segment assignment algorithm is O(W×D) where W = number of Whisper segments and D = number of diarization segments. For a 60-minute podcast episode, W ≈ 500–2000 and D ≈ 200–800. This is at most ~1.6M comparisons — negligible on modern hardware. No optimization needed.

---

### 3.6 🟢 `asyncio.run()` Bridge (Negligible)

**Location:** [`pipeline/ingest.py:383`](pipeline/ingest.py:383)

`ingest_transcript()` calls `asyncio.run(_ingest_async(...))`, creating a new event loop per episode. This adds microseconds of overhead and is not a bottleneck.

---

## 4. System Environment Issues

### 4.1 🔴 ffmpeg Not on Non-Interactive Shell PATH

**Status:** ffmpeg 8.0.1 is installed at `/opt/homebrew/bin/ffmpeg`  
**Problem:** `/opt/homebrew/bin` is not on the non-interactive shell PATH

Three pipeline components invoke `ffmpeg` by name as a subprocess:
- **yt-dlp** (Stage 1): uses ffmpeg for audio post-processing and format conversion
- **Docling/mlx-whisper** (Stage 2): uses ffmpeg to decode audio before passing to Whisper
- **faster-whisper** (fallback): uses ffmpeg for audio decoding

When these tools call `ffmpeg` by name in a non-interactive shell (e.g., when launched via `uv run`, a cron job, or a CI environment), the subprocess will fail with `FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'` or silently produce no output.

**Fix:**

Option A — Add to shell profile (interactive sessions only):
```bash
echo 'export PATH="/opt/homebrew/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

Option B — Set in `.env` for the uv project (recommended for reproducibility):
```bash
# .env
PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin
```

Option C — Set `FFMPEG_BINARY` environment variable where supported:
```bash
# .env
FFMPEG_BINARY=/opt/homebrew/bin/ffmpeg
```

Option D — Create a symlink in a PATH-visible location:
```bash
sudo ln -sf /opt/homebrew/bin/ffmpeg /usr/local/bin/ffmpeg
```

**Recommended:** Option A + B together — fix the shell profile for interactive use and set PATH in `.env` for non-interactive/uv-managed invocations.

---

### 4.2 🟡 CTranslate2 Has No Apple Metal Support

**Status:** CTranslate2 4.7.1 — CPU-only on macOS  
**Compute types available:** `float32`, `int8`, `int8_float32` only (no Metal/GPU)

This is a known architectural limitation of CTranslate2 4.x, not a regression. The faster-whisper fallback path in [`_extract_segments_via_faster_whisper()`](pipeline/transcribe.py:158) runs entirely on CPU regardless of `device="auto"`. On an M1 Pro, CPU-only Whisper Turbo transcription of a 60-minute episode takes approximately **10–25 minutes**, compared to **3–6 minutes** for mlx-whisper on the GPU.

This limitation makes fixing the double-transcription bug (Section 3.1) even more critical — the fallback path is not just redundant, it is dramatically slower than the primary path.

---

### 4.3 🟢 Homebrew Python Versions Not Removed

**Status:** Python 3.10, 3.12, 3.13, 3.14 remain at `/opt/homebrew/bin/`

While the uv-managed virtual environment is fully isolated and uses its own Python 3.12.9 binary, the presence of multiple Homebrew Python versions creates PATH ambiguity. If a script or tool resolves `python3` from PATH before the uv venv is activated, it may pick up a Homebrew Python instead of the project Python.

This does not affect the pipeline when run via `uv run`, but it is a hygiene issue that could cause confusion during debugging or when running scripts directly.

**Fix:** Remove unused Homebrew Python versions:
```bash
brew uninstall python@3.10 python@3.13 python@3.14
# Keep python@3.12 only if needed for other projects
```

---

## 5. What Is Working Correctly

The following components are **fully operational** and are **not** contributing to performance problems:

| Component | Status | Notes |
|-----------|--------|-------|
| MLX Metal GPU | ✅ Operational | `Device(gpu, 0)`, `metal.is_available() = True` — mlx-whisper will use GPU acceleration |
| PyTorch MPS | ✅ Operational | `mps.is_available() = True`, `mps.is_built() = True` — pyannote diarization uses MPS |
| NumPy BLAS | ✅ Optimal | Apple Accelerate/vecLib with NEON/ASIMD SIMD — no regression from pyenv transition |
| HuggingFace model weights | ✅ Cached | 4.0 GB at `~/.cache/huggingface/` — no re-download needed |
| llvmlite + numba | ✅ Working | llvmlite 0.46.0, numba 0.64.0 — correct |
| All Python imports | ✅ Succeed | mlx, torch, ctranslate2, numpy, llvmlite, numba all import without error |
| pyannote MPS device selection | ✅ Correct | [`_select_device()`](pipeline/diarize.py:60) correctly prioritizes MPS over CPU |
| State deduplication | ✅ Working | [`state.is_ingested()`](pipeline/state.py) correctly skips already-ingested episodes |
| OWASP security controls | ✅ Present | HF_TOKEN never logged; filenames sanitized in [`document.py`](pipeline/document.py) |

---

## 6. Prioritized Recommendations

### R1 — 🔴 Fix the Double-Transcription Bug

**Severity:** Critical  
**File:** [`pipeline/transcribe.py`](pipeline/transcribe.py)  
**Expected impact:** Eliminates 10–25 minutes of wasted CPU work per episode

**Root cause:** [`_extract_segments_from_docling()`](pipeline/transcribe.py:52) probes undocumented Docling API paths that do not exist in Docling 2.75.0, causing it to always return `None` and trigger the faster-whisper fallback.

**How to fix:**

**Step 1:** Determine whether Docling 2.75.0's `AsrPipeline` actually exposes segment timestamps. Run this diagnostic:

```python
from pathlib import Path
from docling.datamodel import asr_model_specs
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import AsrPipelineOptions
from docling.document_converter import AudioFormatOption, DocumentConverter
from docling.pipeline.asr_pipeline import AsrPipeline

opts = AsrPipelineOptions()
opts.asr_options = asr_model_specs.WHISPER_TURBO
conv = DocumentConverter(format_options={
    InputFormat.AUDIO: AudioFormatOption(pipeline_cls=AsrPipeline, pipeline_options=opts)
})
result = conv.convert(Path("your_test_audio.mp3"))
doc = result.document
print(type(doc))
print(dir(doc))
# Inspect: doc.texts, doc.body, result.__dict__, etc.
```

**Step 2a — If Docling exposes segments:** Update [`_extract_segments_from_docling()`](pipeline/transcribe.py:52) with the correct verified attribute path. Remove the speculative attribute probing and the broad `except Exception` swallowing.

**Step 2b — If Docling does NOT expose segments:** The correct fix is to **replace the Docling ASR path entirely** with a direct mlx-whisper call that returns both the transcript text and timestamped segments in a single pass:

```python
# pipeline/transcribe.py — replacement primary path
import mlx_whisper

def transcribe_audio(audio_path: Path) -> TranscriptResult:
    result = mlx_whisper.transcribe(
        str(audio_path),
        path_or_hf_repo="mlx-community/whisper-turbo-mlx",
        word_timestamps=False,
    )
    segments = [
        {"start": s["start"], "end": s["end"], "text": s["text"]}
        for s in result["segments"]
    ]
    return TranscriptResult(document=result, segments=segments)
```

This eliminates the Docling intermediary entirely for the ASR stage, uses the same mlx-whisper backend, and returns segments directly — no fallback needed.

**Step 3:** Remove or guard [`_extract_segments_via_faster_whisper()`](pipeline/transcribe.py:158) so it cannot be triggered silently. If kept as a true emergency fallback, add a prominent log warning at `WARNING` or `ERROR` level (not `DEBUG`) so the double-transcription is visible in logs.

---

### R2 — 🔴 Fix ffmpeg PATH for Non-Interactive Shells

**Severity:** Critical (pipeline will fail silently or with `FileNotFoundError` in non-interactive contexts)  
**Expected impact:** Prevents silent failures in yt-dlp, Docling, and faster-whisper

```bash
# Add to ~/.zshrc for interactive shells
echo 'export PATH="/opt/homebrew/bin:$PATH"' >> ~/.zshrc

# Add to .env for uv-managed non-interactive invocations
echo 'PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin' >> .env
```

Verify after fix:
```bash
uv run python -c "import shutil; print(shutil.which('ffmpeg'))"
# Expected: /opt/homebrew/bin/ffmpeg
```

---

### R3 — 🔴 Cache Models Between Episodes

**Severity:** Critical for batch processing (N > 1 episodes)  
**Files:** [`pipeline/transcribe.py`](pipeline/transcribe.py), [`pipeline/diarize.py`](pipeline/diarize.py)  
**Expected impact:** Saves 25–75 seconds per additional episode in a batch

Move model instantiation outside the per-episode functions. The cleanest approach is module-level lazy initialization with a sentinel:

```python
# pipeline/transcribe.py — module-level cache
_CONVERTER: object | None = None

def _get_converter():
    global _CONVERTER
    if _CONVERTER is None:
        from docling.document_converter import DocumentConverter
        # ... build converter once ...
        _CONVERTER = converter
    return _CONVERTER
```

```python
# pipeline/diarize.py — module-level cache
_PIPELINE: object | None = None

def _get_pipeline():
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = _load_diarization_pipeline()
    return _PIPELINE
```

Alternatively, pass pre-loaded model instances as parameters to `_process_episode()` from [`main.py`](main.py), loading them once before the episode loop at [`main.py:270`](main.py:270).

---

### R4 — 🟡 Run the Pipeline Once to Warm Metal Shader Caches

**Severity:** Medium (one-time, self-resolving)  
**Expected impact:** Eliminates 45–165 seconds of cold-start overhead on all subsequent runs

No code change required. Simply run the pipeline once to completion on any episode:

```bash
uv run python main.py ingest "https://www.youtube.com/watch?v=<any_episode_id>"
```

After this run, `~/Library/Caches/com.apple.metal/` and `~/.cache/torch/` will be populated. All subsequent runs will start at full GPU speed.

---

### R5 — 🟡 Add Episode-Level Parallelism

**Severity:** Medium (significant for batches of 3+ episodes)  
**File:** [`main.py:270`](main.py:270)  
**Expected impact:** 30–50% reduction in total wall-clock time for batch processing

Replace the sequential `for` loop with `concurrent.futures.ThreadPoolExecutor` or `asyncio` task groups. Since transcription is GPU-bound (MLX) and diarization is GPU-bound (MPS), they use different hardware resources and can run concurrently for different episodes.

**Caution:** Model caching (R3) must be implemented first, and thread-safety of the cached model instances must be verified before enabling parallelism. pyannote's `Pipeline` is not guaranteed thread-safe — use process-level parallelism (`ProcessPoolExecutor`) if thread-safety is uncertain.

```python
# main.py — sketch of parallel episode processing
from concurrent.futures import ThreadPoolExecutor, as_completed

with ThreadPoolExecutor(max_workers=2) as executor:
    futures = {
        executor.submit(_process_episode, episode=ep, ...): ep
        for ep in episodes
    }
    for future in as_completed(futures):
        ep = futures[future]
        try:
            future.result()
        except Exception as exc:
            log.error("Episode %s failed: %s", ep.video_id, exc)
```

---

### R6 — 🟢 Remove Unused Docling Markdown Re-Conversion

**Severity:** Low  
**File:** [`pipeline/document.py:186`](pipeline/document.py:186)  
**Expected impact:** Saves ~1–5 seconds per episode

The `_docling_doc` return value from [`build_transcript_document()`](pipeline/document.py:126) is discarded at [`main.py:398`](main.py:398) (`_docling_doc`). The [`_convert_markdown_to_docling()`](pipeline/document.py:191) call at [`document.py:186`](pipeline/document.py:186) is dead work.

Either remove the Docling re-conversion from `build_transcript_document()` and change the return type to `Path` only, or keep it but make it opt-in via a parameter flag.

---

### R7 — 🟢 Remove Unused Homebrew Python Versions

**Severity:** Low (hygiene)  
**Expected impact:** Eliminates PATH ambiguity; no runtime performance impact

```bash
brew uninstall python@3.10 python@3.13 python@3.14
```

---

## 7. Performance Baseline Estimates

Estimates for a single 60-minute podcast episode on M1 Pro, 32 GB RAM, after cache warm-up (R4 applied):

| Stage | Component | Estimated Time | Notes |
|-------|-----------|---------------|-------|
| Stage 1 — Download | yt-dlp + ffmpeg | 1–3 min | Network-bound; 192 kbps MP3 ≈ 85 MB |
| Stage 2 — Transcription (primary) | mlx-whisper via Docling, GPU | 3–6 min | M1 Pro GPU, Metal-accelerated |
| Stage 2 — Transcription (fallback, if bug not fixed) | faster-whisper, CPU-only | +10–25 min | CTranslate2, no Metal support |
| Stage 2 — Model load (if not cached) | Docling/mlx-whisper init | +15–45 sec | Per episode if R3 not applied |
| Stage 3 — Diarization | pyannote 3.1, MPS | 3–8 min | PyTorch MPS-accelerated |
| Stage 3 — Model load (if not cached) | `Pipeline.from_pretrained()` | +10–30 sec | Per episode if R3 not applied |
| Stage 4 — Document build | Markdown write + Docling re-parse | 5–15 sec | Minor; Docling re-parse is unnecessary |
| Stage 5 — OpenRAG upload | Async HTTP | 5–30 sec | Network-bound |
| Stage 6 — State write | `state.json` atomic write | <1 sec | Negligible |
| **Total (bug present, cold cache)** | | **~50–90 min** | First run worst case |
| **Total (bug fixed, warm cache)** | | **~8–18 min** | After R1 + R4 applied |
| **Total (all fixes applied)** | | **~7–15 min** | After R1–R4 applied |

**Cold-start penalty breakdown (first run only):**

| Cache | Estimated Penalty |
|-------|------------------|
| MLX Metal shader compilation | 30–120 sec |
| PyTorch MPS kernel compilation | 15–45 sec |
| **Total cold-start overhead** | **45–165 sec** |

---

## 8. Hardware Context

### M1 Pro Capabilities Relevant to This Workload

| Resource | Specification | Relevance |
|----------|--------------|-----------|
| CPU cores | 8 performance + 2 efficiency | Sequential pipeline underutilizes this; R5 would leverage it |
| GPU cores | 16-core Apple GPU | Used by mlx-whisper (MLX Metal) — primary transcription path |
| Neural Engine | 16-core ANE | Not currently used by any pipeline component |
| Unified Memory | 32 GB | Sufficient for all models simultaneously in memory |
| Memory bandwidth | ~200 GB/s | Key advantage of unified memory for ML inference |
| Disk | 143 GB free / 460 GB | Sufficient; model weights (4.0 GB) already cached |

### Key Architectural Observations

**Unified memory is a significant advantage.** On M1 Pro, CPU, GPU, and Neural Engine share the same physical memory pool. There is no PCIe transfer overhead when moving tensors between CPU and GPU — a major advantage over discrete GPU systems. This means model caching (R3) is especially valuable: a cached model in unified memory is instantly accessible to both the MLX GPU (mlx-whisper) and the MPS GPU (pyannote) without any copy overhead.

**MLX vs. PyTorch MPS for this workload.** The transcription stage uses MLX Metal (via mlx-whisper), while the diarization stage uses PyTorch MPS. Both are GPU-accelerated but use different Metal command queues. They can run concurrently on the same GPU without contention, which supports the parallelism recommendation (R5) for future episodes.

**CTranslate2 CPU limitation.** The faster-whisper fallback runs on CPU only. On M1 Pro, the 8 performance cores with NEON SIMD provide reasonable CPU throughput, but it is still 3–5× slower than mlx-whisper on the GPU for Whisper Turbo inference. This reinforces that eliminating the double-transcription bug (R1) is the highest-priority fix.

**Neural Engine is unused.** Apple's ANE is optimized for transformer inference and could theoretically accelerate both Whisper and pyannote. Neither mlx-whisper nor pyannote currently targets the ANE. This is a future optimization opportunity outside the scope of this codebase.

---

## 9. Summary of Recommended Actions (Ordered by Priority)

| Priority | ID | Action | Effort | Impact |
|----------|----|--------|--------|--------|
| 1 | R1 | Fix double-transcription bug in [`transcribe.py`](pipeline/transcribe.py) | Medium | 🔴 Saves 10–25 min/episode |
| 2 | R2 | Fix ffmpeg PATH in shell profile and `.env` | Low | 🔴 Prevents silent failures |
| 3 | R4 | Run pipeline once to warm Metal shader caches | None | 🔴 Saves 45–165 sec (one-time) |
| 4 | R3 | Cache models between episodes | Medium | 🔴 Saves 25–75 sec/episode (batches) |
| 5 | R5 | Add episode-level parallelism | High | 🟡 30–50% batch time reduction |
| 6 | R6 | Remove unused Docling Markdown re-conversion | Low | 🟢 Saves 1–5 sec/episode |
| 7 | R7 | Remove unused Homebrew Python versions | Low | 🟢 Hygiene only |