# Sample Files Directory

This directory is where you place audio/video files for testing the demo scripts.

## Supported Formats

The following audio and video formats are supported by Docling:

- **Audio:** `.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`
- **Video:** `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`

## Recommended Test Files

For best results when testing the demos:

- **Duration:** 10-30 seconds (quick transcription)
- **Content:** Clear speech with minimal background noise
- **Size:** Under 10 MB for faster processing

## Naming Convention

For convenience, name your primary test file `test.mp3` (or another extension). This allows you to run demos without specifying the full path:

```bash
# From the examples/ directory
python demo_docling.py sample_files/test.mp3
```

## Where to Get Test Files

### Option 1: Record Your Own
Use your phone or computer to record a short voice memo.

### Option 2: Download Free Audio
- [Freesound.org](https://freesound.org/) - Free sound effects and recordings
- [BBC Sound Effects](https://sound-effects.bbcrewind.co.uk/) - Public domain audio
- [YouTube Audio Library](https://www.youtube.com/audiolibrary) - Royalty-free music and sound effects

### Option 3: Extract from Video
If you have a video file, you can extract audio using ffmpeg:

```bash
ffmpeg -i video.mp4 -vn -acodec libmp3lame audio.mp3
```

## Example Files

Here are some example files you might add:

```
sample_files/
├── test.mp3              # Your primary test file
├── podcast_clip.mp3      # Short podcast excerpt
├── interview.wav         # Interview recording
└── presentation.mp4      # Video presentation
```

## Important Notes

- **Privacy:** Don't commit sensitive audio files to version control
- **Copyright:** Only use audio you have rights to use
- **Size:** Keep files small for faster testing
- **Quality:** Higher quality audio = better transcription accuracy

## Testing the Demos

Once you have a test file, try the demos in order:

1. **Transcription only:**
   ```bash
   python demo_docling.py sample_files/test.mp3
   ```

2. **Upload only:**
   ```bash
   python demo_openrag.py sample_files/transcript.md
   ```

3. **Full pipeline:**
   ```bash
   python demo_full_pipeline.py sample_files/test.mp3
   ```

Happy testing! 🎵