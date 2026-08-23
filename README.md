# Subzero

Universal subtitle and audio AI toolkit: clean SDH, auto-sync, shift, convert, extract, and translate with zero setup.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-97%20passed-brightgreen.svg)]()
[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-Donate-yellow.svg?style=flat&logo=buy-me-a-coffee)](https://buymeacoffee.com/yurirod)

**Subzero** solves the most annoying subtitle problems in one fast, zero-dependency CLI: out-of-sync audio, deafening sound cues, broken dual-speaker formatting, missing translations, and embedded video subtitle extraction.

---

### Before and After Subzero

```srt
1
00:00:01,000 --> 00:00:03,000
[tense dramatic music playing]

2
00:00:03,500 --> 00:00:05,000
- MITCH: Yes, sir. - MAN: So I have a question.
```

**After `subzero fix show.srt`:**

```srt
1
00:00:03,500 --> 00:00:05,000
- Yes, sir.
- So I have a question.
```

---

## Quick Install

```console
# via pip
pip install subzero-cli

# via uv
uv tool install subzero-cli

# or run directly with uvx without installing
uvx subzero-cli --help
```

*Requirements:* Python 3.9+ with zero required Python dependencies. `ffmpeg` is optional and only used for container extraction and audio delay synchronization.

---

## Interactive Terminal Menu

If you prefer an interactive interface over command-line flags:

```console
subzero menu
```

A clean, zero-dependency terminal UI prompts for paths, options, format conversions, time shifting, extraction, and automated sweeps.

---

## Command Reference

### 1. Strip SDH and Clean Formatting (`subzero fix`)

Strips sound effects, speaker tags, HTML tags, ASS override codes, and repairs dual-speaker dialog spacing:

```console
subzero fix movie.srt                              # Clean a single file in place
subzero fix /media/library --backup ~/backup       # Batch clean a folder tree with backup
subzero fix movie.srt --keep-music                 # Preserve musical notes
```

### 2. Auto-Sync to Video Speech (`subzero sync`)

Detects container audio start delays and synchronizes subtitles to match the video:

```console
subzero sync movie.mkv movie.srt                   # Auto-align subtitle to video
subzero sync movie.mp4 movie.srt -o movie.synced.srt
```

### 3. Shift Timecodes (`subzero shift`)

Offsets timestamps forward or backward with millisecond accuracy:

```console
subzero shift movie.srt --seconds +1.5             # Advance by 1.5 seconds
subzero shift ./subs --seconds -0.800              # Shift entire directory by -800ms
```

### 4. Convert Subtitle Formats (`subzero convert`)

Lossless conversion between SRT, WebVTT, ASS, SSA, and SUB formats:

```console
subzero convert show.vtt --to srt                  # WebVTT -> SRT
subzero convert show.ass --to srt --fix            # ASS -> SRT + SDH cleanup
subzero convert ./subs --to vtt                    # Batch convert folder
```

### 5. Extract Soft Subtitles from Video (`subzero extract`)

Pulls embedded subtitle streams from MP4, MKV, MOV, WEBM, AVI, MLV, and other containers:

```console
subzero streams movie.mkv                          # List all embedded subtitle streams
subzero extract movie.mkv                          # Extract first text track to .srt
subzero extract movie.mkv --all --language eng por # Extract specific language streams
subzero extract ./library --format vtt --fix       # Extract and clean all library files
```

### 6. Local LLM Translation (`subzero translate`)

Translates subtitle files into other languages using local Ollama models (e.g. Gemma 3 12B, Qwen) with cue-preserving batching:

```console
subzero translate show.srt --to pt-BR              # Translate English -> Brazilian Portuguese
subzero translate show.srt --to es --model gemma3:12b
```

### 7. OpenSubtitles MovieHash (`subzero moviehash`)

Calculates the 64-bit file hash used by OpenSubtitles to identify video and subtitle releases:

```console
subzero moviehash movie.mkv
```

### 8. Background Library Watcher (`subzero watch`)

Runs as a background daemon to automatically clean and prep newly arrived subtitles:

```console
subzero watch /media/movies --interval 300 --backup ~/sub-backup
```

---

## Python Library API

Subzero is also fully usable as a clean Python library:

```python
from subzero import (
    Options,
    fix_text,
    fix_file,
    convert_text,
    convert_file,
    shift_timestamps,
    shift_file,
    auto_sync_file,
    probe_audio_delay,
    moviehash,
    extract_from_video,
    list_subtitle_streams,
)

# 1. Clean SDH from subtitle text
cleaned = fix_text(raw_srt)
print(f"Cues: {cleaned.cues}, Dropped: {cleaned.dropped}")

# 2. Shift timecodes
shifted_text, count = shift_timestamps(raw_srt, delta_seconds=+1.5)

# 3. Convert formats
vtt_text = convert_text(raw_srt, target="vtt", source="srt").text

# 4. Extract from video
extract_from_video("movie.mkv", fmt="srt", languages=("eng",), fix=fix_file)
```

---

## Support and Sponsorship

If Subzero saved you time, fixed your family movie night, or improved your media setup, consider buying me a coffee:

<a href="https://buymeacoffee.com/yurirod"><img src="https://img.buymeacoffee.com/button-api/?text=Buy%20me%20a%20coffee&emoji=☕&slug=yurirod&button_colour=FFDD00&font_colour=000000&font_family=Inter&outline_colour=000000&coffee_colour=ffffff" /></a>

---

## Contributing

```console
git clone https://github.com/yuri-rod/subzero.git
cd subzero
pip install -e ".[dev]"
pytest
```

---

## License

MIT License (c) 2026 Yuri Barreira
