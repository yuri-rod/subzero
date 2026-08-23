# Subzero

Clean subtitles, adjust timestamps, convert formats, extract embedded tracks, and translate through a configured provider.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-97%20passed-brightgreen.svg)]()
[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-Donate-yellow.svg?style=flat&logo=buy-me-a-coffee)](https://buymeacoffee.com/yurirod)

Subzero removes sound descriptions, repairs collapsed dialogue turns, adjusts timestamps, and extracts or translates subtitle files.

---

### Before and after cleanup

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

## Supported formats

| Category | Supported Formats | Operations |
| :--- | :--- | :--- |
| Subtitle files | SubRip (`.srt`), WebVTT (`.vtt`), ASS (`.ass`), SSA (`.ssa`) | Convert between formats. Use SRT for cleanup, shifting, translation, and synchronization. |
| **Video Containers** | Matroska (`.mkv`), MP4 (`.mp4`, `.m4v`), QuickTime (`.mov`), WebM (`.webm`), AVI (`.avi`), MLV (`.mlv`), Transport Stream (`.ts`) | Inspect, Extract Soft Subs, Stream Listing, Audio Probe |
| **Encodings** | UTF-8, UTF-8-BOM, Windows-1252 (CP1252), ISO-8859-1 (Latin-1) | Auto-detect, Decode, Normalize to UTF-8 |

---

## Installation

```console
# via pip
pip install subzero-cli

# via uv
uv tool install subzero-cli

# or run directly with uvx without installing
uvx subzero-cli --help
```

*Requirements:* Python 3.9+ with zero mandatory third-party Python packages. `ffmpeg` is optional and only used for container extraction and audio delay synchronization.

---

## Interactive terminal menu

If you prefer an interactive terminal UI over command-line arguments:

```console
subzero menu
```

The terminal menu prompts for paths and options for cleanup, conversion, timestamp shifting, extraction, and directory watching.

---

## Commands

### 1. Clean SDH and repair dialogue (`subzero fix`)

Strips hearing-impaired markup, speaker labels, HTML tags, ASS override codes, and repairs dual-speaker dialog spacing:

```console
subzero fix movie.srt                              # Clean a single file in place
subzero fix /media/library --backup ~/backup       # Batch clean a folder tree with backup
subzero fix movie.srt --keep-music                 # Preserve musical notes
subzero fix movie.srt --max-line 38                # Custom max line length
subzero fix movie.srt --dry-run                    # Preview changes without modifying files
```

### 2. Apply container timing offsets (`subzero sync`)

Read the container audio/video start-time difference with `ffprobe` and apply it as a subtitle offset. This does not verify speech alignment:

```console
subzero sync movie.mkv movie.srt                   # Auto-align subtitle to video
subzero sync movie.mp4 movie.srt -o movie.synced.srt
```

### 3. Shift timestamps (`subzero shift`)

Offsets timestamps forward or backward with millisecond accuracy and zero-floor protection:

```console
subzero shift movie.srt --seconds +1.5             # Advance by 1.5 seconds
subzero shift ./subs --seconds -0.800              # Shift entire directory by -800ms
subzero shift movie.srt -s +2.0 -o shifted.srt     # Save to specific output path
```

### 4. Convert formats (`subzero convert`)

Convert subtitle text and timing between SRT, WebVTT, ASS, and SSA. Format-specific styling can be lost:

```console
subzero convert show.vtt --to srt                  # WebVTT -> SRT
subzero convert show.ass --to srt --fix            # ASS -> SRT + SDH cleanup
subzero convert ./subs --to vtt                    # Batch convert folder
```

### 5. Extract embedded subtitles (`subzero extract`)

Pulls embedded subtitle streams from MP4, MKV, MOV, WEBM, AVI, MLV, and other containers:

```console
subzero streams movie.mkv                          # List all embedded subtitle streams
subzero extract movie.mkv                          # Extract first text track to .srt
subzero extract movie.mkv --all --language eng por # Extract specific language streams
subzero extract ./library --format vtt --fix       # Extract and clean all library files
```

### 6. Translate subtitles (`subzero translate`)

Translates subtitle files into other languages using local Ollama models (e.g. Gemma 3 12B, Qwen 2.5) with cue-preserving batching and retry resilience:

```console
subzero translate show.srt --to pt-BR              # Translate English -> Brazilian Portuguese
subzero translate show.srt --to es --model gemma3:12b
subzero translate show.srt --to fr --url http://127.0.0.1:11434
```

### 7. Calculate an OpenSubtitles hash (`subzero moviehash`)

Calculates the 64-bit file hash used by OpenSubtitles to identify video and subtitle releases:

```console
subzero moviehash movie.mkv
```

### 8. Watch a subtitle directory (`subzero watch`)

Poll a local media directory and clean new or changed subtitles:

```console
subzero watch /media/movies --interval 300 --backup ~/sub-backup
```

---

## Options

| Option | Flag | Description | Default |
| :--- | :--- | :--- | :--- |
| **Languages** | `--lang` | Language codes for role label matching (`en`, `pt`, `es`, `fr`, `de`, `it`) | `en pt` |
| **Max Line** | `--max-line` | Target characters per line before breaking | `42` |
| **Keep Brackets** | `--keep-brackets` | Preserve bracketed cues like `[Applause]` | `False` |
| **Keep Parens** | `--keep-parens` | Preserve parenthetical cues like `(Sighs)` | `False` |
| **Keep Music** | `--keep-music` | Preserve musical symbols like `♪` | `False` |
| **Keep Labels** | `--keep-labels` | Preserve speaker labels like `JOHN:` | `False` |
| **Rewrap All** | `--rewrap-all` | Re-break every cue, even already broken lines | `False` |
| **Backup** | `--backup DIR` | Directory to copy original files before modification | `None` |
| **Dry Run** | `--dry-run` | Report what would change without modifying files | `False` |

---

## Cleanup behavior

1. **Existing Human Line Breaks are Respected:**
   Release subtitles are often timed and broken by human translators for pacing. Subzero preserves intact line breaks and only reflows cues that exceed character limits or glue multiple speakers onto a single line. Use `--rewrap-all` to override.
2. **Conservative Speaker Label Detection:**
   ALL-CAPS before a colon is treated as a speaker tag (`OFFICER:`). Lower-case words before a colon are validated against multilingual role dictionaries (`man:`, `mulher:`, `doctor:`) to prevent corrupting natural dialogue like `Time: 10:00`.
3. **Dialogue Dash Repair:**
   When an SDH cue containing one speaker is stripped from a two-speaker line, the remaining single line has its dangling leading dash cleanly formatted.

---

## Python library API

Call the subtitle functions directly from Python:

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
cleaned = fix_text(raw_srt, Options(max_line=40))
print(f"Cues: {cleaned.cues}, Dropped: {cleaned.dropped}")

# 2. Shift timecodes
shifted_text, count = shift_timestamps(raw_srt, delta_seconds=+1.5)

# 3. Convert formats
vtt_text = convert_text(raw_srt, target="vtt", source="srt").text

# 4. Extract from video
extract_from_video("movie.mkv", fmt="srt", languages=("eng",), fix=fix_file)
```

---

## Support

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
