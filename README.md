# Subzero

```
  ███████╗██╗   ██╗██████╗ ███████╗███████╗██████╗  ██████╗ 
  ██╔════╝██║   ██║██╔══██╗╚══███╔╝██╔════╝██╔══██╗██╔═══██╗
  ███████╗██║   ██║██████╔╝  ███╔╝ █████╗  ██████╔╝██║   ██║
  ╚════██║██║   ██║██╔══██╗ ███╔╝  ██╔══╝  ██╔══██╗██║   ██║
  ███████║╚██████╔╝██████╔╝███████╗███████╗██║  ██║╚██████╔╝
  ╚══════╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝╚═╝  ╚═╝ ╚═════╝ 
```

**The universal subtitle and audio AI toolkit.**  
*Clean SDH, auto-sync, shift, convert, extract, and translate with zero setup.*

[![Version](https://img.shields.io/badge/version-1.2.0-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python: 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![Dependencies: Zero](https://img.shields.io/badge/dependencies-zero-brightgreen.svg)]()
[![Platform: Linux | macOS | Windows](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey.svg)]()
[![CI](https://github.com/yuri-rod/subzero/actions/workflows/ci.yml/badge.svg)](https://github.com/yuri-rod/subzero/actions/workflows/ci.yml)
[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-Donate-yellow.svg?style=flat&logo=buy-me-a-coffee)](https://buymeacoffee.com/yurirod)

---

## Why Subzero?

Most subtitle workflows are fragmented between slow Python 2 legacy scripts, heavy GUIs, and cloud subscription APIs. **Subzero** delivers a single, high-performance CLI and Python library that handles the entire subtitle lifecycle:

* **Zero Dependencies:** Pure Python standard library core. Starts in under 20ms with negligible RAM usage.
* **Smart SDH Removal:** Strips sound cues (`[LAUGHTER]`, `(SIGHS)`, `♪`), speaker labels, and HTML/ASS tags without corrupting real dialogue.
* **Dual-Speaker Repair:** Automatically fixes collapsed dialogue lines and normalizes speaker dashes.
* **Timing verification:** Checks subtitle activity against local speech detection across the movie, with explicit pass, reject and inconclusive results.
* **Direct Video Extraction:** Pulls soft subtitle tracks from Matroska (`.mkv`), MP4, MOV, WebM, and AVI files.
* **Local AI Translation:** Translates entire series into target languages using local Ollama LLMs with cue-preserving batching.
* **Universal Format Engine:** Losslessly converts between SRT, WebVTT, ASS, SSA, and MicroDVD formats.
* **Self-Hosted Ready:** Runs as a standalone background watcher for Jellyfin, Plex, Sonarr, and Radarr libraries.

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

## Comparison Matrix

| Feature | Subzero | SubCleaner | Bazarr | Raw ffmpeg |
| :--- | :---: | :---: | :---: | :---: |
| **Zero Runtime Dependencies** | **Yes** (pure Python) | No | No (heavy stack) | Yes (binary only) |
| **SDH Removal + Line Repair** | **Yes** | Yes (regex only) | Basic | No |
| **Container Audio Delay Sync** | **Yes** | No | No | Manual scripting |
| **Local LLM Translation (Ollama)** | **Yes** | No | API keys only | No |
| **Direct Container Extraction** | **Yes** | No | Yes | Complex syntax |
| **Format Conversion (SRT/VTT/ASS)** | **Yes** | No | No | Basic |
| **Interactive Terminal Menu** | **Yes** | No | Web UI only | No |
| **OpenSubtitles MovieHash** | **Yes** | No | Internal only | No |
| **Startup Latency** | **<20ms** | ~200ms | Multi-second | <50ms |

---

## Quick Install

### Using `pip`
```console
pip install subzero-cli
```

### Using `uv` (Recommended)
```console
uv tool install subzero-cli
```

### Run Directly with `uvx` (No installation needed)
```console
uvx subzero-cli menu
```

*Requirements:* Python 3.9+ on Linux, macOS, or Windows. `ffmpeg` is optional and only required when extracting tracks from video files or probing audio stream delay.

---

## Interactive Terminal Menu

Run `subzero menu` to launch an interactive terminal interface:

```
============================================================
  subzero 1.0.0: interactive menu
============================================================
  langs: en,pt   max-line: 42   extract-> srt
  flags: defaults
------------------------------------------------------------
  1) Fix SDH in subtitle files
  2) Check subtitle files (report only)
  3) Extract subtitles from video (mp4/mkv/mlv/...)
  4) Extract from video + fix SDH
  5) Convert subtitle format (srt/vtt/ass)
  6) Convert + fix SDH
  7) Shift subtitle timestamps (+/- seconds)
  8) Calculate OpenSubtitles MovieHash
  9) List subtitle streams in a video
 10) Watch a directory for new subtitles
 11) Configure options
  0) Exit
------------------------------------------------------------
Choice [0]:
```

---

## Command Reference and Recipes

### 1. Cleaning SDH and Reflowing Dialogue (`subzero fix`)

Strips sound effects, speaker tags, HTML tags, ASS override codes, and normalizes smart quotes:

```console
# Clean a single subtitle file in place
subzero fix movie.srt

# Clean an entire movie or TV library, preserving originals in a backup folder
subzero fix /media/library --backup ~/subs-backup

# Preview changes without modifying files (dry run)
subzero fix show.srt --dry-run -v

# Keep musical symbols while stripping other sound effects
subzero fix concert.srt --keep-music
```

### 2. Checking timing and applying container delay

Install the optional speech dependencies for timing verification:

```console
pip install 'subzero-cli[sync] @ git+https://github.com/yuri-rod/subzero.git@v1.2.0'
subzero verify-sync movie.mkv movie.pt-BR.srt
```

`verify-sync` emits JSON with per-window offset, correlation and confidence.
Exit codes are `0` for pass, `2` for rejection, `3` for inconclusive evidence and
`4` for an operational error. It never changes the subtitle. FFmpeg extracts audio
locally, and speech detection runs locally; no audio is uploaded. References are
cached under `~/.cache/subzero/references`, keyed by video identity and validator
version. Use `--cache DIR` to choose another location.

Verification examines overlapping windows across the dialogue span. The default
subtitle tolerance is 500 ms. Sparse speech, ambiguous correlations and isolated
mismatches remain inconclusive. Repeated timing mismatches reject the subtitle.
Embedded dialogue tracks are used only when local audio evidence supports them;
otherwise speech activity supplies the reference. Forced and commentary tracks
are excluded. Timing verification does not establish translation accuracy or
prove that every individual cue is correct. Videos from two minutes to four hours
are supported by the audio extraction path.

The Python API exposes `subzero.timing.correction` for conservative constant-delay
or linear framerate correction. Fit and held-out windows must agree, and consumers
must verify the corrected file again before accepting it. Different cuts and
irregular drift do not receive automatic piecewise repairs.

The older `sync` command only measures container audio/video start-time skew.
It does not detect release mismatches or certify dialogue synchronization:

```console
# Apply the container audio delay
subzero sync movie.mkv movie.srt

# Output the aligned subtitle to a new file
subzero sync movie.mp4 movie.srt -o movie.synced.srt
```

### 3. Shifting Timestamps (`subzero shift`)

Offsets timecodes forward or backward with millisecond accuracy and zero-floor protection:

```console
# Delay subtitles by 1.5 seconds
subzero shift episode.srt --seconds +1.5

# Convert framerate timing (e.g. PAL 25fps to Film 23.976fps)
subzero shift movie.srt --from-fps 25 --to-fps 23.976

# Advance subtitles by 800 milliseconds across a whole folder
subzero shift ./subs --seconds -0.800 --backup ./backup
```

### 4. Format Conversion (`subzero convert`)

Losslessly converts between SRT, WebVTT, ASS, SSA, and MicroDVD formats:

```console
# Convert WebVTT to SubRip
subzero convert video.vtt --to srt

# Convert ASS with complex styling into clean SRT and strip SDH cues in one step
subzero convert anime.ass --to srt --fix

# Batch convert a directory to WebVTT for browser streaming
subzero convert ./library --to vtt --pattern "*.srt"
```

### 5. Extracting Soft Subtitles from Video Containers (`subzero extract`)

Extracts embedded text subtitle streams without memorizing ffmpeg stream mapping arguments:

```console
# Inspect available streams in a container
subzero streams movie.mkv

# Extract the default text track to a .srt file next to the video
subzero extract movie.mkv

# Extract all English and Portuguese subtitle tracks from a series folder and clean SDH
subzero extract /media/series --all --language eng por --format srt --fix
```

### 6. AI Translation (`subzero translate`)

Translates subtitles into target languages using local Ollama LLMs or cloud OpenAI-compatible APIs with cue-preserving batching:

```console
# Translate English subtitle to Brazilian Portuguese using local Ollama (Gemma 3 12B)
subzero translate episode.srt --to pt-BR

# Translate using OpenAI or Groq / DeepSeek / OpenRouter
subzero translate movie.srt --to es --provider groq --api-key "$GROQ_API_KEY"

# Translate using a custom Ollama host or model
subzero translate movie.srt --to es --model qwen2.5-coder:7b --url http://192.168.1.50:11434
```

### 7. Bilingual Subtitle Merge (`subzero merge`)

Merges two language tracks into a single bilingual subtitle file (ideal for language learners and dual-audio streaming):

```console
# Merge English and Portuguese subtitles
subzero merge movie.en.srt movie.pt.srt -o movie.dual.srt

# Highlight secondary language with custom color
subzero merge anime.jp.srt anime.en.srt -o anime.bilingual.srt --color "#ffff00"
```

### 8. OpenSubtitles MovieHash (`subzero moviehash`)

Calculates the 64-bit file hash used by OpenSubtitles to identify video releases:

```console
subzero moviehash movie.mkv
```

### 9. Background Media Server Daemon (`subzero watch`)

Runs as a lightweight daemon to automatically clean and prep newly arrived subtitles:

```console
subzero watch /media/library --interval 300 --backup ~/sub-backup
```

---

## Integration with Media Servers

### Sonarr / Radarr Custom Script
Add a Custom Script hook in Sonarr/Radarr under **Settings > Connect > Custom Script**:
```bash
#!/usr/bin/env bash
# Triggered on Download / Upgrade
if [ "$sonarr_eventtype" = "Download" ]; then
    subzero extract "$sonarr_episodefile_path" --fix
    subzero fix "$(dirname "$sonarr_episodefile_path")" --pattern "*.srt"
fi
```

### Systemd Service (Linux Home Server)
Create `/etc/systemd/system/subzero-watch.service`:
```ini
[Unit]
Description=Subzero Subtitle Watcher Daemon
After=network.target

[Service]
Type=simple
User=media
ExecStart=/usr/local/bin/subzero watch /media/library --interval 300
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

---

## Supported Formats Matrix

| Kind | Extension | Supported Operations |
| :--- | :--- | :--- |
| **SubRip** | `.srt` | Read, Clean, Convert, Shift, Extract, Translate, Sync |
| **WebVTT** | `.vtt`, `.webvtt` | Read, Clean, Convert, Shift, Extract, Translate |
| **Advanced SubStation Alpha** | `.ass` | Read, Clean, Convert to SRT, Extract |
| **SubStation Alpha** | `.ssa` | Read, Clean, Convert to SRT, Extract |
| **MicroDVD** | `.sub` | Read, Convert to SRT |
| **Video Containers** | `.mkv`, `.mp4`, `.mov`, `.webm`, `.avi`, `.mlv`, `.ts`, `.m2ts` | Stream Inspection, Soft Subtitle Extraction, Audio Probe |
| **Character Encodings** | UTF-8, UTF-8-BOM, CP1252, Latin-1, ISO-8859-1 | Automatic detection and decoding to UTF-8 |

---

## Options and Configuration

| Option | Argument | Description | Default |
| :--- | :--- | :--- | :--- |
| **Languages** | `--lang CODE...` | Language codes for role dictionary matching (`en`, `pt`, `es`, `fr`, `de`, `it`) | `en pt` |
| **Max Line Length** | `--max-line N` | Maximum characters per line before breaking dialogue | `42` |
| **Keep Brackets** | `--keep-brackets` | Keep bracketed cues such as `[Applause]` | `False` |
| **Keep Parens** | `--keep-parens` | Keep parenthetical cues such as `(Sighs)` | `False` |
| **Keep Music** | `--keep-music` | Keep musical symbols such as `♪` | `False` |
| **Keep Labels** | `--keep-labels` | Keep speaker labels such as `JOHN:` | `False` |
| **Rewrap All** | `--rewrap-all` | Re-break every cue, overriding intact human line breaks | `False` |
| **Backup Directory** | `--backup DIR` | Copy originals to a backup folder before writing changes | `None` |
| **Dry Run** | `--dry-run` | Report what would change without modifying files | `False` |

---

## Architecture and Engineering Decisions

1. **Human Line Breaks are Preserved:**
   Professional release subtitles are timed and broken by human editors for reading pace. Subzero respects intact human line breaks and only modifies cues that exceed character limits or glue multiple speakers onto a single line.
2. **Conservative Label Matching:**
   ALL-CAPS text before a colon is treated as a speaker tag (`OFFICER:`). Lower-case words before a colon are matched against a closed dictionary of role words per language (`man:`, `mulher:`, `doctor:`, `medico:`) to avoid eating valid dialogue like `Score: 10`.
3. **Dialogue Dash Repair:**
   When an SDH cue containing one speaker is stripped from a two-speaker exchange, the remaining line has its leading dash cleaned to maintain dialogue integrity.

---

## Python Library API

Subzero can be integrated directly into Python pipelines:

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

# Clean SDH from subtitle text in memory
result = fix_text(raw_srt_text, Options(max_line=40))
print(f"Cues: {result.cues}, Dropped: {result.dropped}, Rewrapped: {result.rewrapped}")

# Shift timecodes forward by 2.5 seconds
shifted_text, count = shift_timestamps(raw_srt_text, delta_seconds=+2.5)

# Convert between formats
vtt_result = convert_text(raw_srt_text, target="vtt", source="srt")

# Extract soft subtitles from video container
extract_from_video("movie.mkv", fmt="srt", languages=("eng",), fix=fix_file)
```

---

## Support and Sponsorship

If Subzero saved you time, improved your home media setup, or fixed your movie night, consider buying me a coffee:

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
