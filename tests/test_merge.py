from pathlib import Path
from subzero.convert import Cue
from subzero.merge import merge_cues, merge_files

SAMPLE_EN = """1
00:00:01,000 --> 00:00:03,000
Hello world!

2
00:00:04,000 --> 00:00:06,000
How are you?
"""

SAMPLE_PT = """1
00:00:01,050 --> 00:00:03,000
Olá mundo!

2
00:00:04,000 --> 00:00:06,100
Como você está?
"""


def test_merge_cues_basic():
    cues_en = [
        Cue("00:00:01,000", "00:00:03,000", "Hello world!"),
        Cue("00:00:04,000", "00:00:06,000", "How are you?"),
    ]
    cues_pt = [
        Cue("00:00:01,050", "00:00:03,000", "Olá mundo!"),
        Cue("00:00:04,000", "00:00:06,100", "Como você está?"),
    ]
    merged = merge_cues(cues_en, cues_pt)
    assert len(merged) == 2
    assert "Hello world!\nOlá mundo!" == merged[0].text
    assert "How are you?\nComo você está?" == merged[1].text


def test_merge_cues_with_color():
    cues_en = [Cue("00:00:01,000", "00:00:03,000", "Hello")]
    cues_pt = [Cue("00:00:01,000", "00:00:03,000", "Olá")]
    merged = merge_cues(cues_en, cues_pt, secondary_color="#ffff00")
    assert len(merged) == 1
    assert 'Hello\n<font color="#ffff00">Olá</font>' == merged[0].text


def test_merge_files(tmp_path: Path):
    f1 = tmp_path / "movie.en.srt"
    f2 = tmp_path / "movie.pt.srt"
    f1.write_text(SAMPLE_EN, encoding="utf-8")
    f2.write_text(SAMPLE_PT, encoding="utf-8")

    out, count = merge_files(f1, f2)
    assert count == 2
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "Olá mundo!" in content
    assert "Hello world!" in content
