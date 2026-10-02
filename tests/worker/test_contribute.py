import json
import sqlite3
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from subzero.worker import contribute
from subzero.worker.opensubs import OpenSubtitlesError


@pytest.mark.parametrize("outcome", ["dry-run", "created", "failed"])
def test_contribution_batch_returns_a_summary_and_logs_out(tmp_path, monkeypatch, capsys, outcome):
    video = tmp_path / "movie.mkv"
    subtitle = video.with_suffix(".pt-BR.srt")
    subtitle.write_text("1\n00:00:01,000 --> 00:00:02,000\nOlá.\n", encoding="utf-8")
    env_file = tmp_path / ".env"
    env_file.write_text("JELLYFIN_URL=http://jf\nJELLYFIN_API_KEY=key\nBEARER_TOKEN=token\n")
    jellyfin = Mock()
    jellyfin.all_items.return_value = [{"Id": "movie", "Path": str(video)}]
    jellyfin.media.return_value = SimpleNamespace(imdb_id="tt123")
    client = Mock()
    upload = Mock(return_value=("created", 42, ""))
    if outcome == "failed":
        upload.side_effect = OpenSubtitlesError("upload failed")
    monkeypatch.setattr(contribute, "JellyfinClient", lambda *args, **kwargs: jellyfin)
    monkeypatch.setattr(contribute, "OpenSubtitles", lambda *args, **kwargs: client)
    monkeypatch.setattr(contribute, "upload", upload)
    monkeypatch.setattr(contribute.time, "sleep", lambda _: None)
    ledger = tmp_path / "uploads.db"
    args = ["--env", str(env_file), "--ledger", str(ledger), "--limit", "1"]
    if outcome == "dry-run":
        args.append("--dry-run")

    assert contribute.main(args) == (1 if outcome == "failed" else 0)

    tally = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert tally == {"sent": int(outcome != "failed"), "duplicate": 0,
                     "no_imdb": 0, "failed": int(outcome == "failed")}
    client.login.assert_called_once_with()
    client.logout.assert_called_once_with()
    assert upload.call_count == (0 if outcome == "dry-run" else 1)
    with sqlite3.connect(ledger) as db:
        assert db.execute("SELECT COUNT(*) FROM uploads").fetchone()[0] == int(outcome == "created")
