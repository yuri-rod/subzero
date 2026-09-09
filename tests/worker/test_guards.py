from pathlib import Path
import pytest

from subzero.worker.guards import check_excellence_guards, sanitize_to_excellence
from tests.worker.test_syncflow import dialogue, run, setup


def test_guards_reject_non_pt_br():
    sample = dialogue()
    report = check_excellence_guards(sample, "en")
    assert not report.ok
    assert "esperado pt-BR" in report.reason


def test_guards_reject_sdh_cues():
    dirty = "1\n00:00:01,000 --> 00:00:03,000\n[som de chuva]\nOla mundo!\n"
    report = check_excellence_guards(dirty, "pt-BR")
    assert not report.ok
    assert "SDH" in report.reason


def test_sanitize_to_excellence_cleans_sdh_and_formats():
    dirty = "1\n00:00:01,000 --> 00:00:03,000\n[som de chuva]\nOla mundo!\n"
    cleaned = sanitize_to_excellence(dirty)
    report = check_excellence_guards(cleaned, "pt-BR")
    assert report.ok
    assert "[som de chuva]" not in cleaned
    assert "Ola mundo!" in cleaned


def test_guards_reject_collapsed_dialogue():
    dirty = "1\n00:00:01,000 --> 00:00:03,000\n- Maui! - Sim?\n"
    report = check_excellence_guards(dirty, "pt-BR")
    assert not report.ok
    assert "dialogo" in report.reason


def test_guards_reject_long_lines():
    dirty = "1\n00:00:01,000 --> 00:00:03,000\nEsta frase e deliberadamente muito longa para caber em uma linha de video sem quebra.\n"
    report = check_excellence_guards(dirty, "pt-BR")
    assert not report.ok
    assert "longas" in report.reason


def test_sanitize_to_excellence_fixes_collapsed_dialogue():
    dirty = "1\n00:00:01,000 --> 00:00:03,000\nMaui! - Sim?\n"
    cleaned = sanitize_to_excellence(dirty)
    assert check_excellence_guards(cleaned, "pt-BR").ok
    assert "- Maui!\n- Sim?" in cleaned


def test_syncflow_rejects_non_pt_br_jobs(setup):
    flow, jobs, provider, media = setup
    job = jobs.enqueue("id", "audit", "en")
    jobs.start(job.id)
    with pytest.raises(ValueError, match="Worker only accepts pt-BR subtitles"):
        flow.run(job, lambda *args: None)


def test_syncflow_current_rejects_subtitles_with_sdh(setup):
    flow, jobs, provider, media = setup
    dirty = "1\n00:00:15,000 --> 00:00:17,000\n[musica alegre]\nOla!\n"
    path = Path(media.path).with_suffix(".pt-BR.srt")
    path.write_text(dirty, encoding="utf-8")
    assert not flow.current(media, "pt-BR")


def test_syncflow_audit_cleans_existing_sdh_subtitle(setup):
    flow, jobs, provider, media = setup
    base = dialogue()
    dirty = base.replace("Test dialogue", "[music playing]\nTest dialogue", 1)
    path = Path(media.path).with_suffix(".pt-BR.srt")
    path.write_text(dirty, encoding="utf-8")

    job = run(flow, jobs, "audit")
    assert job.state == "done"
    installed_text = path.read_text(encoding="utf-8")
    assert "[music playing]" not in installed_text
    assert check_excellence_guards(installed_text, "pt-BR").ok
