from pathlib import Path
import pytest

from subzero.worker.guards import check_excellence_guards, sanitize_to_excellence
from tests.worker.test_syncflow import dialogue, run, setup


def test_guards_reject_non_accepted_language():
    sample = dialogue()
    report = check_excellence_guards(sample, "en", accepted_langs=("pt-BR",))
    assert not report.ok
    assert "nao aceito" in report.reason


def test_guards_accept_any_language_when_unrestricted():
    sample = dialogue()
    report = check_excellence_guards(sample, "en")
    assert report.ok


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
    flow.cfg.accepted_langs = ["pt-BR"]
    job = jobs.enqueue("id", "audit", "en")
    jobs.start(job.id)
    with pytest.raises(ValueError, match="Worker configured to only accept"):
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


def test_guards_reject_unclosed_tags():
    sample = "1\n00:00:01,000 --> 00:00:03,000\n<i\nOla mundo!\n"
    report = check_excellence_guards(sample, "pt-BR")
    assert not report.ok
    assert "malformadas" in report.reason


def test_guards_reject_pt_machine_translation_capitalization():
    sample = "1\n00:00:01,000 --> 00:00:03,000\nTamatoa Não sempre foi assim.\n"
    report = check_excellence_guards(sample, "pt-BR")
    assert not report.ok
    assert "particulas capitalizadas" in report.reason


def test_sanitize_to_excellence_cleans_unclosed_tags():
    dirty = "1\n00:00:01,000 --> 00:00:03,000\n<i\nOla mundo!\n"
    cleaned = sanitize_to_excellence(dirty, "pt-BR")
    assert check_excellence_guards(cleaned, "pt-BR").ok
    assert "<i" not in cleaned
    assert "Ola mundo!" in cleaned


def test_sanitize_to_excellence_fixes_pt_mt_capitalization():
    dirty = "1\n00:00:01,000 --> 00:00:03,000\nTamatoa Não sempre foi assim.\n"
    cleaned = sanitize_to_excellence(dirty, "pt-BR")
    assert check_excellence_guards(cleaned, "pt-BR").ok
    assert "não sempre foi" in cleaned.lower()


def test_guards_reject_half_translated_subtitles():
    pt_cues = [f"{i}\n00:00:{i:02d},000 --> 00:00:{i:02d},500\nEu não sei o que você está dizendo." for i in range(1, 25)]
    en_cues = [f"{i}\n00:01:{i:02d},000 --> 00:01:{i:02d},500\nI told you that with this they were about what there have from the." for i in range(25, 40)]
    half_translated = "\n\n".join(pt_cues + en_cues) + "\n"
    report = check_excellence_guards(half_translated, "pt-BR")
    assert not report.ok
    assert "nao traduzido" in report.reason
