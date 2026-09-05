import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from subzero.worker.jobs import JobStore, Runner
from subzero.worker.jellyfin import Media
from subzero.worker.opensubs import Candidate
from subzero.worker.service import Service
from subzero.worker.srt import Cue,dump
from subzero.worker.syncflow import SyncFlow
from subzero.timing import spans


def dialogue(offset=0):
    rng=random.Random(4)
    cues=[]
    start=15
    while start<900:
        end=start+rng.uniform(1,3)
        cues.append(Cue(len(cues)+1,start+offset,end+offset,'Test dialogue'))
        start=end+rng.uniform(.4,4)
    return dump(cues)


@pytest.fixture
def setup(tmp_path):
    video=tmp_path/'movie.mkv';video.write_bytes(b'video')
    media=Media('id','movie',str(video),'mkv',940,'eng','src',imdb_id='tt123')
    jf=SimpleNamespace(media=lambda _:media,refresh=lambda _:None)
    candidates=[Candidate(i,f'movie.WEB.1080p-{i}','pt-BR',100,False,True,False,
                          imdb_id='123') for i in range(1,5)]
    provider=SimpleNamespace(search=lambda **_:candidates,download=lambda _:dialogue(8))
    service=Service(jf,provider,None,None)
    jobs=JobStore(str(tmp_path/'jobs.db'))
    cfg=SimpleNamespace(sync_cache=str(tmp_path/'cache'),daily_download_budget=10,
                        excluded_paths=[],sync_audit_only=False)
    ref={'speech':spans(dialogue()),'spans':spans(dialogue()),'text':dialogue(),'language':'eng'}
    flow=SyncFlow(jobs,service,cfg,reference_builder=lambda *args:ref)
    return flow,jobs,provider,media


def run(flow,jobs,kind):
    job=jobs.enqueue('id',kind,'pt-BR')
    Runner(jobs,{kind:flow.run}).run_once()
    return jobs.get(job.id)


def test_refetch_tries_distinct_candidates_and_only_installs_verified(setup):
    flow,jobs,provider,media=setup
    provider.download=lambda fid:dialogue() if fid==3 else dialogue(8)
    job=run(flow,jobs,'refetch')
    assert job.state=='done'
    assert Path(job.result_path).read_text()==dialogue()
    assert flow.state.downloads_today()==3


def test_refetch_stops_at_three_then_resyncs(setup):
    flow,jobs,provider,media=setup
    job=run(flow,jobs,'refetch')
    assert job.kind=='resync' and job.state=='queued'
    assert flow.state.downloads_today()==3
    Runner(jobs,{'resync':flow.run}).run_once()
    assert jobs.get(job.id).state=='done'


def test_cancelled_job_never_installs(setup):
    flow,jobs,provider,media=setup
    job=jobs.enqueue('id','refetch','pt-BR');jobs.start(job.id)
    jobs.cancel(job.id)
    with pytest.raises(RuntimeError,match='cancel'):
        flow.run(job,lambda *args:None)
    assert not list(Path(media.path).parent.glob('*.srt'))


def test_existing_pass_is_cached_and_replacement_has_backup(setup):
    flow,jobs,provider,media=setup
    path=Path(media.path).with_suffix('.pt-BR.srt');path.write_text(dialogue(8))
    provider.download=lambda _:dialogue()
    job=run(flow,jobs,'refetch')
    assert job.state=='done'
    assert path.read_text()==dialogue()
    assert list((Path(flow.cfg.sync_cache)/'backups').rglob('*.srt'))
    assert flow.current(media,'pt-BR')


def test_ambiguous_reference_does_not_overwrite_existing_subtitle(setup):
    flow,jobs,provider,media=setup
    original=Path(media.path).with_suffix('.pt-BR.srt');original.write_text(dialogue())
    flow.reference_builder=lambda *args:{'speech':[],'spans':[],'text':''}
    job=run(flow,jobs,'audit')
    assert job.state=='needs_review'
    assert original.read_text()==dialogue()


def test_exhausted_stages_end_in_review(setup):
    flow,jobs,provider,media=setup
    flow.reference_builder=lambda *args:{'speech':[],'spans':[],'text':''}
    job=run(flow,jobs,'rebuild')
    assert job.state=='needs_review'


def test_embedded_translation_preserves_verified_timings(setup):
    flow,jobs,provider,media=setup
    flow.service.ollama=SimpleNamespace(translate_block=lambda cues,lang:['Fala traduzida']*len(cues))
    job=run(flow,jobs,'embedded_translate')
    assert job.state=='done'
    assert 'Fala traduzida' in Path(job.result_path).read_text()
    assert spans(Path(job.result_path).read_text()) == spans(dialogue())


def test_rebuild_uses_audio_timings_and_validates_before_install(setup,monkeypatch,tmp_path):
    from subzero.worker import syncflow
    from subzero.worker.srt import parse
    flow,jobs,provider,media=setup
    flow.service.ollama=SimpleNamespace(release=lambda:None,
                                       translate_block=lambda cues,lang:['Fala traduzida']*len(cues))
    audio=tmp_path/'audio.wav';audio.write_bytes(b'audio')
    monkeypatch.setattr(syncflow,'extract_audio',lambda *args:str(audio))
    monkeypatch.setattr(syncflow,'audio_start_offset',lambda *args:0)
    monkeypatch.setattr(syncflow,'transcribe',lambda *args:(parse(dialogue()),'en'))
    job=run(flow,jobs,'rebuild')
    assert job.state=='done'
    assert not audio.exists()


def test_wrong_movie_candidate_is_not_downloaded(setup):
    flow,jobs,provider,media=setup
    for c in provider.search():
        c.imdb_id='456'
    job=run(flow,jobs,'refetch')
    assert flow.state.downloads_today()==0
    assert job.kind=='resync'


def test_audit_quarantines_confirmed_bad_subtitle_before_refetch(setup):
    flow,jobs,provider,media=setup
    path=Path(media.path).with_suffix('.pt-BR.srt');path.write_text(dialogue(8))
    job=run(flow,jobs,'audit')
    assert not path.exists()
    assert list((flow.cache/'quarantine').rglob('*.srt'))
    assert job.kind=='refetch'


def test_audit_only_never_replaces_or_quarantines(setup):
    flow,jobs,provider,media=setup
    flow.cfg.sync_audit_only=True
    path=Path(media.path).with_suffix('.pt-BR.srt');path.write_text(dialogue(8))
    job=run(flow,jobs,'audit')
    assert path.read_text()==dialogue(8)
    assert job.state=='needs_review'
    assert not (flow.cache/'quarantine').exists()
