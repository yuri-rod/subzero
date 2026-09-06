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
    flow.service.ollama=Translator()
    job=run(flow,jobs,'embedded_translate')
    assert job.state=='done'
    assert 'Fala traduzida' in Path(job.result_path).read_text()
    assert spans(Path(job.result_path).read_text()) == spans(dialogue())
    assert set(flow.service.ollama.source_languages) == {'eng'}


def test_rebuild_uses_audio_timings_and_validates_before_install(setup,monkeypatch,tmp_path):
    from subzero.worker import syncflow
    from subzero.worker.srt import parse
    flow,jobs,provider,media=setup
    flow.service.ollama=Translator()
    audio=tmp_path/'audio.wav';audio.write_bytes(b'audio')
    monkeypatch.setattr(syncflow,'extract_audio',lambda *args:str(audio))
    monkeypatch.setattr(syncflow,'audio_start_offset',lambda *args:0)
    monkeypatch.setattr(syncflow,'transcribe',lambda *args:(parse(dialogue()),'en'))
    job=run(flow,jobs,'rebuild')
    assert job.state=='done'
    assert not audio.exists()
    assert set(flow.service.ollama.source_languages) == {'en'}


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


class Translator:
    def __init__(self, failure=None, unavailable=None):
        self.failure = failure
        self.unavailable = unavailable
        self.loaded = False
        self.releases = 0
        self.source_languages = []

    def ensure_available(self):
        if self.unavailable:
            raise RuntimeError(self.unavailable)

    def translate_block(self, cues, lang, source_lang=None):
        self.source_languages.append(source_lang)
        self.loaded = True
        if self.failure=='invalid translation lines':
            return []
        if self.failure:
            raise RuntimeError(self.failure)
        return ['Fala traduzida'] * len(cues)

    def release(self):
        self.loaded = False
        self.releases += 1


@pytest.mark.parametrize('translator', [None, Translator(unavailable='model missing')])
def test_rebuild_without_translation_stops_before_audio_reference(setup, translator):
    flow,jobs,provider,media=setup
    flow.service.ollama=translator
    def no_audio(*args):
        raise AssertionError('Audio work must wait for translation availability')
    flow.reference_builder=no_audio
    job=run(flow,jobs,'rebuild')
    assert job.state=='needs_review'
    assert not Path(media.path).with_suffix('.pt-BR.srt').exists()


@pytest.mark.parametrize('failure', ['Ollama unavailable', 'invalid translation lines'])
def test_verified_source_translation_failure_does_not_rebuild(setup, failure):
    flow,jobs,provider,media=setup
    translator=Translator(failure=failure)
    flow.service.ollama=translator
    job=run(flow,jobs,'embedded_translate')
    assert job.state=='needs_review'
    assert job.kind=='embedded_translate'
    assert not translator.loaded
    assert translator.releases==1
    assert not Path(media.path).with_suffix('.pt-BR.srt').exists()


def test_translation_releases_model_after_completion(setup):
    flow,jobs,provider,media=setup
    translator=Translator()
    flow.service.ollama=translator
    job=run(flow,jobs,'embedded_translate')
    assert job.state=='done'
    assert not translator.loaded
    assert translator.releases==1


def test_translation_releases_model_when_job_cancelled(setup):
    flow,jobs,provider,media=setup
    translator=Translator()
    flow.service.ollama=translator
    job=jobs.enqueue('id','embedded_translate','pt-BR')
    jobs.start(job.id)
    def cancel_on_translation(phase,percent):
        if phase.startswith('traduzindo'):
            jobs.cancel(job.id)
    flow.run(job,cancel_on_translation)
    assert jobs.get(job.id).state=='cancelled'
    assert not translator.loaded
    assert translator.releases==1
    assert not Path(media.path).with_suffix('.pt-BR.srt').exists()


@pytest.mark.parametrize('kind', ['embedded_translate', 'rebuild'])
def test_aligned_source_sidecar_avoids_transcription(setup, monkeypatch, kind):
    from subzero.worker import syncflow
    flow,jobs,provider,media=setup
    flow.service.ollama=Translator()
    reference=flow.reference_builder()
    reference['text']=''
    source=Path(media.path).with_suffix('.en.srt')
    source.write_text(dialogue())
    def no_transcription(*args):
        raise AssertionError('Verified source subtitles do not need transcription')
    monkeypatch.setattr(syncflow,'extract_audio',no_transcription)
    monkeypatch.setattr(syncflow,'transcribe',no_transcription)
    job=run(flow,jobs,kind)
    assert job.state=='done'
    assert source.read_text()==dialogue()
    output=Path(job.result_path).read_text()
    assert 'Fala traduzida' in output
    assert spans(output)==spans(dialogue())


def test_misaligned_source_sidecar_is_rejected(setup):
    flow,jobs,provider,media=setup
    flow.service.ollama=Translator()
    reference=flow.reference_builder()
    reference['text']=''
    Path(media.path).with_suffix('.en.srt').write_text(dialogue(8))
    job=run(flow,jobs,'embedded_translate')
    assert job.kind=='rebuild' and job.state=='queued'
    assert not Path(media.path).with_suffix('.pt-BR.srt').exists()


@pytest.mark.parametrize('source_kind', ['forced', 'other_video', 'symlink', 'oversized', 'directory'])
def test_source_sidecar_must_be_safe_and_belong_to_video(setup, source_kind):
    flow,jobs,provider,media=setup
    flow.service.ollama=Translator()
    reference=flow.reference_builder()
    reference['text']=''
    source=Path(media.path).with_suffix('.en.srt')
    if source_kind=='forced':
        source=source.with_suffix('.forced.srt')
    elif source_kind=='other_video':
        source=source.with_name('another.en.srt')
    if source_kind=='symlink':
        original=source.with_name('outside.srt')
        original.write_text(dialogue())
        source.symlink_to(original)
    elif source_kind=='directory':
        source.mkdir()
    else:
        source.write_text(dialogue()+('\n'*2_000_000 if source_kind=='oversized' else ''))
    job=run(flow,jobs,'embedded_translate')
    assert job.kind=='rebuild' and job.state=='queued'
    assert not Path(media.path).with_suffix('.pt-BR.srt').exists()


def test_source_sidecar_respects_configured_source_languages(setup):
    flow,jobs,provider,media=setup
    flow.service.ollama=Translator()
    flow.cfg.translate_from=['es']
    reference=flow.reference_builder()
    reference['text']=''
    Path(media.path).with_suffix('.en.srt').write_text(dialogue())
    job=run(flow,jobs,'embedded_translate')
    assert job.kind=='rebuild' and job.state=='queued'


@pytest.mark.parametrize('audio_lang', ['por', 'und', ''])
@pytest.mark.parametrize('configured', [False, True])
@pytest.mark.parametrize('source_sidecar', [False, True])
def test_same_language_audio_rebuild_without_ollama(setup, monkeypatch, tmp_path, audio_lang, configured, source_sidecar):
    from subzero.worker import syncflow
    from subzero.worker.srt import parse
    flow,jobs,provider,media=setup
    media.audio_lang=audio_lang
    if source_sidecar:
        Path(media.path).with_suffix('.en.srt').write_text(dialogue())
    if configured:
        flow.service.ollama=Translator(failure='Translation must not run',unavailable='Ollama unavailable')
    reference=flow.reference_builder()
    reference['text']=''
    audio=tmp_path/'audio.wav'
    audio.write_bytes(b'audio')
    monkeypatch.setattr(syncflow,'extract_audio',lambda *args:str(audio))
    monkeypatch.setattr(syncflow,'audio_start_offset',lambda *args:0)
    monkeypatch.setattr(syncflow,'transcribe',lambda *args:(parse(dialogue()),'pt'))
    job=run(flow,jobs,'rebuild')
    assert job.state=='done'
    assert Path(job.result_path).read_text()==dialogue()
    assert not audio.exists()


def test_source_translation_with_wrong_timing_goes_to_review(setup, monkeypatch):
    from subzero.worker import syncflow
    from subzero.worker.srt import parse
    flow,jobs,provider,media=setup
    flow.service.ollama=Translator()
    monkeypatch.setattr(syncflow,'translate',lambda *args,**kwargs:parse(dialogue(8)))
    job=run(flow,jobs,'embedded_translate')
    assert job.state=='needs_review'
    assert job.kind=='embedded_translate'
    assert not Path(media.path).with_suffix('.pt-BR.srt').exists()


@pytest.mark.parametrize('portable_open', [False, True])
def test_source_sidecar_accepts_three_letter_language_tag(setup, monkeypatch, portable_open):
    flow,jobs,provider,media=setup
    flow.service.ollama=Translator()
    reference=flow.reference_builder()
    reference['text']=''
    Path(media.path).with_suffix('.eng.srt').write_text(dialogue())
    if portable_open:
        from subzero.worker import syncflow
        monkeypatch.delattr(syncflow.os,'O_NOFOLLOW',raising=False)
        monkeypatch.delattr(syncflow.os,'O_NONBLOCK',raising=False)
    job=run(flow,jobs,'embedded_translate')
    assert job.state=='done'
    assert 'Fala traduzida' in Path(job.result_path).read_text()


@pytest.mark.parametrize('audio_lang', ['und', ''])
def test_unknown_audio_needing_translation_stops_after_detection(setup, monkeypatch, tmp_path, audio_lang):
    from subzero.worker import syncflow
    from subzero.worker.srt import parse
    flow,jobs,provider,media=setup
    media.audio_lang=audio_lang
    reference=flow.reference_builder()
    reference['text']=''
    audio=tmp_path/'audio.wav'
    audio.write_bytes(b'audio')
    monkeypatch.setattr(syncflow,'extract_audio',lambda *args:str(audio))
    monkeypatch.setattr(syncflow,'audio_start_offset',lambda *args:0)
    monkeypatch.setattr(syncflow,'transcribe',lambda *args:(parse(dialogue()),'en'))
    job=run(flow,jobs,'rebuild')
    assert job.state=='needs_review'
    assert not audio.exists()
    assert not Path(media.path).with_suffix('.pt-BR.srt').exists()


@pytest.mark.parametrize('failure', ['Ollama unavailable', 'invalid translation lines'])
def test_rebuild_does_not_transcribe_after_source_translation_failure(setup, monkeypatch, failure):
    from subzero.worker import syncflow
    flow,jobs,provider,media=setup
    media.audio_lang='por'
    Path(media.path).with_suffix('.en.srt').write_text(dialogue())
    flow.service.ollama=Translator(failure=failure)
    def no_transcription(*args):
        raise AssertionError('Do not retranscribe after starting source translation')
    monkeypatch.setattr(syncflow,'extract_audio',no_transcription)
    job=run(flow,jobs,'rebuild')
    assert job.state=='needs_review'
    assert job.kind=='rebuild'
    assert not Path(media.path).with_suffix('.pt-BR.srt').exists()
