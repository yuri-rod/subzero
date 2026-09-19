import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from subzero.worker.jobs import JobStore, Runner
from subzero.worker.jellyfin import Media
from subzero.worker.opensubs import Candidate
from subzero.worker.service import Service
from subzero.worker.srt import Cue,dump,parse
from subzero.worker.syncflow import SyncFlow
from subzero.timing import Report, spans


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
    ref={'speech':spans(dialogue()),'spans':spans(dialogue()),'text':dialogue(),'language':'eng','duration':940}
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


@pytest.fixture
def audio_rebuild(setup, monkeypatch, tmp_path):
    from subzero.worker import syncflow

    flow, jobs, _, media = setup
    flow.cfg.ocr_enabled = True
    flow.service.ollama = Translator()
    complete = dialogue().replace('Test dialogue', 'We should build a shelter.')
    cues = parse(complete)
    source = dump(cues[:10] + cues[11:])
    reference = flow.reference_builder()
    reference.update(text='', spans=[], speech=spans(complete), audio_index=3)
    target = Path(media.path).with_suffix('.pt-BR.srt')
    target.write_text('broken old translation')
    audio = tmp_path / 'audio.wav'
    audio.write_bytes(b'audio')
    calls = []

    def extract(*args, **kwargs):
        calls.append(('audio', kwargs))
        return str(audio)

    def transcribe(*args):
        calls.append(('transcribe', None))
        return syncflow.shift(parse(source), -1.4), 'en'

    def recover(video, subtitle_path, **kwargs):
        calls.append(('ocr', None))
        assert Path(subtitle_path).read_text() == source
        assert kwargs['target_lang'] == 'en'
        assert kwargs['all_captions'] is True
        assert kwargs['backup'] is False
        Path(kwargs['output']).write_text(complete)
        return SimpleNamespace(cues_recovered=1)

    monkeypatch.setattr(syncflow, 'audio_start_offset', lambda *args: 1.4)
    monkeypatch.setattr(syncflow, 'extract_audio', extract)
    monkeypatch.setattr(syncflow, 'transcribe', transcribe)
    monkeypatch.setattr(syncflow, 'fill_subtitle_gaps', recover)
    return flow, jobs, media, target, reference, source, complete, audio, calls


def test_english_audio_rebuild_uses_ocr_and_preserves_independent_reference(audio_rebuild):
    import copy

    flow, jobs, _, target, reference, source, complete, audio, calls = audio_rebuild
    original_reference = copy.deepcopy(reference)
    job = run(flow, jobs, 'rebuild')
    assert job.state == 'done', job.phase
    assert calls == [('audio', {'audio_index': 3}), ('transcribe', None), ('ocr', None)]
    assert reference == original_reference
    assert spans(target.read_text()) == spans(complete)
    assert len(parse(target.read_text())) == len(parse(source)) + 1
    assert set(flow.service.ollama.source_languages) == {'en'}
    assert 'We should' not in target.read_text()
    assert not audio.exists()
    assert list((flow.cache / 'candidates').glob('*/en/*.srt'))
    assert list((flow.cache / 'backups').rglob('*.srt'))[0].read_text() == 'broken old translation'


@pytest.mark.parametrize('phase', [0, 45])
@pytest.mark.parametrize('status', ['reject', 'inconclusive'])
def test_english_audio_rebuild_requires_both_timing_checks(audio_rebuild, monkeypatch, phase, status):
    from subzero.worker import syncflow
    from subzero.timing import Report

    flow, jobs, _, target, _, _, _, audio, calls = audio_rebuild
    verify = syncflow.verify_text

    def reject_phase(text, reference, **kwargs):
        if kwargs.get('phase', 0) == phase:
            return Report(status, 'Independent audio timing failed', [])
        return verify(text, reference, **kwargs)

    monkeypatch.setattr(syncflow, 'verify_text', reject_phase)
    job = run(flow, jobs, 'rebuild')
    assert job.state == 'needs_review'
    assert target.read_text() == 'broken old translation'
    assert ('ocr', None) not in calls
    assert flow.service.ollama.source_languages == []
    assert not audio.exists()
    assert not list((flow.cache / 'backups').rglob('*.srt'))


@pytest.mark.parametrize('existed', [False, True])
def test_audio_rebuild_preserves_target_changed_during_transcription(audio_rebuild, monkeypatch, existed):
    from subzero.worker import syncflow

    flow, jobs, _, target, _, _, _, audio, _ = audio_rebuild
    if not existed:
        target.unlink()
    transcribe = syncflow.transcribe

    def change_target(*args):
        target.write_text('new subtitle installed during transcription')
        return transcribe(*args)

    monkeypatch.setattr(syncflow, 'transcribe', change_target)
    job = run(flow, jobs, 'rebuild')
    assert job.state == 'needs_review'
    assert target.read_text() == 'new subtitle installed during transcription'
    assert not audio.exists()
    assert not list((flow.cache / 'backups').rglob('*.srt'))


@pytest.mark.parametrize('audio_index', [None, True, 1.0, '1', -1])
def test_ocr_audio_rebuild_rejects_invalid_reference_stream_before_audio(audio_rebuild, audio_index):
    flow, jobs, _, target, reference, _, _, _, calls = audio_rebuild
    reference['audio_index'] = audio_index
    job = run(flow, jobs, 'rebuild')
    assert job.state == 'needs_review'
    assert 'audio stream index' in job.phase.lower()
    assert calls == []
    assert target.read_text() == 'broken old translation'


@pytest.mark.parametrize('failure', ['ocr', 'translation', 'cancel'])
def test_english_audio_rebuild_failure_preserves_target(audio_rebuild, monkeypatch, failure):
    from subzero.worker import syncflow

    flow, jobs, _, target, _, _, _, audio, calls = audio_rebuild
    if failure == 'ocr':
        def fail(*args, **kwargs):
            raise OSError('Unreadable caption observations')
        monkeypatch.setattr(syncflow, 'fill_subtitle_gaps', fail)
    elif failure == 'translation':
        flow.service.ollama.failure = 'translation unavailable'
    else:
        transcribe = syncflow.transcribe
        def cancel(*args):
            jobs.cancel(jobs.active()[0].id)
            return transcribe(*args)
        monkeypatch.setattr(syncflow, 'transcribe', cancel)
    job = run(flow, jobs, 'rebuild')
    assert job.state == ('cancelled' if failure == 'cancel' else 'needs_review')
    assert target.read_text() == 'broken old translation'
    assert not audio.exists()
    assert not list((flow.cache / 'backups').rglob('*.srt'))
    if failure == 'cancel':
        assert ('ocr', None) not in calls
        assert flow.service.ollama.source_languages == []


@pytest.mark.parametrize('detected', ['pt', 'und'])
def test_english_audio_rebuild_rejects_non_english_detection(audio_rebuild, monkeypatch, detected):
    from subzero.worker import syncflow

    flow, jobs, _, target, _, source, _, audio, calls = audio_rebuild
    monkeypatch.setattr(syncflow, 'transcribe', lambda *args: (parse(source), detected))
    job = run(flow, jobs, 'rebuild')
    assert job.state == 'needs_review'
    assert 'not transcribed as English' in job.phase
    assert target.read_text() == 'broken old translation'
    assert ('ocr', None) not in calls
    assert flow.service.ollama.source_languages == []
    assert not audio.exists()


def test_audio_rebuild_respects_explicit_ocr_off(audio_rebuild):
    flow, jobs, _, target, reference, source, _, _, calls = audio_rebuild
    flow.cfg.ocr_enabled = False
    reference.pop('audio_index')
    job = run(flow, jobs, 'rebuild')
    assert job.state == 'done'
    assert calls == [('audio', {}), ('transcribe', None)]
    assert spans(target.read_text()) == spans(source)


def test_non_english_audio_rebuild_does_not_use_english_ocr(audio_rebuild, monkeypatch):
    from subzero.worker import syncflow

    flow, jobs, media, target, reference, _, _, audio, calls = audio_rebuild
    media.audio_lang = 'por'
    reference['language'] = 'por'
    flow.service.ollama = None
    portuguese = dialogue().replace('Test dialogue', 'Precisamos construir um abrigo.')
    monkeypatch.setattr(syncflow, 'transcribe',
                        lambda *args: (syncflow.shift(parse(portuguese), -1.4), 'pt'))
    job = run(flow, jobs, 'rebuild')
    assert job.state == 'done'
    assert target.read_text() == portuguese
    assert calls == [('audio', {'audio_index': 3})]
    assert not audio.exists()


@pytest.mark.parametrize('kind', ['symlink', 'fifo'])
def test_audio_rebuild_rejects_unsafe_target_before_audio(audio_rebuild, kind):
    import os

    if kind == 'fifo' and not hasattr(os, 'mkfifo'):
        pytest.skip('FIFO creation requires a POSIX filesystem')
    flow, jobs, _, target, _, _, _, _, calls = audio_rebuild
    target.unlink()
    unrelated = target.parent / 'unrelated.txt'
    unrelated.write_text('keep this text')
    if kind == 'symlink':
        target.symlink_to(unrelated)
    else:
        os.mkfifo(target)
    job = run(flow, jobs, 'rebuild')
    assert job.state == 'needs_review'
    assert 'regular subtitle file' in job.phase
    assert unrelated.read_text() == 'keep this text'
    assert calls == []


def test_broken_file_is_never_downloaded_again(setup):
    from subzero.reference import fingerprint
    flow,jobs,provider,media=setup
    key=fingerprint(str(media.path))
    flow.state.reserve(key,'pt-BR',1,'old',10)
    flow.state.update(key,'pt-BR',1,status='broken')
    asked=[]
    real=provider.download
    provider.download=lambda fid:asked.append(fid) or real(fid)
    job=run(flow,jobs,'refetch')
    assert 1 not in asked
    assert job.kind=='resync'


def test_empty_download_is_recorded_as_broken(setup):
    from subzero.reference import fingerprint
    flow,jobs,provider,media=setup
    key=fingerprint(str(media.path))
    provider.download=lambda fid:'' if fid==1 else dialogue(8)
    job=run(flow,jobs,'refetch')
    assert job.kind=='resync'
    statuses={a['file_id']:a['status'] for a in flow.state.attempts(key,'pt-BR')}
    assert statuses[1]=='broken'


def test_frame_rate_mismatch_skips_every_candidate(setup):
    from subzero.worker.jellyfin import Media
    flow,jobs,provider,media=setup
    for c in provider.search():
        c.fps=23.976
    bad=Media('id','movie',str(media.path),'mkv',940,'eng','src',imdb_id='tt123',fps=25.0)
    flow.service.jellyfin=SimpleNamespace(media=lambda _:bad,refresh=lambda _:None)
    job=run(flow,jobs,'refetch')
    assert flow.state.downloads_today()==0
    assert job.kind=='resync'


def test_exhausted_quota_skips_downloads_and_resyncs(setup):
    flow,jobs,provider,media=setup
    provider.quota_exhausted=lambda:True
    asked=[]
    provider.download=lambda fid:asked.append(fid) or dialogue(8)
    job=run(flow,jobs,'refetch')
    assert asked==[]
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
    assert not source.exists()
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


def test_install_prunes_bare_srt_when_installing_tagged_sub(setup):
    flow,jobs,provider,media=setup
    bare = Path(media.path).with_suffix('.srt')
    bare.write_text('old bare sub')
    assert bare.exists()
    provider.download = lambda fid: dialogue()
    job = run(flow, jobs, 'refetch')
    assert job.state == 'done'
    target = Path(job.result_path)
    assert target.exists()
    assert target.name.endswith('.pt-BR.srt')
    assert not bare.exists()


def test_install_prunes_intermediate_sidecars_matching_embedded_tracks(setup):
    flow,jobs,provider,media=setup
    intermediate = Path(media.path).parent / f"{Path(media.path).stem}.en.srt"
    intermediate.write_text('extracted intermediate en sub')
    assert intermediate.exists()
    # media has embedded stream with lang='eng'
    from types import SimpleNamespace
    media.embedded = [SimpleNamespace(index=2, lang='eng', codec='subrip', title='English', external=False)]
    provider.download = lambda fid: dialogue()
    job = run(flow, jobs, 'refetch')
    assert job.state == 'done'
    target = Path(job.result_path)
    assert target.exists()
    assert not intermediate.exists()


@pytest.mark.parametrize('suffix', ['', '[EZTVx.to]', '[TGx]'])
def test_install_prunes_literal_stem_sidecars_and_preserves_neighbors(setup, suffix):
    flow, jobs, provider, media = setup
    video = Path(media.path).with_name(f'movie{suffix}.mkv')
    Path(media.path).rename(video)
    media.path = str(video)
    media.embedded = [SimpleNamespace(lang='eng')]
    stale = [video.with_suffix('.srt'), video.with_suffix('.en.srt')]
    preserved = [video.with_suffix('.fr.srt'), video.parent / f'{video.stem}.en.srt.bak',
                 video.parent / f'{video.stem}Xen.srt', video.parent / f'{video.stem}.extended.en.srt']
    for path in stale + preserved:
        path.write_text('existing subtitle')
    provider.download = lambda _: dialogue()
    job = run(flow, jobs, 'refetch')
    assert job.state == 'done'
    assert Path(job.result_path).exists()
    assert all(not path.exists() for path in stale)
    assert all(path.read_text() == 'existing subtitle' for path in preserved)


@pytest.mark.parametrize('ocr_enabled', [False, True])
@pytest.mark.parametrize('existed', [False, True])
def test_rebuild_sidecar_fallback_keeps_early_target_snapshot(audio_rebuild, monkeypatch, ocr_enabled, existed):
    flow, jobs, media, target, _, source, _, _, calls = audio_rebuild
    flow.cfg.ocr_enabled = ocr_enabled
    if not existed:
        target.unlink()
    Path(media.path).with_suffix('.en.srt').write_text(source)
    source_sidecar = flow.source_sidecar

    def replace_target(*args, **kwargs):
        selected = source_sidecar(*args, **kwargs)
        assert selected is not None
        target.write_text('changed during source selection')
        return selected

    monkeypatch.setattr(flow, 'source_sidecar', replace_target)
    job = run(flow, jobs, 'rebuild')
    assert job.state == 'needs_review'
    assert target.read_text() == 'changed during source selection'
    assert not any(kind in ('audio', 'transcribe') for kind, _ in calls)
    assert not list((flow.cache / 'backups').rglob('*.srt'))


@pytest.fixture
def gap_recovery(setup, monkeypatch):
    from subzero.worker import syncflow
    flow, jobs, provider, media = setup
    flow.cfg.ollama_url = 'http://127.0.0.1:11434'
    flow.cfg.ollama_model = 'translategemma:4b'
    flow.service.ollama = Translator()
    complete = dialogue().replace('Test dialogue', 'Fala traduzida')
    cues = parse(complete)
    original = dump(cues[:10] + cues[11:])
    target = Path(media.path).with_suffix('.pt-BR.srt')
    target.write_text(original)

    def recover(video, subtitle_path, **kwargs):
        assert Path(video) == Path(media.path)
        assert Path(subtitle_path) != target
        assert Path(subtitle_path).read_text() == original
        assert kwargs['target_lang'] == 'pt-BR'
        assert kwargs['translation_client'] is flow.service.ollama
        assert kwargs['cache_dir'] == flow.cache / 'references'
        assert kwargs['backup'] is False
        assert Path(kwargs['output']) != target
        kwargs['progress'](1, 1)
        Path(kwargs['output']).write_text(complete)
        return SimpleNamespace(cues_recovered=1)

    monkeypatch.setattr(syncflow, 'fill_subtitle_gaps', recover, raising=False)
    return flow, jobs, media, target, original, complete


def test_recover_gaps_adds_translated_captions_and_backs_up_original(gap_recovery):
    flow, jobs, media, target, original, complete = gap_recovery
    job = run(flow, jobs, 'recover_gaps')
    assert job.state == 'done'
    assert job.result_path == str(target)
    assert target.read_text() == complete
    backups = list((flow.cache / 'backups').rglob('*.srt'))
    assert len(backups) == 1
    assert backups[0].read_text() == original
    assert not list(flow.cache.glob('ocr-*'))


def test_recover_gaps_wraps_long_recovered_caption_before_validation(gap_recovery, monkeypatch):
    from subzero.worker import syncflow
    flow, jobs, media, target, original, complete = gap_recovery
    caption = 'Precisamos falar com o grupo antes da votação de hoje.'

    def recover(video, subtitle_path, **kwargs):
        merged = parse(complete)
        merged[10] = Cue(merged[10].index, merged[10].start, merged[10].end, caption)
        Path(kwargs['output']).write_text(dump(merged), encoding='utf-8')
        return SimpleNamespace(cues_recovered=1)

    monkeypatch.setattr(syncflow, 'fill_subtitle_gaps', recover)
    job = run(flow, jobs, 'recover_gaps')
    assert job.state == 'done'
    recovered = parse(target.read_text(encoding='utf-8'))[10]
    assert recovered.text.replace('\n', ' ') == caption
    assert all(len(line) <= 42 for line in recovered.text.splitlines())


def test_recover_gaps_accepts_frame_captions_when_merged_audio_metric_drifts(gap_recovery, monkeypatch):
    from subzero.ocr import uncovered_intervals
    from subzero.reference import verify_text
    from subzero.worker import syncflow
    flow, jobs, media, target, original, complete = gap_recovery
    reference = flow.reference_builder()
    gaps = uncovered_intervals([(0, 940)], [(c.start, c.end) for c in parse(original)])
    captions = [Cue(0, start, end, 'Fala traduzida na tela.') for start, end in gaps]
    augmented = dump(sorted(parse(original) + captions, key=lambda cue: cue.start))
    assert verify_text(original, reference).status == 'pass'
    assert verify_text(augmented, reference).status != 'pass'

    def recover(video, subtitle_path, **kwargs):
        Path(kwargs['output']).write_text(augmented)
        return SimpleNamespace(cues_recovered=len(captions))

    monkeypatch.setattr(syncflow, 'fill_subtitle_gaps', recover)
    job = run(flow, jobs, 'recover_gaps')
    assert job.state == 'done'
    assert target.read_text() == augmented


@pytest.mark.parametrize('fault', ['changed_anchor', 'untranslated_source'])
def test_recover_gaps_rejects_changed_anchors_or_existing_english(gap_recovery, monkeypatch, fault):
    from subzero.worker import syncflow
    flow, jobs, media, target, original, complete = gap_recovery
    if fault == 'untranslated_source':
        original = original.replace('Fala traduzida', 'We need to find the hidden immunity idol.')
        complete = complete.replace('Fala traduzida', 'We need to find the hidden immunity idol.')
        target.write_text(original)
    else:
        complete = complete.replace('Fala traduzida', 'Outra fala.', 1)

    def recover(video, subtitle_path, **kwargs):
        Path(kwargs['output']).write_text(complete)
        return SimpleNamespace(cues_recovered=1)

    monkeypatch.setattr(syncflow, 'fill_subtitle_gaps', recover)
    job = run(flow, jobs, 'recover_gaps')
    assert job.state == 'needs_review'
    assert target.read_text() == original
    assert not (flow.cache / 'backups').exists()


@pytest.mark.parametrize('failure', ['translation', 'invalid_timing', 'empty_output', 'no_captions'])
def test_recover_gaps_failure_preserves_original_without_fallback(gap_recovery, monkeypatch, failure):
    from subzero.worker import syncflow
    flow, jobs, media, target, original, complete = gap_recovery

    def recover(video, subtitle_path, **kwargs):
        if failure == 'translation':
            raise RuntimeError('Translation returned incomplete captions')
        if failure == 'no_captions':
            return SimpleNamespace(cues_recovered=0)
        Path(kwargs['output']).write_text(dialogue(8) if failure == 'invalid_timing' else '')
        return SimpleNamespace(cues_recovered=1)

    monkeypatch.setattr(syncflow, 'fill_subtitle_gaps', recover)
    job = run(flow, jobs, 'recover_gaps')
    assert job.state == 'needs_review'
    assert job.kind == 'recover_gaps'
    assert target.read_text() == original
    assert not (flow.cache / 'backups').exists()
    assert not list(flow.cache.glob('ocr-*'))


def test_recover_gaps_cancellation_preserves_original(gap_recovery):
    flow, jobs, media, target, original, complete = gap_recovery
    job = jobs.enqueue('id', 'recover_gaps', 'pt-BR')
    jobs.start(job.id)

    def cancel_on_ocr(phase, percent):
        if phase.startswith('recuperando') and percent > 0:
            jobs.cancel(job.id)

    try:
        flow.run(job, cancel_on_ocr)
    except RuntimeError as err:
        assert 'cancel' in str(err).lower()
    assert jobs.get(job.id).state == 'cancelled'
    assert target.read_text() == original
    assert not (flow.cache / 'backups').exists()
    assert not list(flow.cache.glob('ocr-*'))


@pytest.mark.parametrize('translator', [None, Translator(unavailable='model missing')])
def test_recover_gaps_requires_translation_before_ocr(gap_recovery, monkeypatch, translator):
    from subzero.worker import syncflow
    flow, jobs, media, target, original, complete = gap_recovery
    flow.service.ollama = translator

    def no_ocr(*args, **kwargs):
        raise AssertionError('Translation must be available before caption recovery')

    monkeypatch.setattr(syncflow, 'fill_subtitle_gaps', no_ocr)
    job = run(flow, jobs, 'recover_gaps')
    assert job.state == 'needs_review'
    assert target.read_text() == original


def test_recover_gaps_rejects_replaced_target_without_reading_symlink(gap_recovery, monkeypatch):
    from subzero.worker import syncflow
    flow, jobs, media, target, original, complete = gap_recovery
    outside = target.with_name('outside.srt')
    outside.write_text(original)
    read_text = Path.read_text

    def no_symlink_read(path, *args, **kwargs):
        if path == target and path.is_symlink():
            raise AssertionError('Do not read a changed subtitle symlink')
        return read_text(path, *args, **kwargs)

    def recover(video, subtitle_path, **kwargs):
        target.unlink()
        target.symlink_to(outside)
        Path(kwargs['output']).write_text(complete)
        return SimpleNamespace(cues_recovered=1)

    monkeypatch.setattr(syncflow, 'fill_subtitle_gaps', recover)
    monkeypatch.setattr(Path, 'read_text', no_symlink_read)
    job = run(flow, jobs, 'recover_gaps')
    assert job.state == 'needs_review'
    assert target.is_symlink()
    assert outside.read_text() == original


def test_recover_gaps_does_not_overwrite_concurrent_sidecar_edit(gap_recovery, monkeypatch):
    from subzero.worker import syncflow
    flow, jobs, media, target, original, complete = gap_recovery
    newer = original.replace('Fala traduzida', 'Legenda revisada')

    def recover(video, subtitle_path, **kwargs):
        target.write_text(newer)
        Path(kwargs['output']).write_text(complete)
        return SimpleNamespace(cues_recovered=1)

    monkeypatch.setattr(syncflow, 'fill_subtitle_gaps', recover)
    job = run(flow, jobs, 'recover_gaps')
    assert job.state in ('failed', 'needs_review')
    assert target.read_text() == newer
    assert not list((flow.cache / 'backups').rglob('*.srt'))


@pytest.mark.parametrize('source_kind', ['missing', 'symlink', 'directory', 'oversized'])
def test_recover_gaps_requires_safe_existing_target(gap_recovery, monkeypatch, source_kind):
    from subzero.worker import syncflow
    flow, jobs, media, target, original, complete = gap_recovery
    target.unlink()
    if source_kind == 'symlink':
        outside = target.with_name('outside.srt')
        outside.write_text(original)
        target.symlink_to(outside)
    elif source_kind == 'directory':
        target.mkdir()
    elif source_kind == 'oversized':
        target.write_text(original + '\n' * 2_000_000)

    def no_ocr(*args, **kwargs):
        raise AssertionError('Unsafe source must stop before OCR')

    monkeypatch.setattr(syncflow, 'fill_subtitle_gaps', no_ocr)
    job = run(flow, jobs, 'recover_gaps')
    assert job.state in ('failed', 'needs_review')
    assert job.kind == 'recover_gaps'
    assert not (flow.cache / 'backups').exists()
    if source_kind == 'symlink':
        assert target.is_symlink()
        assert outside.read_text() == original


def test_recover_gaps_audit_only_preserves_original(gap_recovery):
    flow, jobs, media, target, original, complete = gap_recovery
    flow.cfg.sync_audit_only = True
    job = run(flow, jobs, 'recover_gaps')
    assert target.read_text() == original
    assert not (flow.cache / 'backups').exists()


@pytest.fixture
def repair_flow(setup, monkeypatch):
    from subzero.worker import syncflow
    flow, jobs, provider, media = setup
    complete = dialogue().replace('Test dialogue', 'SAM:\nWe should build a shelter.')
    cues = parse(complete)
    source = dump(cues[:10] + cues[11:])
    reference = flow.reference_builder()
    reference['text'] = source
    target = Path(media.path).with_suffix('.pt-BR.srt')
    target.write_text('broken old translation')
    calls = []

    class ContextTranslator(Translator):
        def translate_block(self, cues, lang, source_lang=None):
            self.source_cues = getattr(self, 'source_cues', []) + cues
            return super().translate_block(cues, lang, source_lang)

    flow.service.ollama = ContextTranslator()

    def recover(video, subtitle_path, **kwargs):
        calls.append(str(subtitle_path))
        assert Path(subtitle_path).read_text() == reference['text']
        assert kwargs['target_lang'] == 'en'
        assert kwargs['backup'] is False
        assert kwargs['all_captions'] is True
        assert kwargs['caption_cache_dir'] == flow.cache / 'caption-scans'
        kwargs['progress'](1, 1)
        merged = parse(reference['text'])
        merged.insert(10, cues[10])
        Path(kwargs['output']).write_text(dump(merged), encoding='utf-8')
        return SimpleNamespace(cues_recovered=1)

    monkeypatch.setattr(syncflow, 'fill_subtitle_gaps', recover)
    return flow, jobs, media, target, reference, complete, calls


def test_repair_replaces_entire_translation_from_english_and_ocr(repair_flow):
    flow, jobs, media, target, reference, complete, calls = repair_flow
    job = run(flow, jobs, 'repair')
    assert job.state == 'done'
    translated = target.read_text()
    assert 'broken old translation' not in translated
    assert 'We should' not in translated
    assert spans(translated) == spans(complete)
    assert all('SAM:' in cue.text for cue in flow.service.ollama.source_cues)
    assert set(flow.service.ollama.source_languages) == {'en'}
    assert len(calls) == 1
    assert list((flow.cache / 'candidates').glob('*/en/*.srt'))
    assert list((flow.cache / 'candidates').glob('*/pt-BR/*.srt'))
    assert list((flow.cache / 'backups').rglob('*.srt'))[0].read_text() == 'broken old translation'
    assert not list(flow.cache.glob('ocr-*tmp*'))


def test_automatic_english_translation_recovers_ocr_for_new_target(repair_flow):
    flow, jobs, media, target, reference, complete, calls = repair_flow
    flow.cfg.ocr_enabled = True
    target.unlink()
    job = run(flow, jobs, 'embedded_translate')
    assert job.state == 'done'
    assert len(calls) == 1
    assert spans(target.read_text()) == spans(complete)
    assert len(flow.service.ollama.source_cues) == len(parse(complete))
    assert all('SAM:' in cue.text for cue in flow.service.ollama.source_cues)


@pytest.mark.parametrize('existing', [False, True])
def test_automatic_ocr_failure_never_installs_incomplete_translation(repair_flow, monkeypatch, existing):
    from subzero.worker import syncflow
    flow, jobs, media, target, reference, complete, calls = repair_flow
    flow.cfg.ocr_enabled = True
    if not existing:
        target.unlink()

    def fail(*args, **kwargs):
        raise RuntimeError('Apple Vision unavailable')

    monkeypatch.setattr(syncflow, 'fill_subtitle_gaps', fail)
    job = run(flow, jobs, 'embedded_translate')
    assert job.state == 'needs_review'
    assert not getattr(flow.service.ollama, 'source_cues', [])
    assert target.read_text() == 'broken old translation' if existing else not target.exists()


def test_automatic_english_ocr_can_be_disabled(repair_flow):
    flow, jobs, media, target, reference, complete, calls = repair_flow
    flow.cfg.ocr_enabled = False
    target.unlink()
    assert run(flow, jobs, 'embedded_translate').state == 'done'
    assert not calls
    assert spans(target.read_text()) == spans(reference['text'])


def test_automatic_english_ocr_obeys_audit_only(repair_flow):
    flow, jobs, media, target, reference, complete, calls = repair_flow
    flow.cfg.ocr_enabled = True
    flow.cfg.sync_audit_only = True
    assert run(flow, jobs, 'embedded_translate').state == 'needs_review'
    assert not calls
    assert target.read_text() == 'broken old translation'


def test_automatic_ocr_does_not_replace_target_created_during_generation(repair_flow, monkeypatch):
    from subzero.worker import syncflow
    flow, jobs, media, target, reference, complete, calls = repair_flow
    flow.cfg.ocr_enabled = True
    target.unlink()
    recover = syncflow.fill_subtitle_gaps

    def create_target(*args, **kwargs):
        recovered = recover(*args, **kwargs)
        target.write_text('Subtitle created while OCR was running')
        return recovered

    monkeypatch.setattr(syncflow, 'fill_subtitle_gaps', create_target)
    assert run(flow, jobs, 'embedded_translate').state == 'needs_review'
    assert target.read_text() == 'Subtitle created while OCR was running'


def test_new_target_install_rejects_creation_after_absence_check(setup, monkeypatch):
    from subzero.worker import syncflow
    from subzero.timing import Report
    flow, jobs, _, media = setup
    job = jobs.enqueue('id', 'repair', 'pt-BR')
    jobs.start(job.id)
    target = Path(media.path).with_suffix('.pt-BR.srt')
    original = 'Subtitle created during installation'
    lexists = syncflow.os.path.lexists

    def create_after_check(path):
        exists = lexists(path)
        if Path(path) == target:
            target.write_text(original)
        return exists

    monkeypatch.setattr(syncflow.os.path, 'lexists', create_after_check)
    with pytest.raises(RuntimeError, match='Subtitle changed'):
        flow.install(media, job, syncflow.fingerprint(media.path),
                     dialogue().replace('Test dialogue', 'Fala traduzida.'), Report('pass', 'test'),
                     expected_source=(target, None))
    assert target.read_text() == original
    assert not list(target.parent.glob('.subtitle-*.tmp'))


def test_new_target_install_fails_safely_when_exclusive_creation_is_unsupported(setup, monkeypatch):
    import errno
    from subzero.worker import syncflow
    from subzero.timing import Report
    flow, jobs, _, media = setup
    job = jobs.enqueue('id', 'repair', 'pt-BR')
    jobs.start(job.id)
    target = Path(media.path).with_suffix('.pt-BR.srt')

    def unsupported(*args, **kwargs):
        raise OSError(errno.EOPNOTSUPP, 'Operation not supported')

    monkeypatch.setattr(syncflow.os, 'link', unsupported)
    with pytest.raises(RuntimeError, match='Cannot create subtitle atomically'):
        flow.install(media, job, syncflow.fingerprint(media.path),
                     dialogue().replace('Test dialogue', 'Fala traduzida.'), Report('pass', 'test'),
                     expected_source=(target, None))
    assert not target.exists()
    assert not list(target.parent.glob('.subtitle-*.tmp'))


def test_repair_reuses_completed_ocr_source_after_translation_failure(repair_flow, monkeypatch):
    from subzero.worker import syncflow
    flow, jobs, media, target, reference, complete, calls = repair_flow
    flow.service.ollama.failure = 'Translation unavailable'
    first = run(flow, jobs, 'repair')
    assert first.state == 'needs_review'
    assert target.read_text() == 'broken old translation'
    flow.service.ollama.failure = None

    def no_rescan(*args, **kwargs):
        raise AssertionError('A translation retry must reuse verified English OCR')

    monkeypatch.setattr(syncflow, 'fill_subtitle_gaps', no_rescan)
    second = run(flow, jobs, 'repair')
    assert second.state == 'done'
    assert spans(target.read_text()) == spans(complete)


def test_cpu_provider_cache_is_pinned_and_does_not_start_ollama(repair_flow, monkeypatch):
    flow, jobs, media, target, reference, complete, calls = repair_flow
    client = flow.service.ollama
    client.provider = 'libretranslate'
    client.model = 'argos-en-pb'
    client.needs_local_compute = False
    client.uses_sentence_units = True
    client.supports_context = False
    client.cache_settings = {'package_sha256': 'a' * 64}

    def no_gpu(*args, **kwargs):
        raise AssertionError('CPU translation must not acquire an Ollama phase')

    monkeypatch.setattr('subzero.worker.syncflow.compute_phase', no_gpu)
    monkeypatch.setattr('subzero.worker.tracks.compute_phase', no_gpu)
    assert run(flow, jobs, 'repair').state == 'done'
    translated_count = len(client.source_cues)
    assert translated_count == len(parse(complete))
    assert run(flow, jobs, 'repair').state == 'done'
    assert len(client.source_cues) == translated_count
    client.cache_settings = {'package_sha256': 'b' * 64}
    assert run(flow, jobs, 'repair').state == 'done'
    assert len(client.source_cues) == 2 * translated_count
    assert len(calls) == 1


def test_gap_recovery_uses_selected_translation_client(gap_recovery, monkeypatch):
    from subzero.worker import syncflow
    flow, jobs, media, target, original, complete = gap_recovery
    recover = syncflow.fill_subtitle_gaps
    clients = []

    def scan(*args, **kwargs):
        clients.append(kwargs['translation_client'])
        assert 'model' not in kwargs and 'provider' not in kwargs
        return recover(*args, **kwargs)

    monkeypatch.setattr(syncflow, 'fill_subtitle_gaps', scan)
    assert run(flow, jobs, 'recover_gaps').state == 'done'
    assert clients == [flow.service.ollama]


@pytest.mark.parametrize('change', ['source', 'cache'])
def test_repair_does_not_reuse_changed_or_corrupt_ocr_source(repair_flow, change):
    flow, jobs, media, target, reference, complete, calls = repair_flow
    assert run(flow, jobs, 'repair').state == 'done'
    if change == 'source':
        reference['text'] = reference['text'].replace('shelter', 'camp')
    else:
        cache = next((flow.cache / 'ocr-sources').rglob('*.json'))
        cache.write_text('{"text":"tampered","digest":"invalid"}')
    assert run(flow, jobs, 'repair').state == 'done'
    assert len(calls) == 2


def test_ocr_rescue_config_pins_source_cache_and_reaches_scanner(repair_flow, monkeypatch):
    from subzero.worker import syncflow
    flow, jobs, media, target, reference, complete, calls = repair_flow
    assert run(flow, jobs, 'repair').state == 'done'
    flow.cfg.ocr_rescue_model = 'qwen3.5:9b'
    flow.cfg.ocr_rescue_model_digest = 'a' * 64
    flow.cfg.ollama_url = 'http://127.0.0.1:11434'
    rescue_flow = syncflow.SyncFlow(jobs, flow.service, flow.cfg, reference_builder=flow.reference_builder)
    flow.service.sync_flow = rescue_flow
    recover = syncflow.fill_subtitle_gaps
    seen = []

    def scan(*args, **kwargs):
        seen.append(kwargs['caption_rescue'].identity)
        return recover(*args, **kwargs)

    monkeypatch.setattr(syncflow, 'fill_subtitle_gaps', scan)
    assert run(rescue_flow, jobs, 'repair').state == 'done'
    assert len(calls) == 2
    assert run(rescue_flow, jobs, 'repair').state == 'done'
    assert len(calls) == 2
    flow.cfg.ocr_rescue_model_digest = 'b' * 64
    changed_flow = syncflow.SyncFlow(jobs, flow.service, flow.cfg, reference_builder=flow.reference_builder)
    flow.service.sync_flow = changed_flow
    assert run(changed_flow, jobs, 'repair').state == 'done'
    assert len(calls) == 3
    assert len(set(seen)) == 2


@pytest.mark.parametrize('failure', ['ocr', 'translation', 'timing', 'cancel', 'target_changed'])
def test_repair_failure_never_installs_partial_translation(repair_flow, monkeypatch, failure):
    from subzero.worker import syncflow
    flow, jobs, media, target, reference, complete, calls = repair_flow
    if failure == 'ocr':
        def broken_ocr(*args, **kwargs):
            raise RuntimeError('Apple Vision failed')
        monkeypatch.setattr(syncflow, 'fill_subtitle_gaps', broken_ocr)
    elif failure == 'translation':
        flow.service.ollama.failure = 'invalid translation lines'
    elif failure == 'timing':
        monkeypatch.setattr(syncflow, 'translate', lambda *args, **kwargs: parse(dialogue(8)))
    elif failure == 'target_changed':
        def edit_target(*args, **kwargs):
            target.write_text('newer subtitle from another task')
            return parse(dialogue().replace('Test dialogue', 'Fala traduzida'))
        monkeypatch.setattr(syncflow, 'translate', edit_target)
    job = jobs.enqueue('id', 'repair', 'pt-BR')
    if failure == 'cancel':
        translate = flow.service.ollama.translate_block
        def cancelled_translation(*args, **kwargs):
            jobs.cancel(job.id)
            return translate(*args, **kwargs)
        flow.service.ollama.translate_block = cancelled_translation
    Runner(jobs, {'repair': flow.run}).run_once()
    final = jobs.get(job.id)
    assert final.state == ('cancelled' if failure == 'cancel' else 'needs_review')
    assert final.kind == 'repair'
    assert target.read_text() == ('newer subtitle from another task' if failure == 'target_changed'
                                  else 'broken old translation')
    assert not list((flow.cache / 'backups').rglob('*.srt'))


def test_repair_requires_verified_english_without_transcription_fallback(repair_flow, monkeypatch):
    from subzero.worker import syncflow
    flow, jobs, media, target, reference, complete, calls = repair_flow
    reference['text'] = ''
    def no_audio(*args):
        raise AssertionError('Repair must not transcribe without verified English')
    monkeypatch.setattr(syncflow, 'transcribe', no_audio)
    job = run(flow, jobs, 'repair')
    assert job.state == 'needs_review'
    assert job.kind == 'repair'
    assert not calls
    assert target.read_text() == 'broken old translation'


def test_repair_accepts_verified_english_sidecar(repair_flow):
    flow, jobs, media, target, reference, complete, calls = repair_flow
    Path(media.path).with_suffix('.en.srt').write_text(reference['text'])
    reference['language'] = 'jpn'
    job = run(flow, jobs, 'repair')
    assert job.state == 'done'
    assert spans(target.read_text()) == spans(complete)


def test_repair_translates_source_when_ocr_has_no_new_captions(repair_flow, monkeypatch):
    from subzero.worker import syncflow
    flow, jobs, media, target, reference, complete, calls = repair_flow
    monkeypatch.setattr(syncflow, 'fill_subtitle_gaps', lambda *args, **kwargs: SimpleNamespace(cues_recovered=0))
    job = run(flow, jobs, 'repair')
    assert job.state == 'done'
    assert spans(target.read_text()) == spans(reference['text'])


def test_repair_omits_pure_sound_cues_before_ocr_and_preserves_dialogue(repair_flow, monkeypatch):
    from subzero.worker import syncflow
    flow, jobs, media, target, reference, complete, calls = repair_flow
    dialogue_source = reference['text']
    missing = parse(complete)[10]
    sound = [Cue(0, missing.start, missing.end, '♪ dramatic music ♪'), Cue(0, 2, 4, '[waves crashing]')]
    reference['text'] = dump(sorted(parse(dialogue_source) + sound, key=lambda cue: cue.start))
    scanned = []

    def recover(video, subtitle_path, **kwargs):
        scanned.append(Path(subtitle_path).read_text())
        Path(kwargs['output']).write_text(complete)
        return SimpleNamespace(cues_recovered=1)

    monkeypatch.setattr(syncflow, 'fill_subtitle_gaps', recover)
    job = run(flow, jobs, 'repair')
    assert job.state == 'done'
    assert scanned == [dialogue_source]
    assert all('SAM:' in cue.text for cue in flow.service.ollama.source_cues)
    assert spans(target.read_text()) == spans(complete)


def test_repair_omits_multiline_sound_cues_before_ocr_and_preserves_dialogue(repair_flow, monkeypatch):
    from subzero.worker import syncflow
    flow, jobs, media, target, reference, complete, calls = repair_flow
    dialogue_source = reference['text']
    missing = parse(complete)[10]
    sound = [
        Cue(0, missing.start, missing.end, '[dramatic orchestral music\nplaying faintly]'),
        Cue(0, 2, 4, '(waves crashing\non the distant shore)'),
    ]
    reference['text'] = dump(sorted(parse(dialogue_source) + sound, key=lambda cue: cue.start))
    scanned = []

    def recover(video, subtitle_path, **kwargs):
        scanned.append(Path(subtitle_path).read_text())
        Path(kwargs['output']).write_text(complete)
        return SimpleNamespace(cues_recovered=1)

    monkeypatch.setattr(syncflow, 'fill_subtitle_gaps', recover)
    job = run(flow, jobs, 'repair')
    assert job.state == 'done'
    assert scanned == [dialogue_source]
    assert all('SAM:' in cue.text for cue in flow.service.ollama.source_cues)
    assert spans(target.read_text()) == spans(complete)



def test_repair_audit_only_never_regenerates(repair_flow):
    flow, jobs, media, target, reference, complete, calls = repair_flow
    flow.cfg.sync_audit_only = True
    job = run(flow, jobs, 'repair')
    assert job.state == 'needs_review'
    assert not calls
    assert target.read_text() == 'broken old translation'


def test_repair_stops_before_ocr_when_cancelled_on_progress(repair_flow):
    flow, jobs, media, target, reference, complete, calls = repair_flow
    job = jobs.enqueue('id', 'repair', 'pt-BR')
    jobs.start(job.id)
    def cancel(phase, percent):
        if phase.startswith('recuperando'):
            jobs.cancel(job.id)
    with pytest.raises(RuntimeError, match='cancel'):
        flow.run(job, cancel)
    assert not calls
    assert target.read_text() == 'broken old translation'


def test_repair_rejects_translation_that_turns_dialogue_into_sdh(repair_flow, monkeypatch):
    from subzero.worker import syncflow
    flow, jobs, media, target, reference, complete, calls = repair_flow
    def lose_dialogue(cues, *args, **kwargs):
        translated = [Cue(c.index, c.start, c.end, 'Fala traduzida') for c in cues]
        translated[10].text = '[risos]'
        return translated
    monkeypatch.setattr(syncflow, 'translate', lose_dialogue)
    job = run(flow, jobs, 'repair')
    assert job.state == 'needs_review'
    assert target.read_text() == 'broken old translation'
    assert not list((flow.cache / 'translations').rglob('*.json'))


def test_repair_resumes_translation_after_later_block_failure(repair_flow, monkeypatch):
    from subzero.worker import syncflow
    flow, jobs, media, target, reference, complete, calls = repair_flow
    requested = []
    fail = True
    def translate_block(cues, *args, **kwargs):
        requested.append(cues[0].index)
        if fail and len(requested) > 1:
            raise RuntimeError('Translation interrupted after first block')
        return [Cue(c.index, c.start, c.end, 'Fala traduzida') for c in cues]
    monkeypatch.setattr(syncflow, 'translate', translate_block)
    first = run(flow, jobs, 'repair')
    assert first.state == 'needs_review'
    assert target.read_text() == 'broken old translation'
    assert list((flow.cache / 'translations').rglob('*.json'))
    assert flow.service.ollama.releases == 1
    fail = False
    requested.clear()
    second = run(flow, jobs, 'repair')
    assert second.state == 'done'
    assert requested[0] == 21
    assert spans(target.read_text()) == spans(complete)
    assert flow.service.ollama.releases == 2


def test_repair_admits_whole_episode_before_any_translation(repair_flow):
    from subzero.worker.deepl import DeepLQuotaExceeded

    flow, jobs, media, target, reference, complete, calls = repair_flow
    client = flow.service.ollama
    client.source_cues = []
    admissions = []
    client.count_episode = lambda cues: sum(len(c.text) for c in cues)

    def quota(required):
        admissions.append(required)
        raise DeepLQuotaExceeded(required, 1)

    client.check_quota = quota
    paused = run(flow, jobs, 'repair')
    assert paused.state == 'paused'
    assert admissions == [client.count_episode(parse(complete))]
    assert not client.source_cues
    assert target.read_text() == 'broken old translation'
    assert not list((flow.cache / 'translations').rglob('*.json'))


def test_quota_resume_only_admits_uncached_blocks(repair_flow):
    from subzero.worker.deepl import DeepLQuotaExceeded

    flow, jobs, media, target, reference, complete, calls = repair_flow
    client = flow.service.ollama
    client.source_cues = []
    admissions = []
    client.count_episode = lambda cues: sum(len(c.text) for c in cues)
    client.check_quota = admissions.append
    original = client.translate_block

    def interrupted(cues, *args, **kwargs):
        if client.source_cues:
            raise DeepLQuotaExceeded(100, 0)
        return original(cues, *args, **kwargs)

    client.translate_block = interrupted
    paused = run(flow, jobs, 'repair')
    assert paused.state == 'paused'
    cached_count = len(client.source_cues)
    assert cached_count == 20
    assert target.read_text() == 'broken old translation'
    client.translate_block = original
    jobs.resume(paused.id)
    Runner(jobs, {'repair': flow.run}).run_once()
    assert jobs.get(paused.id).state == 'done'
    assert admissions == [client.count_episode(parse(complete)),
                          client.count_episode(parse(complete)[cached_count:])]
    assert len(client.source_cues) == len(parse(complete))


def test_repair_resumes_translation_after_cancellation(repair_flow):
    flow, jobs, media, target, reference, complete, calls = repair_flow
    job = jobs.enqueue('id', 'repair', 'pt-BR')
    jobs.start(job.id)
    def cancel_after_block(phase, percent):
        if phase.startswith('traduzindo legendas 20/'):
            jobs.cancel(job.id)
    with pytest.raises(RuntimeError, match='cancel'):
        flow.run(job, cancel_after_block)
    assert target.read_text() == 'broken old translation'
    first_cues = list(flow.service.ollama.source_cues)
    assert len(first_cues) == 20
    flow.service.ollama.source_cues = []
    assert run(flow, jobs, 'repair').state == 'done'
    assert flow.service.ollama.source_cues[0].index == 21


@pytest.mark.parametrize('change', ['model', 'config', 'prompt', 'title', 'corrupt', 'wrong_timing', 'english', 'empty'])
def test_repair_translation_cache_rejects_stale_or_corrupt_blocks(repair_flow, monkeypatch, change):
    import json
    from subzero.worker import syncflow
    flow, jobs, media, target, reference, complete, calls = repair_flow
    assert run(flow, jobs, 'repair').state == 'done'
    flow.service.ollama.source_cues = []
    if change == 'model':
        flow.service.ollama.model = 'different-model:12b'
    elif change == 'config':
        flow.service.ollama.num_ctx = 8192
    elif change == 'prompt':
        monkeypatch.setattr(syncflow, 'TRANSLATION_PROMPT_VERSION', 'next-prompt')
    elif change == 'title':
        media.series_name = 'Another programme'
    else:
        cache = sorted((flow.cache / 'translations').rglob('*.json'))[0]
        cached = json.loads(cache.read_text())
        if change == 'corrupt':
            cached['source'] = 'changed'
        elif change == 'wrong_timing':
            cached['text'] = dump(parse(dialogue(8))[:20])
        elif change == 'english':
            cached['text'] = cached['source']
        else:
            cached['text'] = ''
        cached['digest'] = syncflow.digest(cached['text'])
        cache.write_text(json.dumps(cached))
    assert run(flow, jobs, 'repair').state == 'done'
    translated = flow.service.ollama.source_cues
    assert len(translated) == (len(parse(complete)) if change in ('model', 'config', 'prompt', 'title') else 20)


@pytest.mark.parametrize('series_name', ['Survivor', ''])
def test_repair_keeps_only_prior_source_context_across_block_boundaries(repair_flow, monkeypatch, series_name):
    from subzero.worker import syncflow
    flow, jobs, media, target, reference, complete, calls = repair_flow
    flow.service.ollama.model = 'kaelri/hy-mt2:7b'
    media.series_name = series_name
    requested = []

    def capture(cues, *args, **kwargs):
        requested.append((cues, kwargs.get('context')))
        return [Cue(c.index, c.start, c.end, 'Fala traduzida') for c in cues]

    monkeypatch.setattr(syncflow, 'translate', capture)
    assert run(flow, jobs, 'repair').state == 'done'
    cues = parse(complete)
    context = requested[2][1]
    assert context == {'title': series_name or media.name,
                       'previous_cues': [c.text for c in cues[8:40]]}


def test_repair_bounds_long_background_and_excludes_current_block(repair_flow, monkeypatch):
    from subzero.worker import syncflow
    flow, jobs, media, target, reference, complete, calls = repair_flow
    flow.service.ollama.model = 'hy-mt2:7b'
    requested = []

    def capture(cues, *args, **kwargs):
        requested.append(kwargs['context'])
        return [Cue(c.index, c.start, c.end, 'Fala traduzida') for c in cues]

    monkeypatch.setattr(syncflow, 'translate', capture)
    cues = [Cue(i, i * 2, i * 2 + 1, f'Source cue {i}: ' + 'x' * 150) for i in range(80)]
    job = jobs.enqueue('id', 'repair', 'pt-BR')
    jobs.start(job.id)
    flow.repair_translation(job, 'video', cues, lambda *args: None, title='S' * 1000)
    assert all(len('\n'.join(c['previous_cues'])) <= 6000 and len(c['title']) <= 256 for c in requested)
    background = '\n'.join(requested[2]['previous_cues'])
    assert all(c.text not in background for c in cues[40:])
    assert cues[39].text in background
    assert requested[0]['previous_cues'] == []


def test_repair_cache_resumes_at_complete_sentence_boundary(repair_flow, monkeypatch):
    from subzero.worker import syncflow
    flow, jobs, media, target, reference, complete, calls = repair_flow
    flow.service.ollama.model = 'hy-mt2:7b'
    cues = [Cue(i, i, i + .9, f'Complete sentence {i}.') for i in range(19)]
    cues += [Cue(19, 19, 19.9, 'I spent'), Cue(20, 20, 20.9, 'five years in foster care.')]
    cues += [Cue(i, i, i + .9, f'Complete sentence {i}.') for i in range(21, 45)]
    requested = []

    def capture(block, *args, **kwargs):
        requested.append(block)
        if len(requested) > 1:
            raise RuntimeError('Later block unavailable')
        return [Cue(c.index, c.start, c.end, 'Fala traduzida.') for c in block]

    monkeypatch.setattr(syncflow, 'translate', capture)
    job = jobs.enqueue('id', 'repair', 'pt-BR')
    jobs.start(job.id)
    with pytest.raises(RuntimeError, match='Later block'):
        flow.repair_translation(job, 'video', cues, lambda *args: None)
    first_end = len(requested[0])
    assert first_end != 20
    requested.clear()

    def resumed(block, *args, **kwargs):
        requested.append(block)
        return [Cue(c.index, c.start, c.end, 'Fala traduzida.') for c in block]

    monkeypatch.setattr(syncflow, 'translate', resumed)
    translated = flow.repair_translation(job, 'video', cues, lambda *args: None)
    assert requested[0][0].index == first_end
    assert [(c.index, c.start, c.end) for c in translated] == [(c.index, c.start, c.end) for c in cues]
    assert any(cues[19] in block and cues[20] in block for block in requested) if first_end == 19 else first_end == 21


def test_repair_does_not_cache_untranslated_english(repair_flow, monkeypatch):
    from subzero.worker import syncflow
    flow, jobs, media, target, reference, complete, calls = repair_flow
    monkeypatch.setattr(syncflow, 'translate', lambda cues, *args, **kwargs: cues)
    job = run(flow, jobs, 'repair')
    assert job.state == 'needs_review'
    assert target.read_text() == 'broken old translation'
    assert not list((flow.cache / 'translations').rglob('*.json'))


def test_repair_keeps_verified_anchors_when_ocr_fills_audio_gaps(repair_flow, monkeypatch):
    from subzero.ocr import uncovered_intervals
    from subzero.reference import verify_text
    from subzero.worker import syncflow
    flow, jobs, media, target, reference, complete, calls = repair_flow
    gaps = uncovered_intervals([(0, 940)], spans(reference['text']))
    extra = [Cue(0, start, end, 'Quiet words on screen.') for start, end in gaps]
    augmented = dump(sorted(parse(reference['text']) + extra, key=lambda cue: cue.start))
    assert verify_text(reference['text'], reference).status == 'pass'
    assert verify_text(augmented, reference).status != 'pass'
    def recover(video, subtitle_path, **kwargs):
        Path(kwargs['output']).write_text(augmented)
        return SimpleNamespace(cues_recovered=len(extra))
    monkeypatch.setattr(syncflow, 'fill_subtitle_gaps', recover)
    monkeypatch.setattr(syncflow, 'translate', lambda cues, *args, **kwargs: [
        Cue(c.index, c.start, c.end, 'Palavras discretas na tela.' if c.text == 'Quiet words on screen.'
            else 'Precisamos construir um abrigo.') for c in cues])
    job = run(flow, jobs, 'repair')
    assert job.state == 'done'
    assert spans(target.read_text()) == spans(augmented)


@pytest.mark.parametrize('fault', ['anchor_text', 'anchor_timing', 'overlap', 'range', 'order'])
def test_repair_rejects_and_retains_invalid_ocr_source(repair_flow, monkeypatch, fault):
    from subzero.worker import syncflow
    flow, jobs, media, target, reference, complete, calls = repair_flow
    cues = parse(complete)
    if fault == 'anchor_text':
        cues[0].text = 'Changed original dialogue'
    elif fault == 'anchor_timing':
        cues[0].start += .2
    elif fault == 'overlap':
        cues.insert(1, Cue(0, cues[0].start + .2, cues[0].end, 'First recovered caption'))
        cues.insert(2, Cue(0, cues[0].start + .3, cues[0].end, 'Second recovered caption'))
    elif fault == 'range':
        cues.append(Cue(0, 950, 952, 'Beyond the video'))
    else:
        cues[0], cues[1] = cues[1], cues[0]
    invalid = dump(cues)
    def recover(video, subtitle_path, **kwargs):
        Path(kwargs['output']).write_text(invalid)
        return SimpleNamespace(cues_recovered=1)
    monkeypatch.setattr(syncflow, 'fill_subtitle_gaps', recover)
    job = run(flow, jobs, 'repair')
    assert job.state == 'needs_review'
    assert target.read_text() == 'broken old translation'
    assert any(p.read_text() == invalid for p in (flow.cache / 'candidates').glob('*/en/*.srt'))
    assert list((flow.cache / 'ocr-validation').rglob('*.json'))


def test_ocr_source_validation_rejects_overlap_with_retained_music(repair_flow):
    from subzero.worker.syncflow import validate_ocr_source
    flow, jobs, media, target, reference, complete, calls = repair_flow
    source = dump([Cue(0, 2, 5, '♪ music ♪')] + parse(reference['text']))
    augmented = dump(sorted(parse(source) + [Cue(0, 3, 4, 'Overlapping caption')], key=lambda c: c.start))
    with pytest.raises(RuntimeError, match='overlaps'):
        validate_ocr_source(source, augmented, reference)


@pytest.mark.parametrize('cached', [False, True])
def test_repair_rejects_unstable_ocr_before_translation_and_keeps_review_source(repair_flow, monkeypatch, cached):
    import json
    from subzero.worker import syncflow
    flow, jobs, media, target, reference, complete, calls = repair_flow
    original = parse(reference['text'])
    extra = [Cue(0, 1, 1.7, 'I found the second key.'),
             Cue(0, 1.7, 1.8, '1 found the second key.'),
             Cue(0, 1.8, 2.5, 'I found the second key.')]
    invalid = dump(sorted(original + extra, key=lambda cue: cue.start))
    if cached:
        flow.service.ollama.failure = 'Translation unavailable'
        assert run(flow, jobs, 'repair').state == 'needs_review'
        cache = next((flow.cache / 'ocr-sources').rglob('*.json'))
        cache.write_text(json.dumps({'text': invalid, 'digest': syncflow.digest(invalid)}))
    def recover(video, subtitle_path, **kwargs):
        Path(kwargs['output']).write_text(invalid)
        return SimpleNamespace(cues_recovered=len(extra))
    def no_translation(*args, **kwargs):
        raise AssertionError('Unstable OCR must not reach translation')
    monkeypatch.setattr(syncflow, 'fill_subtitle_gaps', recover)
    monkeypatch.setattr(syncflow, 'translate', no_translation)
    job = run(flow, jobs, 'repair')
    assert job.state == 'needs_review'
    assert 'Unstable OCR caption readings' in job.message
    assert target.read_text() == 'broken old translation'
    assert any(path.read_text() == invalid for path in (flow.cache / 'candidates').glob('*/en/*.srt'))


def test_repair_translates_simultaneous_captions_once_then_composes_display(repair_flow, monkeypatch):
    from subzero.worker import syncflow
    flow, jobs, media, target, reference, complete, calls = repair_flow
    original = parse(reference['text'])
    first = original[0]
    extra = Cue(0, round(first.start + .2, 3), round(first.end - .2, 3), 'I also like Rome.')
    augmented = dump(sorted(original + [extra], key=lambda cue: cue.start))

    def recover(video, subtitle_path, **kwargs):
        assert kwargs['all_captions'] is True
        Path(kwargs['output']).write_text(augmented)
        return SimpleNamespace(cues_recovered=1)

    seen = []

    def translate(cues, *args, **kwargs):
        seen.extend(cues)
        return [Cue(c.index, c.start, c.end,
                    'Eu também gosto do Rome.' if c.text == extra.text else 'Precisamos construir um abrigo.')
                for c in cues]

    monkeypatch.setattr(syncflow, 'fill_subtitle_gaps', recover)
    monkeypatch.setattr(syncflow, 'translate', translate)
    job = run(flow, jobs, 'repair')
    assert job.state == 'done', job.message
    assert sum(c.text == extra.text for c in seen) == 1
    assert len(seen) == len(original) + 1
    displayed = parse(target.read_text(encoding='utf-8'))
    assert all(left.end <= right.start for left, right in zip(displayed, displayed[1:]))
    together = [c for c in displayed if c.start <= extra.start and c.end >= extra.end]
    assert len(together) == 1
    assert 'Precisamos construir um abrigo.' in together[0].text
    assert 'Eu também gosto do Rome.' in together[0].text
    assert displayed[0].start == first.start
    assert displayed[2].end == first.end


def test_repair_keeps_identical_responses_from_simultaneous_speakers(repair_flow, monkeypatch):
    from subzero.worker import syncflow
    flow, jobs, media, target, reference, complete, calls = repair_flow
    original = parse(reference['text'])
    first = original[0]
    extra = Cue(0, round(first.start + .2, 3), round(first.end - .2, 3), 'ROME: Yes.')
    augmented = dump(sorted(original + [extra], key=lambda cue: cue.start))

    def recover(video, subtitle_path, **kwargs):
        Path(kwargs['output']).write_text(augmented)
        return SimpleNamespace(cues_recovered=1)

    def translate(cues, *args, **kwargs):
        return [Cue(c.index, c.start, c.end, 'ROME: Sim.' if c.text == extra.text
                    else 'ANA: Sim.' if c.start == first.start else 'Precisamos construir um abrigo.')
                for c in cues]

    monkeypatch.setattr(syncflow, 'fill_subtitle_gaps', recover)
    monkeypatch.setattr(syncflow, 'translate', translate)
    job = run(flow, jobs, 'repair')
    assert job.state == 'done', job.message
    together = next(c for c in parse(target.read_text()) if c.start == extra.start)
    assert together.text.count('Sim.') == 2
    assert len(together.text.splitlines()) == 2


@pytest.mark.parametrize('existed', [False, True])
def test_repair_closes_staging_files_before_atomic_install(repair_flow, monkeypatch, existed):
    from subzero.worker import syncflow

    flow, jobs, media, target, _, _, _ = repair_flow
    flow.cfg.ocr_enabled = True
    if not existed:
        target.unlink()
    create = syncflow.tempfile.NamedTemporaryFile
    replace = syncflow.os.replace
    link = syncflow.os.link
    handles = []
    published = []

    def staging_file(*args, **kwargs):
        handle = create(*args, **kwargs)
        handles.append(handle)
        return handle

    def publish(operation, source, destination):
        assert all(handle.closed for handle in handles if Path(handle.name) == Path(source))
        published.append(Path(destination))
        return operation(source, destination)

    monkeypatch.setattr(syncflow.tempfile, 'NamedTemporaryFile', staging_file)
    monkeypatch.setattr(syncflow.os, 'replace', lambda a, b: publish(replace, a, b))
    monkeypatch.setattr(syncflow.os, 'link', lambda a, b: publish(link, a, b))
    job = run(flow, jobs, 'embedded_translate')
    assert job.state == 'done', job.message
    assert any('ocr-sources' in str(path) for path in published)
    assert any('translations' in str(path) for path in published)
    assert target in published
    assert not list(flow.cache.rglob('*.tmp'))
    assert not list(target.parent.glob('.subtitle-*.tmp'))


def test_stage_keeps_existing_crlf_bytes(setup, monkeypatch):
    flow, _, _, _ = setup
    write = Path.write_text

    def windows_write(path, text, encoding=None, errors=None, newline=None):
        return write(path, text, encoding=encoding, errors=errors,
                     newline='\r\n' if newline is None else newline)

    monkeypatch.setattr(Path, 'write_text', windows_write)
    text = '1\r\n00:00:01,000 --> 00:00:02,000\r\nUma fala.\r\n'
    assert flow.stage('video', 'pt-BR', text).read_bytes() == text.encode('utf-8')


def test_rebuild_searches_opensubtitles_before_whisper(setup, monkeypatch):
    flow, jobs, provider, media = setup
    flow.cfg.opensubtitles_key = 'test-key'
    whisper_called = []
    monkeypatch.setattr('subzero.worker.syncflow.transcribe', lambda *a, **k: whisper_called.append(True) or ([], 'en'))
    provider.search = lambda **k: [Candidate(99, 'movie.1080p', 'en', 500, False, True, False)]
    provider.download = lambda fid: dialogue()
    flow.service.opensubs = provider
    flow.translation_ready = lambda: None
    flow.translate_cues = lambda cues, *a, **k: cues
    job = run(flow, jobs, 'rebuild')
    assert job.state == 'done', job.message
    assert not whisper_called, 'Whisper was called despite verified OpenSubtitles subtitle'
    assert Path(job.result_path).read_text() == dialogue()


def test_rebuild_shifts_offset_opensubtitles_instead_of_whisper(setup, monkeypatch):
    flow, jobs, provider, media = setup
    flow.cfg.opensubtitles_key = 'test-key'
    whisper_called = []
    monkeypatch.setattr('subzero.worker.syncflow.transcribe', lambda *a, **k: whisper_called.append(True) or ([], 'en'))
    provider.search = lambda **k: [Candidate(99, 'movie.1080p', 'en', 500, False, True, False)]
    provider.download = lambda fid: dialogue(7.5)
    flow.service.opensubs = provider
    flow.translation_ready = lambda: None
    flow.translate_cues = lambda cues, *a, **k: cues
    job = run(flow, jobs, 'rebuild')
    assert job.state == 'done', job.message
    assert not whisper_called, 'Whisper was called despite shiftable OpenSubtitles subtitle'


def test_repair_searches_opensubtitles_when_sidecar_missing(setup):
    flow, jobs, provider, media = setup
    flow.cfg.opensubtitles_key = 'test-key'
    installed = Path(flow.installed(media, 'pt-BR'))
    installed.write_text(dialogue())
    provider.search = lambda **k: [Candidate(99, 'movie.1080p', 'en', 500, False, True, False)]
    provider.download = lambda fid: dialogue()
    flow.service.opensubs = provider
    flow.generate_from_english = lambda media, job, key, ref, source, prog: (dialogue(), Report('pass', 'ok', []))
    ref = flow.reference_builder()
    ref['text'] = ''
    job = run(flow, jobs, 'repair')
    assert job.state == 'done', job.message
    assert Path(job.result_path).read_text() == dialogue()


def test_prune_sidecars_removes_english_after_pt_install(setup, tmp_path):
    flow, _, _, _ = setup
    video = tmp_path / 'movie.mkv'
    en = tmp_path / 'movie.en.srt'
    eng = tmp_path / 'movie.eng.srt'
    target = tmp_path / 'movie.pt-BR.srt'
    en.write_text('english'); eng.write_text('english'); target.write_text('portugues')
    media = SimpleNamespace(path=str(video), embedded=[])
    flow.prune_sidecars(media, target, 'pt-BR')
    assert not en.exists() and not eng.exists()
    assert target.exists()


def test_prune_sidecars_keeps_english_for_english_target(setup, tmp_path):
    flow, _, _, _ = setup
    video = tmp_path / 'movie.mkv'
    target = tmp_path / 'movie.en.srt'
    other = tmp_path / 'movie.eng.srt'
    target.write_text('english'); other.write_text('english')
    media = SimpleNamespace(path=str(video), embedded=[])
    flow.prune_sidecars(media, target, 'en')
    assert target.exists() and other.exists()


def test_source_sidecar_returns_stripped_text(setup):
    flow,jobs,provider,media=setup
    source=Path(media.path).with_suffix('.en.srt')
    source.write_text(dialogue()+'\n5000\n01:23:20,000 --> 01:23:22,000\n[Thunder rumbling]\n')
    text,tag=flow.source_sidecar(media,flow.reference_builder())
    assert tag=='en'
    assert '[Thunder rumbling]' not in text
    assert 'Fala traduzida' not in text


def test_source_sidecar_skips_sdh_only_file(setup):
    flow,jobs,provider,media=setup
    source=Path(media.path).with_suffix('.en.srt')
    source.write_text('1\n00:00:01,000 --> 00:00:03,000\n[Music playing]\n')
    assert flow.source_sidecar(media,flow.reference_builder()) is None


def test_audit_installs_embedded_subtitle_when_sidecar_missing(setup):
    flow,jobs,provider,media=setup
    from subzero.worker.jellyfin import EmbeddedSub
    media.embedded = [
        EmbeddedSub(index=28, lang='por', codec='subrip', title='Brazilian', external=False, sub_index=0)
    ]
    flow.extract_embedded_text = lambda path, stream: dialogue()
    job = run(flow, jobs, 'audit')
    assert job.state == 'done'
    installed = Path(media.path).with_suffix('.pt-BR.srt')
    assert installed.exists()
    assert flow.current(media, 'pt-BR')


def test_audit_prefers_brazilian_over_european_for_pt_br(setup):
    flow,jobs,provider,media=setup
    from subzero.worker.jellyfin import EmbeddedSub
    media.embedded = [
        EmbeddedSub(index=30, lang='por', codec='subrip', title='European', external=False, sub_index=1),
        EmbeddedSub(index=28, lang='por', codec='subrip', title='Brazilian', external=False, sub_index=0),
    ]
    candidates = flow.embedded_candidates(media, 'pt-BR')
    assert candidates[0].title == 'Brazilian'


def test_audit_cleans_sdh_from_embedded_subtitle(setup):
    flow,jobs,provider,media=setup
    from subzero.worker.jellyfin import EmbeddedSub
    media.embedded = [
        EmbeddedSub(index=29, lang='por', codec='subrip', title='Brazilian (SDH)', external=False, sub_index=0)
    ]
    sdh_text = dialogue().replace('Test dialogue', '[APLAUSOS] Test dialogue')
    flow.extract_embedded_text = lambda path, stream: sdh_text
    job = run(flow, jobs, 'audit')
    assert job.state == 'done'
    installed = Path(media.path).with_suffix('.pt-BR.srt')
    assert installed.exists()
    assert '[APLAUSOS]' not in installed.read_text()


def test_audit_falls_back_to_embedded_when_sidecar_rejected(setup):
    flow,jobs,provider,media=setup
    from subzero.worker.jellyfin import EmbeddedSub
    bad_sidecar = Path(media.path).with_suffix('.pt-BR.srt')
    bad_sidecar.write_text(dialogue(15))
    media.embedded = [
        EmbeddedSub(index=28, lang='por', codec='subrip', title='Brazilian', external=False, sub_index=0)
    ]
    flow.extract_embedded_text = lambda path, stream: dialogue()
    job = run(flow, jobs, 'audit')
    assert job.state == 'done'
    assert flow.current(media, 'pt-BR')




