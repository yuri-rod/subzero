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
        assert kwargs['provider'] == 'ollama'
        assert kwargs['model'] == 'translategemma:4b'
        assert kwargs['url'] == 'http://127.0.0.1:11434'
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
        Path(kwargs['output']).write_text(dump(merged))
        return SimpleNamespace(cues_recovered=1)

    monkeypatch.setattr(syncflow, 'fill_subtitle_gaps', recover)
    job = run(flow, jobs, 'recover_gaps')
    assert job.state == 'done'
    recovered = parse(target.read_text())[10]
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
        kwargs['progress'](1, 1)
        merged = parse(reference['text'])
        merged.insert(10, cues[10])
        Path(kwargs['output']).write_text(dump(merged))
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
def test_repair_shares_source_context_across_block_boundaries(repair_flow, monkeypatch, series_name):
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
                       'passage': '\n'.join(c.text for c in cues[8:68])}


def test_repair_bounds_long_background_without_losing_current_block(repair_flow, monkeypatch):
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
    assert all(len(c['passage']) <= 6000 and len(c['title']) <= 256 for c in requested)
    assert all(c.text in requested[2]['passage'] for c in cues[40:60])
    assert cues[39].text in requested[2]['passage']
    assert cues[60].text in requested[2]['passage']


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
        cues.insert(1, Cue(0, cues[0].start + .2, cues[0].end, 'Overlapping text'))
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
