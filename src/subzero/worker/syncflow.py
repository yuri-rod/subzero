import hashlib
import json
import math
import os
import re
import shutil
import stat
import tempfile
from collections import Counter
from pathlib import Path

from subzero.caption_timeline import compose_caption_timeline, validate_caption_timeline
from subzero.convert import parse_srt, dump_srt
from subzero.ocr import fill_subtitle_gaps, uncovered_intervals
from subzero.reference import build_reference, fingerprint, verify_text
from subzero.shift import shift_timestamps
from subzero.timing import Report, correction
from subzero.translate import (TRANSLATION_PROMPT_VERSION, TRANSLATION_CONTEXT_CUES,
                               TRANSLATION_CONTEXT_CHARS, _is_native_translation,
                               _is_translategemma, _previous_context, translation_blocks)

from .guards import check_excellence_guards, check_language_completeness, sanitize_to_excellence
from .moviehash import moviehash
from .service import same_language
from .srt import Cue, dump, parse, strip_hearing_impaired
from .syncstore import SyncStore, broken_file_ids
from .tracks import audio_start_offset, extract_audio, shift, sidecar_path, transcribe, translate
from .watch import EDITIONS, excluded, promoted, release_score, same_title, sync_compatible, title_query, tokens


OCR_SOURCE_VERSION = 9


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def read_gap_source(path):
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > 2_000_000:
        raise RuntimeError('Gap recovery requires a regular subtitle file under 2 MB')
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0)
    with os.fdopen(os.open(path, flags), 'rb') as handle:
        info = os.fstat(handle.fileno())
        if not os.path.samestat(before, info) or info.st_size > 2_000_000:
            raise RuntimeError('Subtitle changed before gap recovery')
        raw = handle.read(2_000_001)
    if len(raw) > 2_000_000:
        raise RuntimeError('Subtitle grew beyond the gap recovery limit')
    return raw.decode('utf-8-sig')


def validate_ocr_source(source, complete, reference, *, all_captions=False):
    anchor_report = verify_text(source, reference)
    if anchor_report.status != 'pass':
        raise RuntimeError(f'Original subtitles failed audio timing validation: {anchor_report.reason}')
    duration = float(reference.get('duration') or 0)
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError('Video duration is missing from OCR validation')
    original = parse(source)
    anchors = Counter((c.start, c.end, c.text) for c in original)
    gaps = uncovered_intervals([(0, duration)], [(c.start, c.end) for c in original])
    previous_start = previous_ocr_end = 0.0
    additions = 0
    for cue in parse(complete):
        if (not math.isfinite(cue.start + cue.end) or cue.start < previous_start
                or cue.start < 0 or cue.end <= cue.start or cue.end > duration):
            raise RuntimeError('Recovered captions have invalid ordering or timestamps')
        previous_start = cue.start
        anchor = (cue.start, cue.end, cue.text)
        if anchors[anchor]:
            anchors[anchor] -= 1
            continue
        if cue.start < previous_ocr_end:
            raise RuntimeError('Recovered OCR captions overlap each other')
        if not all_captions and not any(start <= cue.start and cue.end <= end for start, end in gaps):
            raise RuntimeError('Recovered caption overlaps dialogue or leaves the scanned gaps')
        previous_ocr_end = cue.end
        additions += 1
    if any(anchors.values()):
        raise RuntimeError('Caption recovery changed or removed an original English cue')
    return Report('pass', f'Original audio-aligned cues preserved; {additions} frame-timed OCR captions added',
                  anchor_report.windows)


class SyncFlow:
    def __init__(self, jobs, service, cfg, reference_builder=build_reference):
        self.jobs, self.service, self.cfg = jobs, service, cfg
        self.state = SyncStore(jobs)
        self.reference_builder = reference_builder
        self.cache = Path(cfg.sync_cache).expanduser().resolve()
        self.cache.mkdir(parents=True,exist_ok=True)

    def installed(self, media, lang):
        video = Path(media.path)
        primary = Path(sidecar_path(media.path,lang,bare=self.service._bare(lang)))
        if primary.exists():
            return primary
        for path in sorted(video.parent.glob('*.srt')):
            if path.stem.startswith(video.stem+'.'):
                tag = path.stem[len(video.stem)+1:]
                if same_language(tag,lang):
                    return path
        return primary

    @property
    def accepted_langs(self) -> list[str]:
        return getattr(self.cfg, "accepted_langs", [])

    def current(self, media, lang):
        try:
            if self.accepted_langs and not any(same_language(lang, al) for al in self.accepted_langs):
                return False
            path = self.installed(media,lang)
            if not path.exists():
                return False
            text = path.read_text(encoding='utf-8-sig')
            if not check_excellence_guards(text, lang, self.accepted_langs).ok:
                return False
            return self.state.current(fingerprint(media.path),lang,digest(text))
        except (OSError,UnicodeError):
            return False

    def active(self, job):
        current = self.jobs.get(job.id)
        if current is None or current.state != 'running':
            raise RuntimeError('Subtitle job cancelled')

    def review(self, media, job, key, reason):
        path = self.installed(media,job.target_lang)
        text = path.read_text(encoding='utf-8-sig') if path.exists() else ''
        self.state.audit(key,job.target_lang,digest(text),'inconclusive',{'reason':reason})
        self.jobs.needs_review(job.id,reason)

    def install(self, media, job, key, text, report, expected_source=None):
        if report.status != 'pass':
            raise ValueError('Only validated subtitles can be installed')
        self.active(job)
        if fingerprint(media.path) != key:
            raise RuntimeError('Video changed during subtitle validation')
        if self.accepted_langs and not any(same_language(job.target_lang, al) for al in self.accepted_langs):
            raise ValueError(f'Worker configured to only accept {self.accepted_langs}, got {job.target_lang}')
        cleaned = sanitize_to_excellence(text, job.target_lang, self.accepted_langs)
        guard = check_excellence_guards(cleaned, job.target_lang, self.accepted_langs)
        if not guard.ok:
            raise ValueError(f'Subtitle fails excellence guards: {guard.reason}')
        text = cleaned
        if self.cfg.sync_audit_only:
            self.review(media,job,key,'Validated candidate retained; audit-only mode')
            return None
        target = self.installed(media,job.target_lang)
        if target.is_symlink():
            raise RuntimeError('Refusing to replace a subtitle symlink')
        backup = self.cache/'backups'/key
        backup.mkdir(parents=True,exist_ok=True)
        with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=target.parent,
                                         prefix='.subtitle-',suffix='.tmp',delete=False) as handle:
            tmp = Path(handle.name)
            try:
                handle.write(text);handle.flush();os.fsync(handle.fileno())
                with self.jobs._db() as db:
                    db.execute('BEGIN IMMEDIATE')
                    state = db.execute('SELECT state FROM jobs WHERE id=?',(job.id,)).fetchone()
                    if state is None or state['state'] != 'running':
                        raise RuntimeError('Subtitle job cancelled before installation')
                    if expected_source is not None:
                        source, source_digest = expected_source
                        changed = (os.path.lexists(target) if source_digest is None
                                   else digest(read_gap_source(target)) != source_digest)
                        if target != source or changed:
                            raise RuntimeError('Subtitle changed during gap recovery')
                    if expected_source is not None and expected_source[1] is None:
                        try:
                            os.link(tmp, target)
                        except FileExistsError as err:
                            raise RuntimeError('Subtitle changed during gap recovery') from err
                        except OSError as err:
                            raise RuntimeError(f'Cannot create subtitle atomically without overwriting the target: {err}') from err
                    else:
                        if target.exists():
                            old = target.read_text(encoding='utf-8-sig')
                            shutil.copy2(target,backup/f'{job.target_lang}-{digest(old)}.srt')
                        os.replace(tmp,target)
                    self.prune_sidecars(media, target, job.target_lang)
            finally:
                tmp.unlink(missing_ok=True)
        self.state.audit(key,job.target_lang,digest(text),'pass',report.json())
        self.service.jellyfin.refresh(media.item_id)
        return str(target)

    def prune_sidecars(self, media, target: Path, lang: str):
        video = Path(media.path)
        bare = self.service._bare(lang)
        embedded_langs = {getattr(s, "lang", "").lower() for s in getattr(media, "embedded", [])}
        for path in video.parent.glob(f"{video.stem}*.srt"):
            try:
                if path.resolve() == target.resolve():
                    continue
                if not bare and path.name == f"{video.stem}.srt":
                    path.unlink(missing_ok=True)
                tag = path.name[len(video.stem) + 1:-4].lower()
                if tag and any(same_language(tag, el) for el in embedded_langs if el):
                    path.unlink(missing_ok=True)
            except OSError:
                pass

    def stage(self, key, lang, text):
        folder = self.cache/'candidates'/key/lang
        folder.mkdir(parents=True,exist_ok=True)
        path = folder/f'{digest(text)}.srt'
        path.write_text(text,encoding='utf-8')
        return path

    def run(self, job, progress):
        self.active(job)
        if not re.fullmatch(r'[a-zA-Z]{2,3}(?:-[a-zA-Z]{2})?',job.target_lang):
            raise ValueError('Invalid subtitle language')
        if self.accepted_langs and not any(same_language(job.target_lang, al) for al in self.accepted_langs):
            raise ValueError(f'Worker configured to only accept {self.accepted_langs}')
        media = self.service.jellyfin.media(job.item_id)
        if excluded(media.path,self.cfg.excluded_paths):
            raise RuntimeError('Media library is excluded')
        key = fingerprint(media.path)
        kind = job.kind if job.kind in ('audit','refetch','resync','embedded_translate','rebuild','recover_gaps','repair') else 'audit'
        if self.cfg.sync_audit_only:
            kind = 'audit'
        audio_lang = (media.audio_lang or '').strip()
        if (kind == 'rebuild' and audio_lang.lower() not in ('','und','unknown','mul','zxx')
                and not same_language(audio_lang,job.target_lang)):
            try:
                self.translation_ready()
            except RuntimeError as err:
                return self.review(media,job,key,f'Translation unavailable: {err}')
        progress('verificando referencia de audio',0)
        try:
            reference = self.reference_builder(media.path,self.cache/'references')
        except (ImportError,ValueError) as err:
            return self.review(media,job,key,f'Cannot validate this video: {err}')
        self.active(job)
        if not reference.get('speech'):
            return self.review(media,job,key,'No usable audio evidence; cannot certify timing')
        def guarded_progress(phase,percent):
            self.active(job)
            progress(phase,percent)
        return getattr(self,kind)(media,job,key,reference,guarded_progress)

    def audit(self, media, job, key, reference, progress):
        path = self.installed(media,job.target_lang)
        if path.exists():
            text = path.read_text(encoding='utf-8-sig')
            report = verify_text(text,reference)
            guard = check_excellence_guards(text,job.target_lang)
            if report.status == 'pass' and guard.ok:
                self.state.audit(key,job.target_lang,digest(text),'pass',report.json())
                return str(path)
            if report.status == 'pass' and not guard.ok:
                cleaned = sanitize_to_excellence(text, job.target_lang, self.accepted_langs)
                clean_report = verify_text(cleaned,reference)
                clean_guard = check_excellence_guards(cleaned,job.target_lang)
                if clean_report.status == 'pass' and clean_guard.ok:
                    return self.install(media,job,key,cleaned,clean_report)
            self.state.audit(key,job.target_lang,digest(text),report.status if guard.ok else 'reject',report.json())
            self.stage(key,job.target_lang,text)
            if (report.status == 'reject' or not guard.ok) and not self.cfg.sync_audit_only:
                quarantine = self.cache/'quarantine'/key
                quarantine.mkdir(parents=True,exist_ok=True)
                if path.is_symlink():
                    return self.review(media,job,key,'Subtitle symlink requires manual review')
                with self.jobs._db() as db:
                    db.execute('BEGIN IMMEDIATE')
                    row = db.execute('SELECT state FROM jobs WHERE id=?',(job.id,)).fetchone()
                    if row['state'] != 'running':
                        raise RuntimeError('Subtitle job cancelled before quarantine')
                    if digest(path.read_text(encoding='utf-8-sig')) != digest(text):
                        raise RuntimeError('Subtitle changed during audit')
                    shutil.copy2(path,quarantine/f'{job.target_lang}-{digest(text)}.srt')
                    path.unlink()
                self.service.jellyfin.refresh(media.item_id)
        if self.cfg.sync_audit_only:
            return self.review(media,job,key,'Audit complete; automatic replacement is disabled')
        self.jobs.advance(job.id,'refetch','Procurando outra legenda da mesma edicao')

    def refetch(self, media, job, key, reference, progress):
        attempts = self.state.attempts(key,job.target_lang)
        seen = {a['file_id'] for a in attempts}
        content = {a['digest'] for a in attempts if a['digest']}
        tried = sum(a['job_id']==job.id for a in attempts)
        opensubs = self.service.opensubs
        if getattr(opensubs, "quota_exhausted", lambda: False)():
            # sem cota nao adianta baixar: cai no resync local em vez de girar em falso
            self.jobs.advance(job.id,'resync','Cota do OpenSubtitles esgotada; verificando correcao de tempo')
            return
        bad = broken_file_ids(self.jobs, job.target_lang)
        candidates = opensubs.search(langs=[job.target_lang],
                        moviehash=moviehash(media.path) if Path(media.path).stat().st_size >= 131072 else None,
                        filename=Path(media.path).name,**title_query(media))
        edition = set(tokens(Path(media.path).stem)) & EDITIONS
        stem = Path(media.path).stem
        candidates = [c for c in candidates if same_title(media,c) and c.human
                      and same_language(c.lang,job.target_lang)
                      and sync_compatible(media,c)
                      and c.file_id not in bad
                      and (c.hash_match or (set(tokens(c.release)) & EDITIONS)==edition)]
        candidates.sort(key=lambda c:((c.hash_match or promoted(stem,c.release)),not c.forced,
                                      release_score(stem,c.release),
                                      c.from_trusted,c.downloads),reverse=True)
        for candidate in candidates:
            self.active(job)
            if tried >= 3:
                break
            if candidate.file_id in seen:
                continue
            if not self.state.reserve(key,job.target_lang,candidate.file_id,job.id,
                                      self.cfg.daily_download_budget):
                break
            tried += 1
            seen.add(candidate.file_id)
            progress(f'baixando candidato {tried}/3',20)
            raw = opensubs.download(candidate.file_id)
            if not raw.strip():
                # corpo vazio: legenda quebrada no servidor, marca para nunca
                # gastar cota com ela de novo em nenhum video
                self.state.update(key,job.target_lang,candidate.file_id,status='broken')
                continue
            text = sanitize_to_excellence(raw, job.target_lang, self.accepted_langs)
            if len(text.encode()) > 2_000_000:
                self.state.update(key,job.target_lang,candidate.file_id,status='oversized')
                continue
            sha = digest(text)
            if sha in content:
                self.state.update(key,job.target_lang,candidate.file_id,status='duplicate',digest=sha)
                continue
            content.add(sha)
            path = self.stage(key,job.target_lang,text)
            report = verify_text(text,reference)
            guard = check_excellence_guards(text,job.target_lang)
            status = report.status if guard.ok else 'reject'
            self.state.update(key,job.target_lang,candidate.file_id,status=status,
                              digest=sha,path=str(path),report=report.json())
            if report.status == 'pass' and guard.ok:
                return self.install(media,job,key,text,report)
        self.jobs.advance(job.id,'resync','Downloads esgotados; verificando correcao de tempo')

    def resync(self, media, job, key, reference, progress):
        folder = self.cache/'candidates'/key/job.target_lang
        for path in sorted(folder.glob('*.srt')):
            self.active(job)
            text = path.read_text(encoding='utf-8')
            if path.stem != digest(text):
                continue
            text = sanitize_to_excellence(text, job.target_lang, self.accepted_langs)
            report = verify_text(text,reference)
            guard = check_excellence_guards(text,job.target_lang)
            if report.status == 'pass' and guard.ok:
                return self.install(media,job,key,text,report)
            change = correction(report)
            if change is None:
                continue
            scale,offset = change
            repaired,_ = shift_timestamps(text,offset,scale)
            repaired = sanitize_to_excellence(repaired, job.target_lang, self.accepted_langs)
            if verify_text(repaired,reference).status != 'pass':
                continue
            report = verify_text(repaired,reference,phase=45)
            guard = check_excellence_guards(repaired,job.target_lang)
            if report.status == 'pass' and guard.ok:
                return self.install(media,job,key,repaired,report)
        self.jobs.advance(job.id,'embedded_translate','Sem correcao confiavel; usando faixa embutida')

    def translation_ready(self):
        if self.service.ollama is None:
            raise RuntimeError('Ollama is not configured')
        self.service.ollama.ensure_available()

    def recover_gaps(self, media, job, key, reference, progress):
        target = self.installed(media, job.target_lang)
        original = read_gap_source(target)

        def ocr_progress(done, total):
            progress('recuperando legendas com Apple Vision', int(90 * done / max(1, total)))

        try:
            if not parse(original):
                raise RuntimeError('Gap recovery requires existing subtitle cues')
            if not same_language(job.target_lang, 'en'):
                self.translation_ready()
            with tempfile.TemporaryDirectory(prefix='ocr-', dir=self.cache) as folder:
                source = Path(folder) / 'source.srt'
                output = Path(folder) / 'recovered.srt'
                source.write_text(original, encoding='utf-8')
                progress('recuperando legendas com Apple Vision', 0)
                self.active(job)
                recovered = fill_subtitle_gaps(
                    media.path, source, output=output, target_lang=job.target_lang,
                    provider='ollama', model=self.cfg.ollama_model, url=self.cfg.ollama_url,
                    cache_dir=self.cache / 'references', backup=False, progress=ocr_progress,
                )
                self.active(job)
                if not recovered.cues_recovered:
                    raise RuntimeError('Apple Vision found no captions to add')
                text = output.read_text(encoding='utf-8-sig')
                text = sanitize_to_excellence(text, job.target_lang, self.accepted_langs)
                report = validate_ocr_source(original, text, reference)
                guard = check_excellence_guards(text, job.target_lang, self.accepted_langs)
                if report.status != 'pass' or not guard.ok:
                    reason = guard.reason if not guard.ok else report.reason
                    raise RuntimeError(f'Merged subtitle failed validation: {reason}')
                return self.install(media, job, key, text, report,
                                    expected_source=(target, digest(original)))
        except (ImportError, OSError, RuntimeError, UnicodeError, ValueError) as err:
            self.active(job)
            reason = f'Gap recovery failed: {err}'
            self.state.audit(key, job.target_lang, digest(original), 'inconclusive', {'reason': reason})
            self.jobs.needs_review(job.id, reason)
        finally:
            if self.service.ollama is not None:
                self.service.ollama.release()

    def translate_cues(self, cues, lang, progress, source_lang=None):
        self.translation_ready()
        try:
            return translate(cues,lang,self.service.ollama,progress,strict=True,source_lang=source_lang)
        finally:
            self.service.ollama.release()

    def ocr_source(self, media, job, key, reference, source, progress):
        baseline = verify_text(source, reference)
        if baseline.status != 'pass':
            raise RuntimeError(f'English source failed audio timing validation: {baseline.reason}')
        folder = self.cache / 'ocr-sources' / key
        folder.mkdir(parents=True, exist_ok=True)
        cache = folder / f'{digest(str(OCR_SOURCE_VERSION) + ":" + source)}.json'
        if cache.exists():
            try:
                cached = json.loads(read_gap_source(cache))
                text = cached.get('text') if isinstance(cached, dict) else None
                if isinstance(text, str) and cached.get('digest') == digest(text):
                    validate_ocr_source(source, text, reference, all_captions=True)
                    progress('reutilizando fonte em ingles com Apple Vision', 30)
                    return text
            except (ValueError, UnicodeError, RuntimeError):
                pass

        def ocr_progress(done, total):
            progress('recuperando fonte em ingles com Apple Vision', int(30 * done / max(1, total)))

        with tempfile.TemporaryDirectory(prefix='ocr-source-', dir=self.cache) as tmp_dir:
            input_path = Path(tmp_dir) / 'source.srt'
            output_path = Path(tmp_dir) / 'complete.srt'
            input_path.write_text(source, encoding='utf-8')
            progress('recuperando fonte em ingles com Apple Vision', 0)
            self.active(job)
            recovered = fill_subtitle_gaps(media.path, input_path, output=output_path,
                                          target_lang='en', backup=False,
                                          cache_dir=self.cache / 'references', progress=ocr_progress,
                                          caption_cache_dir=self.cache / 'caption-scans',
                                          all_captions=True)
            self.active(job)
            text = read_gap_source(output_path) if recovered.cues_recovered else source
            candidate = self.stage(key, 'en', text)
        if fingerprint(media.path) != key:
            raise RuntimeError('Video changed during English caption recovery')
        legacy_report = verify_text(text, reference)
        failure = None
        try:
            source_report = validate_ocr_source(source, text, reference, all_captions=True)
        except RuntimeError as err:
            failure = str(err)
            source_report = Report('reject', failure)
        diagnostics = self.cache / 'ocr-validation' / key
        diagnostics.mkdir(parents=True, exist_ok=True)
        (diagnostics / cache.name).write_text(json.dumps({
            'source_digest': digest(source), 'candidate_digest': digest(text),
            'candidate': str(candidate), 'source': baseline.json(),
            'merged_audio_metric': legacy_report.json(), 'validation': source_report.json(),
        }, ensure_ascii=False), encoding='utf-8')
        if failure:
            raise RuntimeError(f'{failure}; OCR candidate retained at {candidate}')
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=folder,
                                         suffix='.tmp', delete=False) as handle:
            tmp = Path(handle.name)
            try:
                json.dump({'text': text, 'digest': digest(text)}, handle, ensure_ascii=False)
                handle.flush()
                os.fsync(handle.fileno())
                self.active(job)
                os.replace(tmp, cache)
            finally:
                tmp.unlink(missing_ok=True)
        return text

    def repair_translation(self, job, key, cues, progress, *, title=''):
        ollama = self.service.ollama
        title = ' '.join(title.split())[:256]
        settings = {
            'source': digest(dump(cues)), 'target_lang': job.target_lang, 'source_lang': 'en',
            'model': getattr(ollama, 'model', getattr(self.cfg, 'ollama_model', '')),
            'url': getattr(ollama, 'url', getattr(self.cfg, 'ollama_url', '')),
            'num_ctx': getattr(ollama, 'num_ctx', getattr(self.cfg, 'ollama_num_ctx', 4096)),
            'num_predict': getattr(ollama, 'num_predict', getattr(self.cfg, 'ollama_num_predict', 2048)),
            'prompt_version': TRANSLATION_PROMPT_VERSION, 'block_size': 20,
            'title': title, 'previous_cues': TRANSLATION_CONTEXT_CUES,
            'following_cues': 0, 'passage_chars': TRANSLATION_CONTEXT_CHARS,
        }
        use_context = _is_native_translation(settings['model']) and not _is_translategemma(settings['model'])
        folder = self.cache / 'translations' / key / digest(json.dumps(settings, sort_keys=True))
        folder.mkdir(parents=True, exist_ok=True)
        completed = []

        def valid_block(source, translated):
            if len(translated) != len(source):
                return False
            if any((a.start, a.end) != (b.start, b.end) or not b.text.strip()
                   for a, b in zip(source, translated)):
                return False
            text = sanitize_to_excellence(dump(strip_hearing_impaired(translated)),
                                          job.target_lang, self.accepted_langs)
            if [(c.start, c.end) for c in parse(text)] != [(c.start, c.end) for c in source]:
                return False
            return check_language_completeness(text, job.target_lang)[0]

        try:
            for block in translation_blocks(cues, settings['block_size'], settings['model']):
                self.active(job)
                start = len(completed)
                source = dump(block)
                cache = folder / f'{start:06d}.json'
                translated = None
                if cache.exists():
                    try:
                        cached = json.loads(read_gap_source(cache))
                        text = cached.get('text') if isinstance(cached, dict) else None
                        if (isinstance(text, str) and cached.get('source') == source
                                and cached.get('digest') == digest(text)):
                            candidate = parse(text)
                            if valid_block(block, candidate):
                                translated = [Cue(a.index, b.start, b.end, b.text)
                                              for a, b in zip(block, candidate)]
                    except (ValueError, UnicodeError):
                        pass
                if translated is None:
                    options = {}
                    if use_context:
                        options['context'] = _previous_context(cues[:start], {'title': title})
                    translated = translate(block, job.target_lang, ollama,
                                           lambda *_: self.active(job), strict=True, source_lang='en', **options)
                    self.active(job)
                    if not valid_block(block, translated):
                        raise RuntimeError(f'Translation block at cue {start + 1} is incomplete or untranslated')
                    text = dump(translated)
                    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=folder,
                                                     suffix='.tmp', delete=False) as handle:
                        tmp = Path(handle.name)
                        try:
                            json.dump({'source': source, 'text': text, 'digest': digest(text)},
                                      handle, ensure_ascii=False)
                            handle.flush()
                            os.fsync(handle.fileno())
                            self.active(job)
                            os.replace(tmp, cache)
                        finally:
                            tmp.unlink(missing_ok=True)
                completed.extend(translated)
                progress(f'traduzindo legendas {len(completed)}/{len(cues)}',
                         30 + int(65 * len(completed) / len(cues)))
            return completed
        finally:
            ollama.release()

    def generate_from_english(self, media, job, key, reference, source, progress):
        source = dump([cue for cue in parse(source) if strip_hearing_impaired([cue])])
        if not same_language(job.target_lang, 'en'):
            self.translation_ready()
        english = self.ocr_source(media, job, key, reference, source, progress)
        self.active(job)
        self.stage(key, 'en', english)
        cues = [cue for cue in parse(english) if strip_hearing_impaired([cue])]
        translated = cues
        if not same_language(job.target_lang, 'en'):
            translated = self.repair_translation(job, key, cues, progress,
                                                 title=media.series_name or media.name)
        self.active(job)
        if [(c.start, c.end) for c in translated] != [(c.start, c.end) for c in cues]:
            raise RuntimeError('Translation changed source cue timing or count')
        cleaned = sanitize_to_excellence(dump(strip_hearing_impaired(translated)),
                                         job.target_lang, self.accepted_langs)
        if [(c.start, c.end) for c in parse(cleaned)] != [(c.start, c.end) for c in cues]:
            raise RuntimeError('Subtitle cleanup changed dialogue timing or count')
        atoms = parse_srt(cleaned)
        timeline = compose_caption_timeline(atoms)
        validate_caption_timeline(atoms, timeline)
        text = sanitize_to_excellence(dump_srt(timeline), job.target_lang, self.accepted_langs)
        validate_caption_timeline(atoms, parse_srt(text))
        self.stage(key, job.target_lang, text)
        report = validate_ocr_source(source, english, reference, all_captions=True)
        guard = check_excellence_guards(text, job.target_lang, self.accepted_langs)
        if report.status != 'pass' or not guard.ok:
            reason = guard.reason if not guard.ok else report.reason
            raise RuntimeError(f'Regenerated subtitle failed validation: {reason}')
        return text, report

    def repair(self, media, job, key, reference, progress):
        target = self.installed(media, job.target_lang)
        original = read_gap_source(target)
        try:
            source = reference.get('text', '')
            if (not same_language(reference.get('language', 'und'), 'en')
                    or not source or verify_text(source, reference).status != 'pass'):
                sidecar = self.source_sidecar(media, reference, languages=('en',))
                if sidecar is None:
                    raise RuntimeError('Full repair requires a verified English subtitle source')
                source = sidecar[0]
            text, report = self.generate_from_english(media, job, key, reference, source, progress)
            return self.install(media, job, key, text, report,
                                expected_source=(target, digest(original)))
        except (ImportError, OSError, RuntimeError, UnicodeError, ValueError) as err:
            self.active(job)
            reason = f'Full repair failed: {err}'
            self.state.audit(key, job.target_lang, digest(original), 'inconclusive', {'reason': reason})
            self.jobs.needs_review(job.id, reason)

    def source_sidecar(self, media, reference, languages=None):
        video = Path(media.path)
        prefix = video.stem+'.'
        paths = sorted(video.parent.glob('*.srt'))
        for lang in languages if languages is not None else getattr(self.cfg,'translate_from',['en']):
            for path in paths:
                if not path.stem.startswith(prefix):
                    continue
                tag = path.stem[len(prefix):]
                if not re.fullmatch(r'[a-zA-Z]{2,3}(?:-[a-zA-Z]{2})?',tag):
                    continue
                if not same_language(tag.split('-')[0].lower(),lang.split('-')[0].lower()):
                    continue
                try:
                    before = path.lstat()
                    if not stat.S_ISREG(before.st_mode) or before.st_size > 2_000_000:
                        continue
                    flags = os.O_RDONLY | getattr(os,'O_NOFOLLOW',0) | getattr(os,'O_NONBLOCK',0)
                    fd = os.open(path,flags)
                    with os.fdopen(fd,'rb') as handle:
                        info = os.fstat(handle.fileno())
                        if not os.path.samestat(before,info) or info.st_size > 2_000_000:
                            continue
                        raw = handle.read(2_000_001)
                    if len(raw) > 2_000_000:
                        continue
                    text = raw.decode('utf-8-sig')
                except (OSError,UnicodeError):
                    continue
                if verify_text(text,reference).status == 'pass':
                    return text,tag
        return None

    def translate_source(self, media, job, key, reference, progress, text, lang):
        if getattr(self.cfg, 'ocr_enabled', False) and same_language(lang, 'en'):
            target = self.installed(media, job.target_lang)
            original = ''
            try:
                existed = target.exists() or target.is_symlink()
                original = read_gap_source(target) if existed else ''
                text, report = self.generate_from_english(media, job, key, reference, text, progress)
                return self.install(media, job, key, text, report,
                                    expected_source=(target, digest(original) if existed else None))
            except (ImportError, OSError, RuntimeError, UnicodeError, ValueError) as err:
                self.active(job)
                reason = f'English subtitle generation failed: {err}'
                self.state.audit(key, job.target_lang, digest(original), 'inconclusive', {'reason': reason})
                self.jobs.needs_review(job.id, reason)
                return None
        cues = strip_hearing_impaired(parse(text))
        if not same_language(lang,job.target_lang):
            try:
                cues = self.translate_cues(cues,job.target_lang,progress,source_lang=lang)
            except RuntimeError as err:
                return self.review(media,job,key,f'Translation failed: {err}')
        raw = dump(cues)
        text = sanitize_to_excellence(raw, job.target_lang, self.accepted_langs)
        report = verify_text(text,reference)
        guard = check_excellence_guards(text,job.target_lang)
        if report.status == 'pass' and guard.ok:
            return self.install(media,job,key,text,report)
        reason = guard.reason if not guard.ok else report.reason
        self.review(media,job,key,f'Translation failed validation: {reason}')

    def embedded_translate(self, media, job, key, reference, progress):
        text = reference.get('text','')
        if text and verify_text(text,reference).status == 'pass':
            source = text,reference.get('language','und')
        else:
            source = self.source_sidecar(media,reference)
        if source is None:
            self.jobs.advance(job.id,'rebuild','Sem legenda de origem verificada; refazendo pelo audio')
            return None
        return self.translate_source(media,job,key,reference,progress,*source)

    def rebuild(self, media, job, key, reference, progress):
        try:
            return self._rebuild(media,job,key,reference,progress)
        except RuntimeError as err:
            self.review(media,job,key,f'Audio rebuild failed: {err}')

    def _rebuild(self, media, job, key, reference, progress):
        source = self.source_sidecar(media,reference)
        if source is not None and not same_language(source[1],job.target_lang):
            try:
                self.translation_ready()
            except RuntimeError:
                audio_lang = (media.audio_lang or '').strip()
                if (audio_lang.lower() not in ('','und','unknown','mul','zxx')
                        and not same_language(audio_lang,job.target_lang)):
                    raise
                source = None
        if source is not None:
            return self.translate_source(media,job,key,reference,progress,*source)
        if self.service.ollama is not None:
            self.service.ollama.release()
        offset = audio_start_offset(media.path)
        audio = extract_audio(media.path,media.duration,progress)
        try:
            cues,detected = transcribe(audio,self.service.holder,progress)
        finally:
            Path(audio).unlink(missing_ok=True)
        cues = shift(cues,offset)
        if not same_language(detected,job.target_lang):
            cues = self.translate_cues(cues,job.target_lang,progress,source_lang=detected)
        raw = dump(cues)
        text = sanitize_to_excellence(raw, job.target_lang, self.accepted_langs)
        report = verify_text(text,reference,phase=45)
        self.stage(key,job.target_lang,text)
        guard = check_excellence_guards(text,job.target_lang)
        if report.status == 'pass' and guard.ok:
            return self.install(media,job,key,text,report)
        reason = guard.reason if not guard.ok else report.reason
        self.review(media,job,key,f'All stages exhausted: {reason}')
