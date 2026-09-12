import hashlib
import os
import re
import shutil
import stat
import tempfile
from pathlib import Path

from subzero.reference import build_reference, fingerprint, verify_text
from subzero.shift import shift_timestamps
from subzero.timing import correction

from .guards import check_excellence_guards, sanitize_to_excellence
from .moviehash import moviehash
from .service import same_language
from .srt import dump, parse, strip_hearing_impaired
from .syncstore import SyncStore, broken_file_ids
from .tracks import audio_start_offset, extract_audio, shift, sidecar_path, transcribe, translate
from .watch import EDITIONS, excluded, promoted, release_score, same_title, sync_compatible, title_query, tokens


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


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

    def install(self, media, job, key, text, report):
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
        for path in video.parent.glob(f"{video.stem}*.srt"):
            try:
                if path.resolve() == target.resolve():
                    continue
                if not bare and path.name == f"{video.stem}.srt":
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
        kind = job.kind if job.kind in ('audit','refetch','resync','embedded_translate','rebuild') else 'audit'
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
                cleaned = sanitize_to_excellence(text)
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
            text = sanitize_to_excellence(raw)
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
            text = sanitize_to_excellence(text)
            report = verify_text(text,reference)
            guard = check_excellence_guards(text,job.target_lang)
            if report.status == 'pass' and guard.ok:
                return self.install(media,job,key,text,report)
            change = correction(report)
            if change is None:
                continue
            scale,offset = change
            repaired,_ = shift_timestamps(text,offset,scale)
            repaired = sanitize_to_excellence(repaired)
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

    def translate_cues(self, cues, lang, progress, source_lang=None):
        self.translation_ready()
        try:
            return translate(cues,lang,self.service.ollama,progress,strict=True,source_lang=source_lang)
        finally:
            self.service.ollama.release()

    def source_sidecar(self, media, reference):
        video = Path(media.path)
        prefix = video.stem+'.'
        paths = sorted(video.parent.glob('*.srt'))
        for lang in getattr(self.cfg,'translate_from',['en']):
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
        cues = strip_hearing_impaired(parse(text))
        if not same_language(lang,job.target_lang):
            try:
                cues = self.translate_cues(cues,job.target_lang,progress,source_lang=lang)
            except RuntimeError as err:
                return self.review(media,job,key,f'Translation failed: {err}')
        raw = dump(cues)
        text = sanitize_to_excellence(raw)
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
        text = sanitize_to_excellence(raw)
        report = verify_text(text,reference,phase=45)
        self.stage(key,job.target_lang,text)
        guard = check_excellence_guards(text,job.target_lang)
        if report.status == 'pass' and guard.ok:
            return self.install(media,job,key,text,report)
        reason = report.reason if not guard.ok else guard.reason
        self.review(media,job,key,f'All stages exhausted: {reason}')
