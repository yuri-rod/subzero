import os
from typing import Callable

from .jobs import Job
from .srt import parse, strip_hearing_impaired
from .tracks import (audio_start_offset, deliver, extract_audio, extract_embedded,
                     shift, sidecar_path, transcribe, translate)

Progress = Callable[[str, int], None]

# a UI fala em pt-BR/en/es/ja, o Jellyfin em codigo de tres letras
THREE_LETTER = {"pt-BR": "por", "pt": "por", "en": "eng", "es": "spa", "ja": "jpn"}


def three_letter(lang: str) -> str:
    return THREE_LETTER.get(lang, lang.split("-")[0][:3].lower())


def same_language(a: str, b: str) -> bool:
    return three_letter(a) == three_letter(b)


class Service:
    def __init__(self, jellyfin, opensubs, holder, ollama, bare_lang: str = ""):
        self.jellyfin = jellyfin
        self.opensubs = opensubs
        self.holder = holder
        self.ollama = ollama
        # idioma entregue com o nome exato do video, sem tag
        self.bare_lang = bare_lang
        self.sync_flow = None

    def _bare(self, lang: str) -> bool:
        return bool(self.bare_lang) and same_language(lang, self.bare_lang)

    def run(self, job: Job, progress: Progress) -> str:
        if self.sync_flow is not None:
            return self.sync_flow.run(job,progress)
        media = self.jellyfin.media(job.item_id)
        handler = getattr(self, f"_{job.kind}")
        return handler(media, job, progress)

    def _embedded(self, media, job: Job, progress: Progress) -> str:
        if job.source_id is None or not str(job.source_id).isdigit():
            raise RuntimeError("faltou o indice da legenda embutida")
        path = extract_embedded(media, self.embedded_track(media, int(job.source_id)).sub_index,
                                job.target_lang, progress, bare=self._bare(job.target_lang))
        self.jellyfin.refresh(media.item_id)
        return path

    def _opensubtitles(self, media, job: Job, progress: Progress) -> str:
        if not job.source_id:
            raise RuntimeError("faltou o file_id do OpenSubtitles")
        progress("baixando do OpenSubtitles", 30)
        text = self.opensubs.download(int(job.source_id))
        cues = strip_hearing_impaired(parse(text))
        if not cues:
            raise RuntimeError("o arquivo baixado nao tem legenda valida")
        progress("gravando sidecar", 90)
        return deliver(media, cues, job.target_lang, self.jellyfin, bare=self._bare(job.target_lang))

    def _whisper(self, media, job: Job, progress: Progress) -> str:
        # a placa e uma so: o gemma sai antes de o whisper entrar
        if self.ollama is not None:
            self.ollama.release()
        # medido antes de extrair: depois do wav o desvio ja se perdeu
        offset = audio_start_offset(media.path)
        audio = extract_audio(media.path, media.duration, progress)
        try:
            cues, detected = transcribe(audio, self.holder, progress)
        finally:
            try:
                os.remove(audio)
            except OSError:
                pass
        cues = shift(cues, offset)

        if not same_language(detected, job.target_lang):
            progress("traduzindo", 0)
            cues = translate(cues, job.target_lang, self.ollama, progress)

        progress("gravando sidecar", 95)
        return deliver(media, cues, job.target_lang, self.jellyfin, bare=self._bare(job.target_lang))

    def _translate(self, media, job: Job, progress: Progress) -> str:
        source = self._source_cues(media, job)
        progress("traduzindo", 0)
        cues = translate(source, job.target_lang, self.ollama, progress)
        progress("gravando sidecar", 95)
        return deliver(media, cues, job.target_lang, self.jellyfin, bare=self._bare(job.target_lang))

    @staticmethod
    def embedded_track(media, index: int):
        stream = next((s for s in media.embedded if s.index == index and not s.external), None)
        if stream is None:
            raise RuntimeError(f"o item nao tem a trilha embutida {index}")
        return stream

    def _source_cues(self, media, job: Job):
        if not job.source_id:
            raise RuntimeError("faltou a legenda de origem")
        if str(job.source_id).startswith("embedded:"):
            stream = self.embedded_track(media, int(str(job.source_id).split(":", 1)[1]))
            path = extract_embedded(media, stream.sub_index, stream.lang)
        else:
            path = sidecar_path(media.path, str(job.source_id))
        try:
            text = open(path, encoding="utf-8", errors="replace").read()
        except OSError as err:
            raise RuntimeError(f"legenda de origem nao encontrada: {err}") from err
        cues = strip_hearing_impaired(parse(text))
        if not cues:
            raise RuntimeError("a legenda de origem esta vazia")
        return cues
