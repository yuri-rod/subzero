import os
import platform as host
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping


def default_whisper_device(platform: str | None = None, machine: str | None = None) -> str:
    """mlx is the accelerated backend on Apple Silicon; cuda everywhere else."""
    platform = sys.platform if platform is None else platform
    machine = host.machine() if machine is None else machine
    return "mlx" if platform == "darwin" and machine == "arm64" else "cuda"


@dataclass
class Config:
    jellyfin_url: str
    jellyfin_key: str = field(repr=False)
    bearer_token: str = field(repr=False)
    opensubtitles_key: str = field(default="", repr=False)
    opensubtitles_user: str = ""
    opensubtitles_password: str = field(default="", repr=False)
    db_path: str = "jobs.db"
    state_path: str = "watch.json"
    log_dir: str = "logs"
    whisper_model: str = "large-v3"
    whisper_device: str = field(default_factory=default_whisper_device)
    whisper_compute_type: str = ""
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "subzero/hy-mt2:7b"
    ollama_keep_alive: str = "2m"
    ollama_num_ctx: int = 4096
    ollama_num_predict: int = 2048
    translation_provider: str = 'ollama'
    translation_fallback: str = ""
    applefm_url: str = 'http://127.0.0.1:1976'
    deepl_api_key: str = field(default='', repr=False)
    libretranslate_runtime: str = str(Path.home() / '.local/share/subzero/libretranslate')
    ocr_enabled: bool = field(default_factory=lambda: sys.platform == 'darwin')
    ocr_rescue_model: str = ""
    ocr_rescue_model_digest: str = ""
    daily_download_budget: int = 15
    auto_langs: list[str] = field(default_factory=lambda: ["pt-BR"])
    accepted_langs: list[str] = field(default_factory=list)
    auto_enabled: bool = True
    auto_window_start: int = 4
    auto_window_end: int = 7
    watch_interval: int = 600
    ntfy_url: str = "https://ntfy.sh"
    ntfy_topic: str = ""
    # idioma que sai sem sufixo: "Filme.srt" ao lado de "Filme.mkv"
    bare_lang: str = ""
    # idiomas tentados no OpenSubtitles quando o principal nao tem nada,
    # antes de gastar GPU com o whisper
    fallback_langs: list[str] = field(default_factory=list)
    # baixados para servir de origem da traducao
    translate_from: list[str] = field(default_factory=lambda: ["en"])
    # pastas que nunca entram na fila, em nenhuma via
    excluded_paths: list[str] = field(default_factory=list)
    asr_compat: bool = False
    asr_max_mb: int = 1024
    sync_cache: str = str(Path.home()/'.cache/subzero')
    sync_audit_only: bool = False
    idle_shutdown_minutes: int = 15
    qbt_url: str = 'http://127.0.0.1:8585'
    qbt_api_key: str = field(default='', repr=False)
    qbt_enabled: bool = False
    qbt_poll_interval: int = 30
    qbt_categories: list[str] = field(default_factory=lambda: ['movies', 'tv shows'])
    qbt_state_path: str = ''

    def __post_init__(self):
        if self.translation_provider not in ('ollama', 'libretranslate', 'deepl-free', 'applefm'):
            raise ValueError('TRANSLATION_PROVIDER must be ollama, libretranslate, deepl-free or applefm')
        if self.translation_fallback and self.translation_fallback not in ('libretranslate', 'applefm'):
            raise ValueError('TRANSLATION_FALLBACK must be libretranslate, applefm or empty')
        if bool(self.ocr_rescue_model) != bool(self.ocr_rescue_model_digest):
            raise ValueError('OCR rescue requires both a model and its digest')
        if self.qbt_enabled and not self.qbt_api_key:
            raise ValueError('QBT_ENABLED requires QBT_API_KEY')

    @classmethod
    def load(cls, env: Mapping[str, str] | None = None) -> "Config":
        env = env if env is not None else os.environ
        missing = [k for k in ("JELLYFIN_URL", "JELLYFIN_API_KEY", "BEARER_TOKEN") if not env.get(k)]
        if missing:
            raise ValueError(f"faltando no .env: {', '.join(missing)}")
        langs = [l.strip() for l in env.get("AUTO_LANGS", "pt-BR").split(",") if l.strip()]
        sync_cache = env.get('SYNC_CACHE', str(Path.home() / '.cache/subzero'))
        return cls(
            jellyfin_url=env["JELLYFIN_URL"].rstrip("/"),
            jellyfin_key=env["JELLYFIN_API_KEY"],
            bearer_token=env["BEARER_TOKEN"],
            opensubtitles_key=env.get("OPENSUBTITLES_API_KEY", ""),
            opensubtitles_user=env.get("OPENSUBTITLES_USERNAME", ""),
            opensubtitles_password=env.get("OPENSUBTITLES_PASSWORD", ""),
            db_path=env.get("DB_PATH", "jobs.db"),
            state_path=env.get("STATE_PATH", "watch.json"),
            log_dir=env.get("LOG_DIR", "logs"),
            whisper_model=env.get("WHISPER_MODEL", "large-v3"),
            whisper_device=env.get("WHISPER_DEVICE") or default_whisper_device(),
            whisper_compute_type=env.get("WHISPER_COMPUTE_TYPE", ""),
            ollama_url=env.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/"),
            ollama_model=env.get("OLLAMA_MODEL", "subzero/hy-mt2:7b"),
            ollama_keep_alive=env.get("OLLAMA_KEEP_ALIVE", "2m"),
            ollama_num_ctx=int(env.get("OLLAMA_NUM_CTX", "4096")),
            ollama_num_predict=int(env.get("OLLAMA_NUM_PREDICT", "2048")),
            translation_provider=env.get('TRANSLATION_PROVIDER', 'ollama').strip().lower(),
            translation_fallback=env.get('TRANSLATION_FALLBACK', env.get('TRANSLATION_FALLBACK_PROVIDER', '')).strip().lower(),
            applefm_url=env.get('APPLEFM_URL', 'http://127.0.0.1:1976').rstrip('/'),
            deepl_api_key=env.get('DEEPL_API_KEY', '').strip(),
            libretranslate_runtime=env.get('LIBRETRANSLATE_RUNTIME', str(Path.home() / '.local/share/subzero/libretranslate')),
            ocr_enabled=env.get('OCR_ENABLED', '1' if sys.platform == 'darwin' else '0').lower() not in ('0', 'false', 'no'),
            ocr_rescue_model=env.get('OCR_RESCUE_MODEL', '').strip(),
            ocr_rescue_model_digest=env.get('OCR_RESCUE_MODEL_DIGEST', '').strip(),
            daily_download_budget=int(env.get("DAILY_DOWNLOAD_BUDGET", "15")),
            auto_langs=langs,
            accepted_langs=[l.strip() for l in env.get("ACCEPTED_LANGS", "").split(",") if l.strip()],
            auto_enabled=env.get("AUTO_ENABLED", "1") not in ("0", "false", "no"),
            auto_window_start=int(env.get("AUTO_WINDOW_START", "4")),
            auto_window_end=int(env.get("AUTO_WINDOW_END", "7")),
            watch_interval=int(env.get("WATCH_INTERVAL", "600")),
            ntfy_url=env.get("NTFY_URL", "https://ntfy.sh").rstrip("/"),
            ntfy_topic=env.get("NTFY_TOPIC", ""),
            bare_lang=env.get("SIDECAR_BARE_LANG", ""),
            fallback_langs=[l.strip() for l in env.get("FALLBACK_LANGS", "").split(",") if l.strip()],
            translate_from=[l.strip() for l in env.get("TRANSLATE_FROM", "en").split(",") if l.strip()],
            excluded_paths=[p.strip() for p in env.get("EXCLUDE_PATHS", "").split(",") if p.strip()],
            asr_compat=env.get("ASR_COMPAT", "0") in ("1", "true", "yes"),
            asr_max_mb=int(env.get("ASR_MAX_MB", "1024")),
            sync_cache=sync_cache,
            sync_audit_only=env.get('SYNC_AUDIT_ONLY','0') in ('1','true','yes'),
            idle_shutdown_minutes=int(env.get("IDLE_SHUTDOWN_MINUTES", "15")),
            qbt_url=env.get('QBT_URL', 'http://127.0.0.1:8585').rstrip('/'),
            qbt_api_key=env.get('QBT_API_KEY', '').strip(),
            qbt_enabled=env.get('QBT_ENABLED', '0') not in ('0', 'false', 'no'),
            qbt_poll_interval=int(env.get('QBT_POLL_INTERVAL', '30')),
            qbt_categories=[c.strip() for c in env.get('QBT_CATEGORIES', 'movies,tv shows').split(',') if c.strip()],
            qbt_state_path=env.get('QBT_STATE_PATH', '').strip() or str(Path(sync_cache) / 'qbt-seen.json'),
        )


def resolve_deepl_key(cfg: Config) -> str:
    if cfg.deepl_api_key:
        return cfg.deepl_api_key
    if sys.platform != 'darwin':
        raise ValueError('DeepL Free requires DEEPL_API_KEY')
    try:
        keychain = subprocess.run(
            ['/usr/bin/security', 'find-generic-password', '-s', 'subzero.deepl.api-free',
             '-a', 'worker', '-w'],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError, UnicodeError):
        raise ValueError('DeepL Free key could not be read from macOS Keychain') from None
    if keychain.returncode != 0 or not keychain.stdout.strip():
        raise ValueError('DeepL Free requires DEEPL_API_KEY or the macOS Keychain entry '
                         'subzero.deepl.api-free for account worker')
    return keychain.stdout.strip()


def read_env_file(path: str | Path, *, required: bool = False) -> dict[str, str]:
    values: dict[str, str] = {}
    p = Path(path)
    if not required and not p.exists():
        return values
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"')
    return values
