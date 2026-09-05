import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping


@dataclass
class Config:
    jellyfin_url: str
    jellyfin_key: str
    bearer_token: str
    opensubtitles_key: str = ""
    opensubtitles_user: str = ""
    opensubtitles_password: str = ""
    db_path: str = "jobs.db"
    state_path: str = "watch.json"
    log_dir: str = "logs"
    whisper_model: str = "large-v3"
    whisper_device: str = "cuda"
    whisper_compute_type: str = ""
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "gemma3:12b"
    daily_download_budget: int = 15
    auto_langs: list[str] = field(default_factory=lambda: ["pt-BR"])
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

    @classmethod
    def load(cls, env: Mapping[str, str] | None = None) -> "Config":
        env = env if env is not None else os.environ
        missing = [k for k in ("JELLYFIN_URL", "JELLYFIN_API_KEY", "BEARER_TOKEN") if not env.get(k)]
        if missing:
            raise ValueError(f"faltando no .env: {', '.join(missing)}")
        langs = [l.strip() for l in env.get("AUTO_LANGS", "pt-BR").split(",") if l.strip()]
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
            whisper_device=env.get("WHISPER_DEVICE", "cuda"),
            whisper_compute_type=env.get("WHISPER_COMPUTE_TYPE", ""),
            ollama_url=env.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/"),
            ollama_model=env.get("OLLAMA_MODEL", "gemma3:12b"),
            daily_download_budget=int(env.get("DAILY_DOWNLOAD_BUDGET", "15")),
            auto_langs=langs,
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
            sync_cache=env.get('SYNC_CACHE',str(Path.home()/'.cache/subzero')),
            sync_audit_only=env.get('SYNC_AUDIT_ONLY','0') in ('1','true','yes'),
            idle_shutdown_minutes=int(env.get("IDLE_SHUTDOWN_MINUTES", "15")),
        )


def read_env_file(path: str | Path) -> dict[str, str]:
    values: dict[str, str] = {}
    p = Path(path)
    if not p.exists():
        return values
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"')
    return values
