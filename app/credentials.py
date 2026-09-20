"""Gestión de credenciales de APIs externas con cascada de fuentes.

Prioridad para la clave de Gemini:
    1. st.secrets  → .streamlit/secrets.toml  (recomendado)
    2. env var     → GEMINI_API_KEY
    3. fichero     → data/credentials.json    (persistencia personal)
    4. manual      → lo que el usuario pega en la barra lateral

El orden garantiza que en un despliegue compartido (servidor) baste
con poner los secretos en .streamlit/secrets.toml, sin que cada
usuario tenga que introducir su clave. En local, si no hay secrets
ni env var, el usuario puede guardar su clave en un fichero para no
tener que pegarla en cada arranque.
"""
from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path

import streamlit as st

CREDENTIALS_PATH = Path("data/credentials.json")

_ENV_GEMINI = "GEMINI_API_KEY"
_SECRETS_GEMINI = "GEMINI_API_KEY"


def _read_secrets() -> str | None:
    try:
        if _SECRETS_GEMINI in st.secrets:
            return str(st.secrets[_SECRETS_GEMINI])
    except Exception:
        # st.secrets lanza excepción si no existe el fichero. Silenciar
        # aquí es lo correcto: solo significa "no hay secrets".
        pass
    return None


def _read_env() -> str | None:
    return os.getenv(_ENV_GEMINI) or None


def _read_file() -> dict:
    if not CREDENTIALS_PATH.exists():
        return {}
    try:
        data: dict = json.loads(CREDENTIALS_PATH.read_text(encoding="utf-8"))
        return data
    except (json.JSONDecodeError, OSError):
        return {}


def _write_file(data: dict) -> None:
    CREDENTIALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CREDENTIALS_PATH.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def get_gemini_key_with_source() -> tuple[str, str]:
    """Devuelve (clave, fuente).

    `fuente` es uno de: 'secrets', 'env', 'fichero', 'manual', 'ninguna'.
    Se usa en la UI para que el usuario sepa de dónde sale la clave.
    """
    key = _read_secrets()
    if key:
        return key, "secrets"

    key = _read_env()
    if key:
        return key, "env"

    file_data = _read_file()
    key = file_data.get("gemini_api_key")
    if key:
        return key, "fichero"

    key = st.session_state.get("gemini_api_key", "")
    if key:
        return key, "manual"

    return "", "ninguna"


def get_gemini_key() -> str:
    """Atajo: solo la clave, sin la fuente."""
    return get_gemini_key_with_source()[0]


def save_gemini_key_to_file(key: str) -> None:
    """Guarda la clave en data/credentials.json (texto plano).

    No se guarda en .gitignore automáticamente -- ver docs del
    proyecto; lo esperable es que el usuario añada esa ruta a su
    .gitignore para no subirla al repo.
    """
    data = _read_file()
    data["gemini_api_key"] = key
    _write_file(data)


def delete_gemini_key_from_file() -> None:
    """Borra la clave guardada en fichero (si existe)."""
    data = _read_file()
    data.pop("gemini_api_key", None)
    if data:
        _write_file(data)
    else:
        with contextlib.suppress(FileNotFoundError):
            CREDENTIALS_PATH.unlink()


def has_file_key() -> bool:
    """True si hay una clave guardada en el fichero local."""
    return bool(_read_file().get("gemini_api_key"))
