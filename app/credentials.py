"""Gestión de credenciales de APIs externas con cascada de fuentes.

Prioridad para la clave de cada proveedor de IA (Gemini, Groq):
    1. st.secrets  → .streamlit/secrets.toml  (recomendado)
    2. env var     → GEMINI_API_KEY / GROQ_API_KEY
    3. fichero     → data/credentials.json    (persistencia personal)
    4. manual      → lo que el usuario pega en la barra lateral

El orden garantiza que en un despliegue compartido (servidor) baste
con poner los secretos en .streamlit/secrets.toml, sin que cada
usuario tenga que introducir su clave. En local, si no hay secrets
ni env var, el usuario puede guardar su clave en un fichero para no
tener que pegarla en cada arranque.

Las funciones públicas son deliberadamente por-proveedor
(`get_gemini_key_with_source`, `get_groq_key_with_source`...) en vez
de una única función con un parámetro `provider` -- así el resto del
código no tiene que pasar strings mágicos, y mypy puede comprobar
cada uso. Por dentro comparten toda la lógica (`_get_key_with_source`,
`_save_key_to_file`...), parametrizada por proveedor.
"""
from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path

import streamlit as st

CREDENTIALS_PATH = Path("data/credentials.json")

_ENV_VARS = {"gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY"}
_FILE_KEYS = {"gemini": "gemini_api_key", "groq": "groq_api_key"}
# "..._manual" -- no el key= del propio widget de texto (ver
# _on_gemini_key_change / _on_groq_key_change en app/main.py):
# Streamlit borra el session_state de un widget en cualquier página
# donde ese widget no se vuelva a crear, así que leer directamente el
# key= del campo hacía "desaparecer" la clave en cuanto se navegaba
# fuera de la portada. Estas son claves "normales" de session_state,
# sincronizadas a mano desde el widget, que sí sobreviven.
_SESSION_MANUAL_KEYS = {"gemini": "gemini_api_key_manual", "groq": "groq_api_key_manual"}


def _read_secrets(provider: str) -> str | None:
    env_var = _ENV_VARS[provider]
    try:
        if env_var in st.secrets:
            return str(st.secrets[env_var])
    except Exception:
        # st.secrets lanza excepción si no existe el fichero. Silenciar
        # aquí es lo correcto: solo significa "no hay secrets".
        pass
    return None


def _read_env(provider: str) -> str | None:
    return os.getenv(_ENV_VARS[provider]) or None


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


def _get_key_with_source(provider: str) -> tuple[str, str]:
    """Devuelve (clave, fuente) para `provider` ('gemini' o 'groq').

    `fuente` es uno de: 'secrets', 'env', 'fichero', 'manual', 'ninguna'.
    Se usa en la UI para que el usuario sepa de dónde sale la clave.
    """
    key = _read_secrets(provider)
    if key:
        return key, "secrets"

    key = _read_env(provider)
    if key:
        return key, "env"

    file_data = _read_file()
    key = file_data.get(_FILE_KEYS[provider])
    if key:
        return key, "fichero"

    key = st.session_state.get(_SESSION_MANUAL_KEYS[provider], "")
    if key:
        return key, "manual"

    return "", "ninguna"


def _save_key_to_file(provider: str, key: str) -> None:
    """Guarda la clave en data/credentials.json (texto plano).

    No se guarda en .gitignore automáticamente -- ver docs del
    proyecto; lo esperable es que el usuario añada esa ruta a su
    .gitignore para no subirla al repo.
    """
    data = _read_file()
    data[_FILE_KEYS[provider]] = key
    _write_file(data)


def _delete_key_from_file(provider: str) -> None:
    """Borra la clave de `provider` guardada en fichero (si existe)."""
    data = _read_file()
    data.pop(_FILE_KEYS[provider], None)
    if data:
        _write_file(data)
    else:
        with contextlib.suppress(FileNotFoundError):
            CREDENTIALS_PATH.unlink()


def _has_file_key(provider: str) -> bool:
    return bool(_read_file().get(_FILE_KEYS[provider]))


# ============================================================
#  Gemini
# ============================================================
def get_gemini_key_with_source() -> tuple[str, str]:
    return _get_key_with_source("gemini")


def get_gemini_key() -> str:
    """Atajo: solo la clave, sin la fuente."""
    return get_gemini_key_with_source()[0]


def save_gemini_key_to_file(key: str) -> None:
    _save_key_to_file("gemini", key)


def delete_gemini_key_from_file() -> None:
    _delete_key_from_file("gemini")


def has_file_key() -> bool:
    """True si hay una clave de Gemini guardada en el fichero local."""
    return _has_file_key("gemini")


# ============================================================
#  Groq
# ============================================================
def get_groq_key_with_source() -> tuple[str, str]:
    return _get_key_with_source("groq")


def get_groq_key() -> str:
    """Atajo: solo la clave, sin la fuente."""
    return get_groq_key_with_source()[0]


def save_groq_key_to_file(key: str) -> None:
    _save_key_to_file("groq", key)


def delete_groq_key_from_file() -> None:
    _delete_key_from_file("groq")


def has_groq_file_key() -> bool:
    """True si hay una clave de Groq guardada en el fichero local."""
    return _has_file_key("groq")
