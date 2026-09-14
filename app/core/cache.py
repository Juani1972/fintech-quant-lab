"""Caché desacoplada de Streamlit.

`app/core` es la capa de negocio y no debería depender obligatoriamente
de que Streamlit esté en ejecución — por ejemplo al usarse desde los
notebooks de `notebooks/`, que importan estas funciones directamente
sin pasar por `streamlit run`.

Este módulo centraliza ese acoplamiento en un único punto: expone un
decorador `cached(ttl)` que usa `st.cache_data` cuando Streamlit está
disponible (que es el caso normal, ya que es una dependencia obligatoria
del proyecto) y cae a un no-op si por lo que sea no lo está. Así,
ningún otro módulo de `app/core` necesita `import streamlit` — solo
este.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def cached(ttl: int, show_spinner: bool = False) -> Callable[[F], Any]:
    """Decorador de caché de resultados con TTL.

    Usa `st.cache_data` si Streamlit está instalado (caso normal);
    si no, devuelve la función sin modificar (sin caché, pero
    funcional) en vez de fallar.
    """
    try:
        import streamlit as st
    except ImportError:
        def _no_cache(fn: F) -> F:
            return fn
        return _no_cache

    return st.cache_data(ttl=ttl, show_spinner=show_spinner)
