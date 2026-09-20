"""Sección compartida de "Interpretación" de una página de análisis.

Renderiza, en este orden:
    1. La interpretación determinista (reglas fijas, sin IA, siempre
       disponible e instantánea) de los resultados ya calculados.
    2. Un botón opcional para ampliarla con IA (Gemini o Groq, según
       lo configurado en la portada) por encima de esa base.
    3. Descarga en PDF de lo que haya en pantalla (solo la
       interpretación determinista, o con la ampliación de IA si se
       generó).

Centralizado aquí porque el patrón se repite en las 8 páginas del
núcleo cuantitativo (GARCH, Cointegración, Fama-French, Riesgo,
Backtest, Walk-Forward, Optimización, Robustez): sin esto, cada
página duplicaría casi el mismo bloque de botón + spinner + manejo de
error + descarga PDF.
"""
from __future__ import annotations

from typing import Any

import streamlit as st

from app.core.ai_report import AIReportError
from app.core.report import build_interpretation_pdf
from app.state import generate_ai_report, get_ai_api_key
from app.styles import callout, section


def render_interpretation_section(
    *,
    title: str,
    bullets: list[str],
    ai_prompt: str,
    state_key: str,
    pdf_title: str,
    pdf_filename: str,
    pdf_meta: dict[str, Any],
) -> None:
    """Renderiza la sección de interpretación descrita arriba.

    Args:
        title: título de la sección (p.ej. "📝 Interpretación").
        bullets: puntos de la interpretación determinista, ya
            generados por `app.core.interpretation`.
        ai_prompt: prompt a enviar al proveedor de IA si el usuario
            pulsa "Ampliar con IA".
        state_key: clave de `st.session_state` donde guardar el texto
            de la ampliación con IA -- debe ser única por página
            (p.ej. "garch_ai_report").
        pdf_title: título del PDF descargable.
        pdf_filename: nombre del archivo PDF descargable.
        pdf_meta: metadatos a mostrar en el PDF (ticker, parámetros...).
    """
    section(title)
    if bullets:
        st.markdown("\n".join(f"- {b}" for b in bullets))
    else:
        callout(
            "No hay suficientes datos para interpretar automáticamente este resultado.",
            variant="info",
        )

    ai_key = get_ai_api_key()
    if not ai_key:
        callout(
            "Configura tu clave gratuita de IA (Gemini o Groq) en la "
            "barra lateral de la portada ('🤖 Informes con IA') para "
            "ampliar esta interpretación con un análisis narrativo -- la "
            "interpretación de arriba ya funciona sin ella.",
            variant="info",
        )
    else:
        if st.button("🤖 Ampliar con IA", key=f"_ai_button_{state_key}"):
            with st.spinner("Generando ampliación con IA..."):
                try:
                    st.session_state[state_key] = generate_ai_report(ai_prompt)
                except AIReportError as e:
                    st.session_state[state_key] = None
                    st.error(f"No se pudo generar la ampliación: {e}")

        if st.session_state.get(state_key):
            st.markdown("**Ampliación con IA:**")
            st.markdown(st.session_state[state_key])

    st.download_button(
        "📄 Descargar interpretación en PDF",
        build_interpretation_pdf(
            title=pdf_title,
            meta=pdf_meta,
            bullets=bullets,
            ai_text=st.session_state.get(state_key),
        ),
        file_name=pdf_filename,
        mime="application/pdf",
        key=f"_pdf_button_{state_key}",
    )
