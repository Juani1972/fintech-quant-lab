"""Tests para app/state.py."""
from unittest.mock import patch

import streamlit as st

from app.state import (
    generate_ai_report,
    get_ai_api_key,
    get_ai_model,
    get_ai_provider,
    get_global_provider,
    get_global_provider_kwargs,
)


def test_get_global_provider_defaults_to_yahoo():
    st.session_state.clear()
    assert get_global_provider() == "yahoo"


def test_get_global_provider_reads_session_state():
    st.session_state.clear()
    st.session_state["global_provider"] = "stooq"
    assert get_global_provider() == "stooq"


def test_get_global_provider_kwargs_empty_for_yahoo_and_stooq():
    st.session_state.clear()
    st.session_state["global_provider"] = "yahoo"
    assert get_global_provider_kwargs() == {}
    st.session_state["global_provider"] = "stooq"
    assert get_global_provider_kwargs() == {}


def test_get_global_provider_kwargs_includes_api_key_for_alphavantage():
    st.session_state.clear()
    st.session_state["global_provider"] = "alphavantage"
    st.session_state["global_provider_api_key"] = "clave-123"
    assert get_global_provider_kwargs() == {"api_key": "clave-123"}


def test_get_global_provider_kwargs_empty_for_alphavantage_without_key():
    st.session_state.clear()
    st.session_state["global_provider"] = "alphavantage"
    assert get_global_provider_kwargs() == {}


# ============================================================
#  Proveedor de IA (Gemini / Groq)
# ============================================================
def test_get_ai_provider_defaults_to_gemini():
    st.session_state.clear()
    assert get_ai_provider() == "gemini"


def test_get_ai_provider_reads_session_state():
    st.session_state.clear()
    st.session_state["ai_provider"] = "groq"
    assert get_ai_provider() == "groq"


def test_get_ai_api_key_dispatches_to_gemini_by_default():
    st.session_state.clear()
    with (
        patch("app.credentials.get_gemini_key", return_value="clave-gemini") as mock_gemini,
        patch("app.credentials.get_groq_key", return_value="clave-groq") as mock_groq,
    ):
        assert get_ai_api_key() == "clave-gemini"
    mock_gemini.assert_called_once()
    mock_groq.assert_not_called()


def test_get_ai_api_key_dispatches_to_groq_when_selected():
    st.session_state.clear()
    st.session_state["ai_provider"] = "groq"
    with (
        patch("app.credentials.get_gemini_key", return_value="clave-gemini") as mock_gemini,
        patch("app.credentials.get_groq_key", return_value="clave-groq") as mock_groq,
    ):
        assert get_ai_api_key() == "clave-groq"
    mock_groq.assert_called_once()
    mock_gemini.assert_not_called()


def test_get_ai_model_defaults_to_gemini_model():
    st.session_state.clear()
    from app.core.ai_report import DEFAULT_MODEL

    assert get_ai_model() == DEFAULT_MODEL


def test_get_ai_model_reads_groq_model_when_selected():
    st.session_state.clear()
    st.session_state["ai_provider"] = "groq"
    st.session_state["groq_model"] = "llama-guard"
    assert get_ai_model() == "llama-guard"


def test_generate_ai_report_dispatches_to_gemini_backend_by_default():
    st.session_state.clear()
    st.session_state["gemini_api_key_manual"] = "clave-gemini"
    with patch(
        "app.core.ai_report.generate_report", return_value="informe gemini",
    ) as mock_gen:
        result = generate_ai_report("prompt")
    assert result == "informe gemini"
    assert mock_gen.call_args.kwargs["api_key"] == "clave-gemini"


def test_generate_ai_report_dispatches_to_groq_backend_when_selected():
    st.session_state.clear()
    st.session_state["ai_provider"] = "groq"
    st.session_state["groq_api_key_manual"] = "clave-groq"
    with patch(
        "app.core.groq_report.generate_report", return_value="informe groq",
    ) as mock_gen:
        result = generate_ai_report("prompt")
    assert result == "informe groq"
    assert mock_gen.call_args.kwargs["api_key"] == "clave-groq"
