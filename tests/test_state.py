"""Tests para app/state.py."""
import streamlit as st

from app.state import get_global_provider, get_global_provider_kwargs


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
