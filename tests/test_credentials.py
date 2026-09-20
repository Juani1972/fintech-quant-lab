"""Tests para app/credentials.py: cascada de credenciales
(secrets -> env var -> fichero -> manual) para cada proveedor de IA
(Gemini, Groq)."""
import streamlit as st

import app.credentials as creds


def _isolate(monkeypatch, tmp_path):
    """Aísla cada test: fichero de credenciales en un tmp_path propio,
    sin variables de entorno ni secrets.toml reales, sesión limpia."""
    monkeypatch.setattr(creds, "CREDENTIALS_PATH", tmp_path / "credentials.json")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    st.session_state.clear()


def test_no_source_configured_returns_none(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    assert creds.get_gemini_key_with_source() == ("", "ninguna")
    assert creds.get_groq_key_with_source() == ("", "ninguna")
    assert creds.get_gemini_key() == ""
    assert creds.get_groq_key() == ""


def test_manual_session_state_key(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    st.session_state["gemini_api_key_manual"] = "clave-gemini"
    assert creds.get_gemini_key_with_source() == ("clave-gemini", "manual")
    # Groq no se ve afectado por la clave manual de Gemini.
    assert creds.get_groq_key_with_source() == ("", "ninguna")


def test_env_var_takes_priority_over_manual(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    st.session_state["gemini_api_key_manual"] = "clave-manual"
    monkeypatch.setenv("GEMINI_API_KEY", "clave-env")
    assert creds.get_gemini_key_with_source() == ("clave-env", "env")


def test_secrets_take_priority_over_env(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "clave-env")
    monkeypatch.setattr(creds.st, "secrets", {"GEMINI_API_KEY": "clave-secrets"})
    assert creds.get_gemini_key_with_source() == ("clave-secrets", "secrets")


def test_save_and_read_file_key(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    assert not creds.has_file_key()

    creds.save_gemini_key_to_file("clave-fichero")

    assert creds.has_file_key()
    assert creds.get_gemini_key_with_source() == ("clave-fichero", "fichero")


def test_file_key_takes_priority_over_manual(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    st.session_state["gemini_api_key_manual"] = "clave-manual"
    creds.save_gemini_key_to_file("clave-fichero")
    assert creds.get_gemini_key_with_source() == ("clave-fichero", "fichero")


def test_gemini_and_groq_file_keys_are_independent(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    creds.save_gemini_key_to_file("clave-gemini")
    creds.save_groq_key_to_file("clave-groq")

    assert creds.get_gemini_key() == "clave-gemini"
    assert creds.get_groq_key() == "clave-groq"
    assert creds.has_file_key()
    assert creds.has_groq_file_key()


def test_delete_one_provider_file_key_keeps_the_other(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    creds.save_gemini_key_to_file("clave-gemini")
    creds.save_groq_key_to_file("clave-groq")

    creds.delete_gemini_key_from_file()

    assert not creds.has_file_key()
    assert creds.has_groq_file_key()
    assert creds.get_groq_key() == "clave-groq"
    # El fichero en sí sigue existiendo -- todavía guarda la de Groq.
    assert creds.CREDENTIALS_PATH.exists()


def test_delete_last_file_key_removes_the_file(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    creds.save_gemini_key_to_file("clave-gemini")

    creds.delete_gemini_key_from_file()

    assert not creds.has_file_key()
    assert not creds.CREDENTIALS_PATH.exists()


def test_delete_file_key_when_no_file_exists_does_not_raise(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    creds.delete_gemini_key_from_file()  # no debe lanzar excepción
    creds.delete_groq_key_from_file()
