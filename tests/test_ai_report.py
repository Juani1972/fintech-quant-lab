"""
tests/test_ai_report.py

Tests unitarios de app/core/ai_report.py. `requests.post` está
mockeado en todos los casos -- no se necesita (ni se usa) una clave
de API real de Google AI Studio.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.core.ai_report import AIReportError, generate_report


def _fake_response(status_code=200, json_data=None, text=""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data if json_data is not None else {}
    resp.text = text
    return resp


def _success_payload(text="Informe generado de prueba."):
    return {
        "candidates": [
            {"content": {"parts": [{"text": text}]}},
        ],
    }


def test_generate_report_success():
    with patch("app.core.ai_report.requests.post", return_value=_fake_response(
        200, _success_payload("Este backtest muestra un Sharpe sólido."),
    )) as mock_post:
        result = generate_report("Interpreta este backtest.", api_key="clave-123")

    assert result == "Este backtest muestra un Sharpe sólido."
    # Confirmar que la clave va en el header correcto, no en la URL/body
    call_kwargs = mock_post.call_args.kwargs
    assert call_kwargs["headers"]["x-goog-api-key"] == "clave-123"
    assert "gemini-2.5-flash" in mock_post.call_args.args[0]


def test_generate_report_joins_multiple_parts():
    """La respuesta puede venir partida en varios 'parts' -- deben
    concatenarse, no quedarse solo con el primero."""
    payload = {
        "candidates": [
            {"content": {"parts": [{"text": "Parte uno. "}, {"text": "Parte dos."}]}},
        ],
    }
    with patch("app.core.ai_report.requests.post", return_value=_fake_response(200, payload)):
        result = generate_report("prompt", api_key="clave-123")
    assert result == "Parte uno. Parte dos."


def test_generate_report_missing_api_key_raises_without_network_call():
    with patch("app.core.ai_report.requests.post") as mock_post, pytest.raises(AIReportError, match="No hay clave"):
            generate_report("prompt", api_key="")
    mock_post.assert_not_called()

    with patch("app.core.ai_report.requests.post") as mock_post2, pytest.raises(AIReportError, match="No hay clave"):
            generate_report("prompt", api_key="   ")
    mock_post2.assert_not_called()


def test_generate_report_invalid_key_401():
    with patch("app.core.ai_report.requests.post", return_value=_fake_response(401)), pytest.raises(AIReportError, match="Clave de API rechazada"):
            generate_report("prompt", api_key="clave-mala")


def test_generate_report_invalid_key_403():
    with patch("app.core.ai_report.requests.post", return_value=_fake_response(403)), pytest.raises(AIReportError, match="Clave de API rechazada"):
            generate_report("prompt", api_key="clave-mala")


def test_generate_report_rate_limited_429():
    with patch("app.core.ai_report.requests.post", return_value=_fake_response(429)), pytest.raises(AIReportError, match="Límite de peticiones"):
            generate_report("prompt", api_key="clave-123")


def test_generate_report_bad_request_400():
    with patch("app.core.ai_report.requests.post", return_value=_fake_response(400)), pytest.raises(AIReportError, match="Petición rechazada"):
            generate_report("prompt", api_key="clave-123", model="modelo-que-no-existe")


def test_generate_report_generic_http_error():
    with patch(
        "app.core.ai_report.requests.post",
        return_value=_fake_response(503, text="Service unavailable"),
    ), pytest.raises(AIReportError, match="503"):
        generate_report("prompt", api_key="clave-123")


def test_generate_report_network_exception():
    import requests

    with patch("app.core.ai_report.requests.post", side_effect=requests.ConnectionError("sin red")), pytest.raises(AIReportError, match="No se pudo contactar"):
            generate_report("prompt", api_key="clave-123")


def test_generate_report_blocked_by_safety_filters():
    payload = {"candidates": [], "promptFeedback": {"blockReason": "SAFETY"}}
    with patch("app.core.ai_report.requests.post", return_value=_fake_response(200, payload)), pytest.raises(AIReportError, match="filtros de seguridad"):
            generate_report("prompt", api_key="clave-123")


def test_generate_report_empty_candidates_no_reason():
    payload = {"candidates": []}
    with patch("app.core.ai_report.requests.post", return_value=_fake_response(200, payload)), pytest.raises(AIReportError, match="respuesta vacía"):
            generate_report("prompt", api_key="clave-123")


def test_generate_report_malformed_response_shape():
    payload = {"candidates": [{"content": {}}]}  # sin "parts"
    with patch("app.core.ai_report.requests.post", return_value=_fake_response(200, payload)), pytest.raises(AIReportError, match="[Ff]ormato de respuesta"):
            generate_report("prompt", api_key="clave-123")


def test_generate_report_not_json_response():
    resp = MagicMock()
    resp.status_code = 200
    resp.json.side_effect = ValueError("not json")
    with patch("app.core.ai_report.requests.post", return_value=resp), pytest.raises(AIReportError, match="no es JSON válido"):
            generate_report("prompt", api_key="clave-123")


def test_generate_report_blank_text_response():
    payload = {"candidates": [{"content": {"parts": [{"text": "   "}]}}]}
    with patch("app.core.ai_report.requests.post", return_value=_fake_response(200, payload)), pytest.raises(AIReportError, match="respuesta vacía"):
            generate_report("prompt", api_key="clave-123")


def test_generate_report_custom_model_used_in_url():
    with patch("app.core.ai_report.requests.post", return_value=_fake_response(
        200, _success_payload(),
    )) as mock_post:
        generate_report("prompt", api_key="clave-123", model="gemini-2.5-pro")
    assert "gemini-2.5-pro:generateContent" in mock_post.call_args.args[0]


def test_generate_report_temperature_passed_through():
    with patch("app.core.ai_report.requests.post", return_value=_fake_response(
        200, _success_payload(),
    )) as mock_post:
        generate_report("prompt", api_key="clave-123", temperature=0.7)
    body = mock_post.call_args.kwargs["json"]
    assert body["generationConfig"]["temperature"] == 0.7
