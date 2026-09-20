"""
tests/test_groq_report.py

Tests unitarios de app/core/groq_report.py. `requests.post`/`.get`
están mockeados en todos los casos -- no se necesita (ni se usa) una
clave de API real de Groq.

Estructura deliberadamente paralela a tests/test_ai_report.py: mismas
firmas de función en ambos backends (ver app/state.py::generate_ai_report),
así que sus tests deben cubrir los mismos casos.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.core.groq_report import AIReportError, generate_report, list_available_models


def _fake_response(status_code=200, json_data=None, text=""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data if json_data is not None else {}
    resp.text = text
    return resp


def _success_payload(text="Informe generado de prueba."):
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


def test_generate_report_success():
    with patch("app.core.groq_report.requests.post", return_value=_fake_response(
        200, _success_payload("Este backtest muestra un Sharpe sólido."),
    )) as mock_post:
        result = generate_report("Interpreta este backtest.", api_key="clave-123")

    assert result == "Este backtest muestra un Sharpe sólido."
    call_kwargs = mock_post.call_args.kwargs
    assert call_kwargs["headers"]["Authorization"] == "Bearer clave-123"
    assert mock_post.call_args.args[0] == "https://api.groq.com/openai/v1/chat/completions"
    assert call_kwargs["json"]["messages"] == [
        {"role": "user", "content": "Interpreta este backtest."}
    ]


def test_generate_report_missing_api_key_raises_without_network_call():
    with patch("app.core.groq_report.requests.post") as mock_post, pytest.raises(
        AIReportError, match="No hay clave"
    ):
        generate_report("prompt", api_key="")
    mock_post.assert_not_called()

    with patch("app.core.groq_report.requests.post") as mock_post2, pytest.raises(
        AIReportError, match="No hay clave"
    ):
        generate_report("prompt", api_key="   ")
    mock_post2.assert_not_called()


def test_generate_report_invalid_key_401():
    with patch(
        "app.core.groq_report.requests.post", return_value=_fake_response(401),
    ), pytest.raises(AIReportError, match="Clave de API de Groq rechazada"):
        generate_report("prompt", api_key="clave-mala")


def test_generate_report_rate_limited_429():
    with patch(
        "app.core.groq_report.requests.post", return_value=_fake_response(429),
    ), pytest.raises(AIReportError, match="Límite de peticiones"):
        generate_report("prompt", api_key="clave-123")


def test_generate_report_model_not_found():
    with patch(
        "app.core.groq_report.requests.post",
        return_value=_fake_response(404, text='{"error": {"message": "model not found"}}'),
    ), pytest.raises(AIReportError, match="modelo-viejo"):
        generate_report("prompt", api_key="clave-123", model="modelo-viejo")


def test_generate_report_generic_http_error():
    # 502, no 503/500 -- esos se reintentan (ver tests de abajo) y
    # este test no mockea time.sleep.
    with patch(
        "app.core.groq_report.requests.post",
        return_value=_fake_response(502, text="Bad Gateway"),
    ), pytest.raises(AIReportError, match="502"):
        generate_report("prompt", api_key="clave-123")


def test_generate_report_network_exception():
    import requests

    with patch(
        "app.core.groq_report.requests.post",
        side_effect=requests.ConnectionError("sin red"),
    ), pytest.raises(AIReportError, match="No se pudo contactar"):
        generate_report("prompt", api_key="clave-123")


def test_generate_report_empty_choices():
    payload = {"choices": []}
    with patch(
        "app.core.groq_report.requests.post", return_value=_fake_response(200, payload),
    ), pytest.raises(AIReportError, match="respuesta vacía"):
        generate_report("prompt", api_key="clave-123")


def test_generate_report_malformed_response_shape():
    payload = {"choices": [{"message": {}}]}  # sin "content"
    with patch(
        "app.core.groq_report.requests.post", return_value=_fake_response(200, payload),
    ), pytest.raises(AIReportError, match="[Ff]ormato de respuesta"):
        generate_report("prompt", api_key="clave-123")


def test_generate_report_not_json_response():
    resp = MagicMock()
    resp.status_code = 200
    resp.json.side_effect = ValueError("not json")
    with patch(
        "app.core.groq_report.requests.post", return_value=resp,
    ), pytest.raises(AIReportError, match="no es JSON válido"):
        generate_report("prompt", api_key="clave-123")


def test_generate_report_blank_text_response():
    payload = {"choices": [{"message": {"content": "   "}}]}
    with patch(
        "app.core.groq_report.requests.post", return_value=_fake_response(200, payload),
    ), pytest.raises(AIReportError, match="respuesta vacía"):
        generate_report("prompt", api_key="clave-123")


def test_generate_report_custom_model_used_in_body():
    with patch("app.core.groq_report.requests.post", return_value=_fake_response(
        200, _success_payload(),
    )) as mock_post:
        generate_report("prompt", api_key="clave-123", model="mixtral-8x7b-32768")
    assert mock_post.call_args.kwargs["json"]["model"] == "mixtral-8x7b-32768"


def test_generate_report_temperature_passed_through():
    with patch("app.core.groq_report.requests.post", return_value=_fake_response(
        200, _success_payload(),
    )) as mock_post:
        generate_report("prompt", api_key="clave-123", temperature=0.7)
    assert mock_post.call_args.kwargs["json"]["temperature"] == 0.7


# ============================================================
#  Reintentos en errores transitorios (500/503)
# ============================================================
def test_generate_report_retries_503_then_succeeds():
    responses = [
        _fake_response(503), _fake_response(503), _fake_response(200, _success_payload("ok")),
    ]
    with (
        patch("app.core.groq_report.time.sleep") as mock_sleep,
        patch("app.core.groq_report.requests.post", side_effect=responses) as mock_post,
    ):
        result = generate_report("prompt", api_key="clave-123")

    assert result == "ok"
    assert mock_post.call_count == 3
    assert mock_sleep.call_count == 2


def test_generate_report_gives_up_after_exhausting_retries():
    with (
        patch("app.core.groq_report.time.sleep") as mock_sleep,
        patch(
            "app.core.groq_report.requests.post",
            return_value=_fake_response(503, text="still overloaded"),
        ) as mock_post,
        pytest.raises(AIReportError, match="sigue sobrecargada"),
    ):
        generate_report("prompt", api_key="clave-123")

    assert mock_post.call_count == 4
    assert mock_sleep.call_count == 3


def test_generate_report_does_not_retry_non_transient_errors():
    with (
        patch("app.core.groq_report.time.sleep") as mock_sleep,
        patch("app.core.groq_report.requests.post", return_value=_fake_response(429)) as mock_post,
        pytest.raises(AIReportError, match="Límite de peticiones"),
    ):
        generate_report("prompt", api_key="clave-123")

    assert mock_post.call_count == 1
    mock_sleep.assert_not_called()


# ============================================================
#  list_available_models
# ============================================================
def test_list_available_models_returns_sorted_ids():
    payload = {
        "data": [
            {"id": "llama-3.3-70b-versatile"},
            {"id": "gemma2-9b-it"},
        ],
    }
    with patch(
        "app.core.groq_report.requests.get", return_value=_fake_response(200, payload),
    ) as mock_get:
        models = list_available_models("clave-123")

    assert models == ["gemma2-9b-it", "llama-3.3-70b-versatile"]
    assert mock_get.call_args.kwargs["headers"]["Authorization"] == "Bearer clave-123"


def test_list_available_models_missing_key_raises_without_network_call():
    with patch("app.core.groq_report.requests.get") as mock_get, pytest.raises(
        AIReportError, match="No hay clave"
    ):
        list_available_models("")
    mock_get.assert_not_called()


def test_list_available_models_invalid_key():
    with patch(
        "app.core.groq_report.requests.get", return_value=_fake_response(401),
    ), pytest.raises(AIReportError, match="Clave de API de Groq rechazada"):
        list_available_models("clave-mala")
