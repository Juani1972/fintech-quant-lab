"""Cliente de la API de Groq (compatible con OpenAI) e informes con IA.

Alternativa a Gemini (ver app/core/ai_report.py) para cuando el nivel
gratuito de Google está saturado o deja de estar disponible -- Groq
ofrece inferencia rápida sobre modelos abiertos (Llama, Gemma...) con
una clave de API gratuita (console.groq.com/keys, sin tarjeta).

Mismo diseño deliberado que Gemini: cada usuario pega su propia clave,
no hay clave compartida en el servidor. Mismas firmas de función que
ai_report.py (generate_report, list_available_models, AIReportError
reutilizada de allí) para que app/state.py pueda despachar entre los
dos backends sin que las páginas (GARCH, Backtest...) necesiten saber
cuál es cuál.

Usa el endpoint REST `chat/completions` (`requests`, sin el SDK
oficial) -- Groq expone la misma forma que la API de Chat Completions
de OpenAI, así que la petición/respuesta siguen ese formato en vez
del de Gemini.
"""
from __future__ import annotations

import time

import requests

from app.core.ai_report import AIReportError

DEFAULT_MODEL = "llama-3.3-70b-versatile"
API_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_TIMEOUT = 30

# Mismo criterio que en ai_report.py: 500/503 son sobrecarga
# transitoria del servicio, vale la pena reintentar. 429 (cuota) y el
# resto de errores no se resuelven solos con un reintento inmediato.
_TRANSIENT_STATUS_CODES = {500, 503}
_RETRY_DELAYS_SECONDS = (2, 4, 8)


def generate_report(
    prompt: str,
    api_key: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    timeout: int = DEFAULT_TIMEOUT,
) -> str:
    """Genera texto con Groq a partir de un prompt ya construido.

    Args:
        prompt: el prompt completo (instrucciones + datos a
            interpretar), ya en texto plano.
        api_key: clave de API de Groq del propio usuario.
        model: nombre del modelo. Por defecto 'llama-3.3-70b-versatile'
            -- igual que con Gemini, los modelos gratuitos disponibles
            cambian con el tiempo, así que se puede indicar otro desde
            la barra lateral (ver list_available_models).
        temperature: 0.0-1.0 -- más bajo = más determinista/ceñido a
            los datos, más alto = más "creativo".
        timeout: segundos antes de abandonar la petición.

    Returns:
        El texto generado.

    Raises:
        AIReportError: si falta la clave, la API devuelve un error, o
            la respuesta no tiene el formato esperado. Los errores
            500/503 se reintentan solos con espera creciente antes de
            propagar el error (ver ai_report.generate_report).
    """
    if not api_key or not api_key.strip():
        raise AIReportError(
            "No hay clave de Groq configurada. Consigue una gratis en "
            "https://console.groq.com/keys y pégala en la "
            "configuración de la barra lateral."
        )

    url = f"{API_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }

    max_attempts = len(_RETRY_DELAYS_SECONDS) + 1
    for delay in (0, *_RETRY_DELAYS_SECONDS):
        if delay:
            time.sleep(delay)
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=timeout)
        except requests.RequestException as exc:
            raise AIReportError(f"No se pudo contactar con la API de Groq: {exc}") from exc

        if resp.status_code not in _TRANSIENT_STATUS_CODES:
            break

    if resp.status_code in _TRANSIENT_STATUS_CODES:
        raise AIReportError(
            f"La API de Groq sigue sobrecargada tras {max_attempts} intentos "
            f"({resp.status_code}). Espera un poco más y vuelve a intentarlo."
        )

    if resp.status_code == 401:
        raise AIReportError(
            "Clave de API de Groq rechazada (401) -- revisa que la has "
            "copiado bien desde https://console.groq.com/keys."
        )
    if resp.status_code == 429:
        raise AIReportError(
            "Límite de peticiones alcanzado (429) -- el nivel gratuito "
            "de Groq tiene un tope diario/por minuto. Espera un poco y "
            "vuelve a intentarlo."
        )
    if resp.status_code in (400, 404):
        raise AIReportError(
            f"El modelo '{model}' no existe o la petición fue rechazada "
            f"({resp.status_code}). Prueba con otro nombre en el campo "
            "'Modelo' de la barra lateral, o pulsa 'Ver modelos "
            f"disponibles':\n\n{resp.text[:300]}"
        )
    if resp.status_code != 200:
        raise AIReportError(
            f"La API de Groq devolvió un error ({resp.status_code}): {resp.text[:300]}"
        )

    try:
        data = resp.json()
    except ValueError as exc:
        raise AIReportError("La respuesta de Groq no es JSON válido.") from exc

    choices = data.get("choices") or []
    if not choices:
        raise AIReportError("Groq no devolvió ningún resultado (respuesta vacía).")

    try:
        text = str(choices[0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as exc:
        raise AIReportError(f"Formato de respuesta de Groq inesperado: {exc}") from exc

    if not text or not text.strip():
        raise AIReportError("Groq devolvió una respuesta vacía.")

    return text.strip()


def list_available_models(api_key: str, timeout: int = DEFAULT_TIMEOUT) -> list[str]:
    """Lista los modelos que admite esta clave de Groq ahora mismo.

    Returns:
        Nombres de modelo, ordenados alfabéticamente.

    Raises:
        AIReportError: si falta la clave o la API devuelve un error.
    """
    if not api_key or not api_key.strip():
        raise AIReportError(
            "No hay clave de Groq configurada. Consigue una gratis en "
            "https://console.groq.com/keys y pégala en la "
            "configuración de la barra lateral."
        )

    headers = {"Authorization": f"Bearer {api_key.strip()}"}

    try:
        resp = requests.get(f"{API_BASE_URL}/models", headers=headers, timeout=timeout)
    except requests.RequestException as exc:
        raise AIReportError(f"No se pudo contactar con la API de Groq: {exc}") from exc

    if resp.status_code == 401:
        raise AIReportError(
            "Clave de API de Groq rechazada (401) -- revisa que la has "
            "copiado bien desde https://console.groq.com/keys."
        )
    if resp.status_code != 200:
        raise AIReportError(
            f"La API de Groq devolvió un error ({resp.status_code}): {resp.text[:300]}"
        )

    try:
        data = resp.json()
    except ValueError as exc:
        raise AIReportError("La respuesta de Groq no es JSON válido.") from exc

    models = [str(m.get("id", "")) for m in data.get("data", []) if m.get("id")]
    return sorted(set(models))
