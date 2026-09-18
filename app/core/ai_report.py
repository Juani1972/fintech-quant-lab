"""Cliente de la API de Gemini (Google AI Studio) e informes con IA.

Diseño deliberado -- cada USUARIO pega su propia clave (igual que ya
se hace con Alpha Vantage en app.core.providers), no una clave
compartida en el servidor: esta app es de escritorio, cada usuario la
ejecuta en su propio equipo, y así el coste/cuota de las llamadas a
la IA corre a cargo de cada uno con su propio nivel gratuito, no del
desarrollador.

Usa el endpoint REST clásico `generateContent` (`requests`, sin el
SDK oficial `google-generativeai`) para no añadir una dependencia
pesada -- coherente con cómo ya está resuelto el resto de conectores
externos del proyecto (providers.py, license.py).
"""
from __future__ import annotations

import requests

DEFAULT_MODEL = "gemini-2.5-flash"
API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_TIMEOUT = 30


class AIReportError(Exception):
    """Error al generar un informe con IA (clave inválida, cuota
    agotada, respuesta inesperada de la API, etc.)."""


def generate_report(
    prompt: str,
    api_key: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    timeout: int = DEFAULT_TIMEOUT,
) -> str:
    """Genera texto con Gemini a partir de un prompt ya construido.

    Args:
        prompt: el prompt completo (instrucciones + datos a
            interpretar), ya en texto plano.
        api_key: clave de API de Google AI Studio del propio usuario.
        model: nombre del modelo. Por defecto 'gemini-2.5-flash'
            (gratuito en Google AI Studio a fecha de esta versión --
            los nombres y niveles gratuitos de Gemini cambian con
            cierta frecuencia, así que se puede indicar otro).
        temperature: 0.0-1.0 -- más bajo = más determinista/ceñido a
            los datos, más alto = más "creativo". 0.3 por defecto:
            para un informe financiero interesa que se ciña a los
            números, no que divague.
        timeout: segundos antes de abandonar la petición.

    Returns:
        El texto generado.

    Raises:
        AIReportError: si falta la clave, la API devuelve un error
            (clave inválida, cuota agotada, prompt bloqueado por los
            filtros de seguridad...), o la respuesta no tiene el
            formato esperado.
    """
    if not api_key or not api_key.strip():
        raise AIReportError(
            "No hay clave de Google AI Studio configurada. Consigue una "
            "gratis en https://aistudio.google.com/apikey y pégala en "
            "la configuración de la barra lateral."
        )

    url = f"{API_BASE_URL}/{model}:generateContent"
    headers = {"x-goog-api-key": api_key.strip(), "Content-Type": "application/json"}
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature},
    }

    try:
        resp = requests.post(url, headers=headers, json=body, timeout=timeout)
    except requests.RequestException as exc:
        raise AIReportError(f"No se pudo contactar con la API de Gemini: {exc}") from exc

    if resp.status_code == 400:
        raise AIReportError(
            "Petición rechazada (400) -- normalmente significa clave de "
            "API con formato inválido, o el modelo indicado no existe."
        )
    if resp.status_code in (401, 403):
        raise AIReportError(
            "Clave de API rechazada (401/403) -- revisa que la has "
            "copiado bien desde https://aistudio.google.com/apikey."
        )
    if resp.status_code == 429:
        raise AIReportError(
            "Límite de peticiones alcanzado (429) -- el nivel gratuito "
            "de Google AI Studio tiene un tope diario/por minuto. "
            "Espera un poco y vuelve a intentarlo."
        )
    if resp.status_code != 200:
        raise AIReportError(
            f"La API de Gemini devolvió un error ({resp.status_code}): {resp.text[:300]}"
        )

    try:
        data = resp.json()
    except ValueError as exc:
        raise AIReportError("La respuesta de Gemini no es JSON válido.") from exc

    candidates = data.get("candidates") or []
    if not candidates:
        # Puede pasar si el prompt lo bloquean los filtros de seguridad
        # de Google (promptFeedback.blockReason) en vez de devolver un
        # candidato -- se distingue del caso genérico para dar un
        # mensaje más útil que "respuesta vacía".
        block_reason = (data.get("promptFeedback") or {}).get("blockReason")
        if block_reason:
            raise AIReportError(
                f"Gemini bloqueó la petición por sus filtros de seguridad "
                f"({block_reason}). Esto no debería pasar con datos "
                f"financieros normales; si persiste, revisa qué se está "
                f"enviando en el prompt."
            )
        raise AIReportError("Gemini no devolvió ningún resultado (respuesta vacía).")

    try:
        parts = candidates[0]["content"]["parts"]
        text = "".join(p.get("text", "") for p in parts)
    except (KeyError, IndexError, TypeError) as exc:
        raise AIReportError(
            f"Formato de respuesta de Gemini inesperado: {exc}"
        ) from exc

    if not text.strip():
        raise AIReportError("Gemini devolvió una respuesta vacía.")

    return text.strip()
