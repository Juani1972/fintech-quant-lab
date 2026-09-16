"""
app/core/license.py

Sistema de licencias para vender Fintech Quant Lab como producto
cerrado: fingerprint de máquina, verificación contra un servidor de
licencias propio, y caché offline para que el cliente no dependa de
tener conexión al servidor en cada arranque.

Uso típico:

    from app.core.license import (
        machine_fingerprint, verify_license, is_license_valid,
        cache_license, load_cached_license,
    )
    from pathlib import Path

    fp = machine_fingerprint()
    try:
        info = verify_license("XXXX-YYYY-ZZZZ", "https://licencias.tudominio.com")
        cache_license(info, Path("~/.fintech_quant_lab/license.json").expanduser())
    except LicenseError:
        info = load_cached_license(Path("~/.fintech_quant_lab/license.json").expanduser())

    if not is_license_valid(info):
        raise SystemExit("Licencia caducada.")
"""

from __future__ import annotations

import hashlib
import json
import platform
import uuid
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path


class LicenseError(Exception):
    """Error genérico al verificar, cargar o guardar una licencia."""


@dataclass
class LicenseInfo:
    """Resultado de verificar una clave de licencia."""

    key: str
    valid: bool
    expires_at: str  # fecha ISO 8601 (YYYY-MM-DD)
    plan: str
    machine_fingerprint: str
    checked_at: str  # timestamp ISO 8601 de cuándo se verificó (online u offline)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "LicenseInfo":
        return cls(**data)

    @property
    def expires_date(self) -> date:
        return date.fromisoformat(self.expires_at)


def machine_fingerprint() -> str:
    """
    Genera un hash estable que identifica esta máquina (MAC + hostname
    + info de CPU), para atar una licencia a un equipo concreto.

    Returns:
        Hash SHA-256 en hexadecimal (64 caracteres).
    """
    mac = uuid.getnode()
    hostname = platform.node()
    cpu = platform.processor() or platform.machine()

    raw = f"{mac}|{hostname}|{cpu}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def verify_license(
    key: str,
    server_url: str,
    timeout: float = 5.0,
) -> LicenseInfo:
    """
    Verifica una clave de licencia contra el servidor de licencias.

    Args:
        key: clave de licencia introducida por el cliente.
        server_url: URL base del servidor de licencias (se le hace un
            POST a `{server_url}/verify`).
        timeout: timeout de la petición HTTP, en segundos.

    Returns:
        LicenseInfo con el resultado de la verificación.

    Raises:
        LicenseError: si la petición falla (red, timeout, respuesta
            inválida) o el servidor rechaza la clave.
    """
    try:
        import requests
    except ImportError as exc:  # pragma: no cover
        raise LicenseError("requests no está instalado. Añádelo a requirements.txt.") from exc

    if not key or not key.strip():
        raise LicenseError("La clave de licencia no puede estar vacía.")

    fingerprint = machine_fingerprint()

    try:
        resp = requests.post(
            f"{server_url.rstrip('/')}/verify",
            json={"key": key, "fingerprint": fingerprint},
            timeout=timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
    except requests.RequestException as exc:
        raise LicenseError(f"No se pudo contactar con el servidor de licencias: {exc}") from exc
    except ValueError as exc:
        raise LicenseError(f"Respuesta del servidor de licencias no es JSON válido: {exc}") from exc

    if not payload.get("valid", False):
        motivo = payload.get("reason", "clave no válida")
        raise LicenseError(f"Licencia rechazada por el servidor: {motivo}")

    expires_at = payload.get("expires_at")
    if not expires_at:
        raise LicenseError("El servidor no devolvió una fecha de expiración.")

    return LicenseInfo(
        key=key,
        valid=True,
        expires_at=expires_at,
        plan=payload.get("plan", "standard"),
        machine_fingerprint=fingerprint,
        checked_at=datetime.now(timezone.utc).isoformat(),
    )


def is_license_valid(info: LicenseInfo, grace_days: int = 7) -> bool:
    """
    Comprueba si una licencia sigue siendo válida hoy, con un margen
    de gracia sobre `checked_at` para tolerar que el cliente lleve
    unos días sin poder verificar online contra el servidor.

    Args:
        info: resultado de `verify_license` o `load_cached_license`.
        grace_days: días de margen tras `checked_at` durante los que
            la licencia se sigue considerando válida sin reverificar,
            incluso si ya se pasó `expires_at`... salvo que ya hayan
            pasado más de `grace_days` desde la última verificación,
            en cuyo caso se exige haber verificado antes de esa fecha.

    Returns:
        True si la licencia es válida hoy (considerando el margen de gracia).
    """
    if not info.valid:
        return False

    today = datetime.now(timezone.utc).date()

    try:
        checked_at = datetime.fromisoformat(info.checked_at).date()
    except ValueError:
        checked_at = today

    days_since_check = (today - checked_at).days
    if days_since_check > grace_days:
        return False

    return today <= info.expires_date


def cache_license(info: LicenseInfo, path: Path) -> None:
    """
    Guarda una LicenseInfo en disco, para poder operar offline dentro
    del margen de gracia sin volver a contactar con el servidor.

    Args:
        info: resultado de `verify_license` a cachear.
        path: ruta del fichero donde guardar la caché (se crean las
            carpetas intermedias si no existen).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


def load_cached_license(path: Path) -> LicenseInfo | None:
    """
    Carga una LicenseInfo previamente guardada con `cache_license`.

    Args:
        path: ruta del fichero de caché.

    Returns:
        La LicenseInfo cacheada, o None si el fichero no existe o está
        corrupto (JSON inválido o le faltan campos).
    """
    path = Path(path)
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return LicenseInfo.from_dict(data)
    except (json.JSONDecodeError, TypeError, KeyError):
        return None
