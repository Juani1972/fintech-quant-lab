"""
tests/test_license.py

Tests unitarios de app/core/license.py. `requests` está mockeado, así
que no hace ninguna llamada de red real.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from app.core.license import (
    LicenseError,
    LicenseInfo,
    cache_license,
    is_license_valid,
    load_cached_license,
    machine_fingerprint,
    verify_license,
)


def _future_date(days: int) -> str:
    return (date.today() + timedelta(days=days)).isoformat()


def _iso_now(days_ago: int = 0) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()


# ---------------------------------------------------------------------------
# machine_fingerprint
# ---------------------------------------------------------------------------

class TestMachineFingerprint:
    def test_returns_stable_sha256_hex(self):
        fp1 = machine_fingerprint()
        fp2 = machine_fingerprint()

        assert fp1 == fp2
        assert len(fp1) == 64
        int(fp1, 16)  # no debe lanzar: es hexadecimal válido


# ---------------------------------------------------------------------------
# verify_license
# ---------------------------------------------------------------------------

class TestVerifyLicense:
    def test_valid_key_returns_license_info(self):
        fake_response = MagicMock()
        fake_response.json.return_value = {
            "valid": True,
            "expires_at": _future_date(30),
            "plan": "pro",
        }
        fake_response.raise_for_status.return_value = None

        with patch("requests.post", return_value=fake_response):
            info = verify_license("ABCD-1234", "https://licencias.example.com")

        assert info.valid is True
        assert info.plan == "pro"
        assert info.machine_fingerprint == machine_fingerprint()

    def test_empty_key_raises(self):
        with pytest.raises(LicenseError, match="no puede estar vacía"):
            verify_license("   ", "https://licencias.example.com")

    def test_rejected_key_raises(self):
        fake_response = MagicMock()
        fake_response.json.return_value = {"valid": False, "reason": "clave revocada"}
        fake_response.raise_for_status.return_value = None

        with patch("requests.post", return_value=fake_response):
            with pytest.raises(LicenseError, match="clave revocada"):
                verify_license("ABCD-1234", "https://licencias.example.com")

    def test_missing_expiration_raises(self):
        fake_response = MagicMock()
        fake_response.json.return_value = {"valid": True}
        fake_response.raise_for_status.return_value = None

        with patch("requests.post", return_value=fake_response):
            with pytest.raises(LicenseError, match="fecha de expiración"):
                verify_license("ABCD-1234", "https://licencias.example.com")

    def test_network_error_raises_license_error(self):
        import requests

        with patch("requests.post", side_effect=requests.ConnectionError("sin red")):
            with pytest.raises(LicenseError, match="No se pudo contactar"):
                verify_license("ABCD-1234", "https://licencias.example.com")

    def test_trailing_slash_in_server_url_is_handled(self):
        fake_response = MagicMock()
        fake_response.json.return_value = {"valid": True, "expires_at": _future_date(10)}
        fake_response.raise_for_status.return_value = None

        with patch("requests.post", return_value=fake_response) as mocked:
            verify_license("ABCD-1234", "https://licencias.example.com/")

        called_url = mocked.call_args[0][0]
        assert called_url == "https://licencias.example.com/verify"


# ---------------------------------------------------------------------------
# is_license_valid
# ---------------------------------------------------------------------------

class TestIsLicenseValid:
    def _make_info(self, expires_in_days: int, checked_days_ago: int = 0, valid: bool = True) -> LicenseInfo:
        return LicenseInfo(
            key="ABCD-1234",
            valid=valid,
            expires_at=_future_date(expires_in_days),
            plan="pro",
            machine_fingerprint="fake-fp",
            checked_at=_iso_now(checked_days_ago),
        )

    def test_not_expired_and_recently_checked_is_valid(self):
        info = self._make_info(expires_in_days=30, checked_days_ago=1)
        assert is_license_valid(info, grace_days=7) is True

    def test_expired_and_recently_checked_is_invalid(self):
        info = self._make_info(expires_in_days=-1, checked_days_ago=1)
        assert is_license_valid(info, grace_days=7) is False

    def test_not_expired_but_checked_too_long_ago_is_invalid(self):
        info = self._make_info(expires_in_days=60, checked_days_ago=10)
        assert is_license_valid(info, grace_days=7) is False

    def test_invalid_flag_is_always_invalid(self):
        info = self._make_info(expires_in_days=30, checked_days_ago=0, valid=False)
        assert is_license_valid(info, grace_days=7) is False


# ---------------------------------------------------------------------------
# cache_license / load_cached_license
# ---------------------------------------------------------------------------

class TestCaching:
    def test_roundtrip(self, tmp_path):
        info = LicenseInfo(
            key="ABCD-1234",
            valid=True,
            expires_at=_future_date(30),
            plan="pro",
            machine_fingerprint="fake-fp",
            checked_at=_iso_now(),
        )
        path = tmp_path / "license.json"

        cache_license(info, path)
        loaded = load_cached_license(path)

        assert loaded == info

    def test_missing_file_returns_none(self, tmp_path):
        assert load_cached_license(tmp_path / "no-existe.json") is None

    def test_corrupt_file_returns_none(self, tmp_path):
        path = tmp_path / "license.json"
        path.write_text("esto no es json", encoding="utf-8")

        assert load_cached_license(path) is None

    def test_creates_parent_directories(self, tmp_path):
        info = LicenseInfo(
            key="ABCD-1234",
            valid=True,
            expires_at=_future_date(30),
            plan="pro",
            machine_fingerprint="fake-fp",
            checked_at=_iso_now(),
        )
        path = tmp_path / "nested" / "dir" / "license.json"

        cache_license(info, path)

        assert path.exists()
