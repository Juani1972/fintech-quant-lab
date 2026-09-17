"""Página de estado y activación de licencia."""
from pathlib import Path

import streamlit as st

from app.core.license import (
    LicenseError,
    cache_license,
    is_license_valid,
    load_cached_license,
    machine_fingerprint,
    verify_license,
)
from app.styles import callout, footer, hero, page_setup, section

page_setup("Licencia", "🔑")

hero(
    title="Licencia",
    subtitle=(
        "Verificación de licencia con caché offline (margen de gracia) y "
        "huella de máquina, para atar una licencia a un equipo concreto."
    ),
    icon="🔑",
)

CACHE_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "license_cache.json"

callout(
    "Este proyecto todavía no tiene un servidor de licencias real "
    "desplegado en ningún sitio -- esta página conecta el código ya "
    "escrito (verificación online, caché offline con margen de gracia, "
    "huella de máquina) para el día que montes ese backend. Hasta "
    "entonces, verificar contra cualquier URL dará un error de conexión "
    "esperado (no hay nada escuchando ahí), no un fallo de esta app.",
    variant="warning",
)

section("🖥️ Huella de esta máquina")
fingerprint = machine_fingerprint()
st.code(fingerprint, language=None)
st.caption(
    "Hash estable derivado de la MAC, el hostname y la CPU de este "
    "equipo -- es lo que un servidor de licencias real usaría para "
    "atar una clave a esta máquina concreta y detectar activaciones "
    "en equipos no autorizados."
)

section("📋 Estado actual")
cached = load_cached_license(CACHE_PATH)
if cached is None:
    callout("No hay ninguna licencia guardada en este equipo todavía.", variant="info")
else:
    valid = is_license_valid(cached)
    col1, col2, col3 = st.columns(3)
    col1.metric("Plan", cached.plan)
    col2.metric("Expira", cached.expires_at)
    col3.metric("Estado", "✅ Válida" if valid else "❌ Expirada / no reverificada")
    st.caption(f"Última verificación: {cached.checked_at}")
    if cached.machine_fingerprint != fingerprint:
        callout(
            "Esta licencia se verificó desde OTRA máquina (huella "
            "distinta a la de este equipo) -- probablemente no es válida "
            "aquí.",
            variant="warning",
        )

section("🔐 Verificar / activar una clave")
with st.form("verify_license_form"):
    key = st.text_input("Clave de licencia", type="password")
    server_url = st.text_input(
        "URL del servidor de licencias",
        placeholder="https://licencias.tudominio.com",
        help="Se hace un POST a {esta_url}/verify con la clave y la huella de máquina.",
    )
    submitted = st.form_submit_button("🔐 Verificar", type="primary")

if submitted:
    if not key or not server_url:
        st.error("Indica la clave y la URL del servidor.")
    else:
        try:
            info = verify_license(key, server_url)
        except LicenseError as e:
            st.error(f"No se pudo verificar: {e}")
        else:
            try:
                cache_license(info, CACHE_PATH)
            except OSError as e:
                st.warning(f"Verificada pero no se pudo guardar en caché: {e}")
            st.success(
                f"Licencia verificada. Plan: {info.plan}, expira: {info.expires_at}."
            )
            st.rerun()

footer()
