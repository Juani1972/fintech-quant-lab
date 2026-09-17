"""Smoke tests de la aplicación Streamlit con AppTest.

Verifica que las páginas cargan sin lanzar excepciones.
Requiere streamlit >= 1.28.
"""
from __future__ import annotations

from pathlib import Path

import pytest

streamlit = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP_DIR = Path(__file__).parent.parent / "app"


def test_main_loads_without_exception():
    """La portada debe cargar sin lanzar excepción."""
    at = AppTest.from_file(str(APP_DIR / "main.py"), default_timeout=30)
    at.run()
    assert not at.exception, f"Excepción en main.py: {at.exception}"


def test_main_has_expected_metrics():
    """La portada debe mostrar las 4 métricas del estado de sesión."""
    at = AppTest.from_file(str(APP_DIR / "main.py"), default_timeout=30)
    at.run()
    # Debe haber al menos 4 métricas (Tickers, Fecha inicio, Fecha fin, Rango)
    assert len(at.metric) >= 4


def test_sidebar_has_global_inputs():
    """La barra lateral debe tener el input de tickers y 2 date_inputs."""
    at = AppTest.from_file(str(APP_DIR / "main.py"), default_timeout=30)
    at.run()
    # Al menos un text_input (tickers) en el sidebar
    assert len(at.sidebar.text_input) >= 1
    assert len(at.sidebar.date_input) >= 2


def test_all_pages_compile_and_have_setup():
    """Cada página debe llamar a page_setup() al inicio."""
    pages_dir = APP_DIR / "pages"
    if not pages_dir.is_dir():
        pytest.skip("No hay páginas")

    pages = [p for p in sorted(pages_dir.glob("*.py")) if not p.name.startswith("_")]
    if not pages:
        pytest.skip("No hay páginas para testear")

    for page in pages:
        source = page.read_text(encoding="utf-8")
        assert "page_setup" in source or "set_page_config" in source, (
            f"{page.name} no llama a page_setup() ni a st.set_page_config()"
        )


def test_history_page_renders_report_for_saved_run():
    """Regresión del informe de investigación (página Histórico): al
    seleccionar una entrada guardada, deben aparecer sus métricas
    formateadas (no solo el JSON en bruto).

    AppTest corre la página en un subproceso propio, así que no se le
    puede inyectar una ruta de BD temporal desde el test — usa la ruta
    real de `app.core.history.DB_PATH` (gitignorada) y la limpia con
    `clear_all()` en el `finally`, tanto si el test pasa como si falla.
    """
    from app.core import history

    history.init_db()
    run_id = history.save_run(
        strategy="Pairs Trading",
        tickers=["KO", "PEP"],
        start_date="2020-01-01",
        end_date="2024-12-31",
        params={"window": 60, "entry": 2.0, "exit_": 0.5},
        metrics={
            "annual_return": 0.142, "sharpe": 1.42, "sortino": 1.87,
            "max_drawdown": -0.124, "calmar": 1.15, "total_return": 0.68,
            "n_trades": 45, "win_rate": 0.58, "profit_factor": 1.6,
        },
        notes="Test de regresión",
    )
    try:
        at = AppTest.from_file(str(APP_DIR / "pages" / "9_📚_Histórico.py"),
                                default_timeout=30)
        at.run()
        assert not at.exception

        detail_selects = [
            w for w in at.selectbox if w.label and "Selecciona un ID" in w.label
        ]
        assert detail_selects, "No se encontró el selectbox de detalle"
        detail_selects[0].select(run_id).run()
        assert not at.exception

        values = [m.value for m in at.metric]
        assert "1.42" in values  # Sharpe
        assert "14.20%" in values  # Retorno anual
        assert "45" in values  # Nº operaciones
    finally:
        history.clear_all()


def test_optimizacion_page_loads_without_default_value_error():
    """Regresión: la página de Optimización tenía un
    st.multiselect(..., default=[20, 40, 60]) donde 40 no estaba en
    la lista de opciones [10, 20, 30, 45, 60, 90, 120] -- Streamlit
    lanza StreamlitDefaultNotInOptionsError inmediatamente al cargar
    la página (antes de cualquier interacción del usuario), así que
    un simple `at.run()` sin más ya lo detecta.
    """
    at = AppTest.from_file(
        str(APP_DIR / "pages" / "7_🎯_Optimización.py"), default_timeout=30,
    )
    at.run()
    assert not at.exception


def test_ticker_search_does_not_crash_page():
    """El buscador de empresas por nombre (portada) no debe romper la
    app aunque la búsqueda falle (p. ej. sin red, o Yahoo Finance caído)
    -- search_ticker envuelve los fallos como ConnectionError y la
    página debe mostrar un aviso en vez de una excepción sin capturar.
    """
    at = AppTest.from_file(str(APP_DIR / "main.py"), default_timeout=30)
    at.run()
    assert not at.exception

    search_box = at.sidebar.text_input(key="_ticker_search_query")
    search_box.set_value("Apple").run()
    assert not at.exception


def test_portfolio_page_loads_and_runs_without_exception():
    """Página de Portfolio (HRP/Markowitz/Risk Parity): debe cargar sin
    excepción, mostrar el aviso inicial, y gestionar con gracia el
    fallo de red al pulsar 'Calcular cartera' (no hay red real a los
    tickers de prueba en el entorno de test). Se prueban también los
    3 modos de rebalanceo (sin rebalanceo, calendario, por desviación).
    """
    at = AppTest.from_file(str(APP_DIR / "pages" / "14_💼_Portfolio.py"), default_timeout=30)
    at.session_state["global_tickers"] = "AAA, BBB, CCC"
    at.run()
    assert not at.exception

    buttons = [b for b in at.sidebar.button if "Calcular cartera" in (b.label or "")]
    assert len(buttons) == 1
    buttons[0].click().run()
    assert not at.exception

    rebal_select = [s for s in at.sidebar.selectbox if "Modo" in (s.label or "")][0]
    for mode in ["Calendario", "Por desviación"]:
        rebal_select.select(mode).run()
        assert not at.exception, f"Excepción con modo '{mode}': {at.exception}"
        buttons = [b for b in at.sidebar.button if "Calcular cartera" in (b.label or "")]
        buttons[0].click().run()
        assert not at.exception, f"Excepción al calcular con modo '{mode}': {at.exception}"


def test_estrategias_page_loads_and_switches_strategy():
    """Página de Estrategias: debe cargar sin excepción y permitir
    cambiar entre las 6 estrategias del registro sin romperse."""
    at = AppTest.from_file(str(APP_DIR / "pages" / "15_🧭_Estrategias.py"), default_timeout=30)
    at.session_state["global_tickers"] = "AAA, BBB, CCC"
    at.run()
    assert not at.exception

    select = at.sidebar.selectbox(key=None)
    for name in [
        "pca_statarb", "risk_parity", "cross_sectional_momentum",
        "carry_trade", "volatility_targeting", "trend_following",
    ]:
        select.select(name).run()
        assert not at.exception, f"Excepción al seleccionar '{name}': {at.exception}"


def test_papertrading_page_without_credentials():
    """Página de Papertrading sin credenciales de Alpaca configuradas:
    debe mostrar la guía de configuración, no una excepción."""
    at = AppTest.from_file(str(APP_DIR / "pages" / "16_📟_Papertrading.py"), default_timeout=30)
    at.run()
    assert not at.exception


def test_walkforward_page_loads_without_exception():
    """Página de WalkForward, con la nueva sección de informe HTML y
    alertas: debe cargar sin excepción."""
    at = AppTest.from_file(str(APP_DIR / "pages" / "6_🔬_WalkForward.py"), default_timeout=30)
    at.session_state["global_tickers"] = "AAA"
    at.run()
    assert not at.exception


def test_riesgo_page_loads_with_filtered_var():
    """Página de Riesgo, con el VaR filtrado por GARCH añadido: debe
    cargar sin excepción."""
    at = AppTest.from_file(str(APP_DIR / "pages" / "4_⚠️_Riesgo.py"), default_timeout=30)
    at.session_state["global_tickers"] = "AAA"
    at.run()
    assert not at.exception


def test_experimentos_page_empty_state():
    """Página de Experimentos sin ningún experimento guardado todavía:
    debe mostrar el aviso informativo, no una excepción."""
    at = AppTest.from_file(str(APP_DIR / "pages" / "17_🧪_Experimentos.py"), default_timeout=30)
    at.run()
    assert not at.exception


def test_licencia_page_loads_without_exception():
    """Página de Licencia sin ninguna clave guardada: debe cargar y
    mostrar la huella de máquina sin excepción."""
    at = AppTest.from_file(str(APP_DIR / "pages" / "18_🔑_Licencia.py"), default_timeout=30)
    at.run()
    assert not at.exception
