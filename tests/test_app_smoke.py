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

    select = at.sidebar.selectbox(key="es_strategy_name")
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


def test_backtest_page_config_upload_applies_valid_values():
    """Regresión: subir un JSON de configuración válido en Backtest debe
    actualizar los widgets del sidebar (estrategia, ventana, capital)
    con los valores del archivo, sin excepción."""
    import json

    config = {
        "strategy": "Mean Reversion", "ticker_a": "AAA", "ticker_b": None,
        "window": 45, "entry": 1.8, "exit_": 0.4,
        "initial_capital": 75000, "commission": 0.0015, "slippage": 0.0008,
    }
    content = json.dumps(config).encode("utf-8")

    at = AppTest.from_file(str(APP_DIR / "pages" / "5_🧪_Backtest.py"), default_timeout=30)
    at.session_state["global_tickers"] = "AAA, BBB"
    at.run()
    assert not at.exception

    uploader = at.get("file_uploader")[0]
    uploader.upload("config.json", content, "application/json").run()
    assert not at.exception

    strategy_select = [s for s in at.sidebar.selectbox if s.label == "Estrategia"][0]
    assert strategy_select.value == "Mean Reversion"
    capital_input = [n for n in at.sidebar.number_input if "Capital" in (n.label or "")][0]
    assert capital_input.value == 75000


def test_backtest_page_config_upload_skips_stale_ticker_and_out_of_range():
    """Regresión: un JSON de configuración con un ticker que ya no está
    en la lista actual, o un valor fuera del rango del slider, no debe
    romper la app -- se ignora ese campo concreto (con aviso), en vez
    de provocar un StreamlitAPIException al intentar fijar un
    selectbox/slider a un valor inválido."""
    import json

    stale_config = {"strategy": "Momentum", "ticker_a": "ZZZ_NO_EXISTE", "window": 9999}
    content = json.dumps(stale_config).encode("utf-8")

    at = AppTest.from_file(str(APP_DIR / "pages" / "5_🧪_Backtest.py"), default_timeout=30)
    at.session_state["global_tickers"] = "AAA, BBB"
    at.run()

    uploader = at.get("file_uploader")[0]
    uploader.upload("config_vieja.json", content, "application/json").run()
    assert not at.exception


def test_licencia_page_loads_without_exception():
    """Página de Licencia sin ninguna clave guardada: debe cargar y
    mostrar la huella de máquina sin excepción."""
    at = AppTest.from_file(str(APP_DIR / "pages" / "18_🔑_Licencia.py"), default_timeout=30)
    at.run()
    assert not at.exception


def test_garch_page_config_upload_applies_valid_values():
    """Regresión: subir un JSON de configuración válido en GARCH debe
    actualizar ticker/p/q/vol/dist en el sidebar sin excepción."""
    import json

    config = {"ticker": "BBB", "p": 2, "q": 2, "vol": "EGARCH", "dist": "t"}
    content = json.dumps(config).encode("utf-8")

    at = AppTest.from_file(str(APP_DIR / "pages" / "1_📈_GARCH.py"), default_timeout=30)
    at.session_state["global_tickers"] = "AAA, BBB"
    at.run()
    assert not at.exception

    uploader = at.get("file_uploader")[0]
    uploader.upload("config.json", content, "application/json").run()
    assert not at.exception

    ticker_select = [s for s in at.sidebar.selectbox if s.label == "Ticker a modelar"][0]
    vol_select = [s for s in at.sidebar.selectbox if s.label == "Tipo de modelo"][0]
    assert ticker_select.value == "BBB"
    assert vol_select.value == "EGARCH"


def test_garch_page_config_upload_skips_invalid_option():
    """Regresión: un JSON con una opción que no existe en el selectbox
    (p.ej. un tipo de modelo GARCH inválido) no debe romper la app."""
    import json

    bad_config = {"vol": "NO_EXISTE", "p": 99}
    content = json.dumps(bad_config).encode("utf-8")

    at = AppTest.from_file(str(APP_DIR / "pages" / "1_📈_GARCH.py"), default_timeout=30)
    at.session_state["global_tickers"] = "AAA, BBB"
    at.run()

    uploader = at.get("file_uploader")[0]
    uploader.upload("mala.json", content, "application/json").run()
    assert not at.exception


@pytest.mark.parametrize(
    "page_path,valid_config,label,expected,stale_config",
    [
        (
            "2_🔗_Cointegración.py",
            {"t1": "BBB", "t2": "AAA", "window": 80, "entry": 2.5},
            "Ticker 1", "BBB",
            {"t1": "ZZZ_NO_EXISTE", "window": 99999},
        ),
        (
            "4_⚠️_Riesgo.py",
            {"ticker": "BBB", "confidence": 0.97, "window": 300},
            "Ticker", "BBB",
            {"ticker": "ZZZ_NO_EXISTE", "confidence": 5.0},
        ),
        (
            "11_📉_Regímenes.py",
            {"ticker": "BBB", "n_states": 3, "cov_type": "full", "n_iter": 500},
            "Ticker", "BBB",
            {"ticker": "ZZZ_NO_EXISTE", "cov_type": "NO_EXISTE"},
        ),
        (
            "12_🎛️_Kalman.py",
            {"t1": "BBB", "t2": "AAA", "delta": 0.001, "r_var": 0.005, "roll_window": 100},
            "Ticker Y (dependiente)", "BBB",
            {"t1": "ZZZ_NO_EXISTE", "delta": 99},
        ),
    ],
)
def test_config_upload_roundtrip(page_path, valid_config, label, expected, stale_config):
    """Regresión parametrizada: en cada una de estas 4 páginas, subir un
    JSON de configuración válido actualiza el widget correspondiente, y
    subir uno con un ticker obsoleto / valor fuera de rango no rompe
    la app (se ignora ese campo con aviso, en vez de un
    StreamlitAPIException al fijar un selectbox/slider a un valor
    inválido)."""
    import json

    full_path = str(APP_DIR / "pages" / page_path)

    at = AppTest.from_file(full_path, default_timeout=30)
    at.session_state["global_tickers"] = "AAA, BBB"
    at.run()
    assert not at.exception

    uploader = at.get("file_uploader")[0]
    uploader.upload("config.json", json.dumps(valid_config).encode("utf-8"), "application/json").run()
    assert not at.exception
    sel = [s for s in at.sidebar.selectbox if s.label == label][0]
    assert sel.value == expected

    at2 = AppTest.from_file(full_path, default_timeout=30)
    at2.session_state["global_tickers"] = "AAA, BBB"
    at2.run()
    uploader2 = at2.get("file_uploader")[0]
    uploader2.upload("mala.json", json.dumps(stale_config).encode("utf-8"), "application/json").run()
    assert not at2.exception


@pytest.mark.parametrize(
    "page_path,valid_config,label,expected,stale_config",
    [
        (
            "6_🔬_WalkForward.py",
            {"strategy": "Momentum", "t1": "BBB", "train_size": 700, "test_size": 150},
            "Estrategia", "Momentum",
            {"strategy": "NO_EXISTE", "t1": "ZZZ_NO_EXISTE", "train_size": 99999},
        ),
        (
            "7_🎯_Optimización.py",
            {"strategy": "Mean Reversion", "t1": "BBB", "objective": "sortino"},
            "Estrategia", "Mean Reversion",
            {"strategy": "NO_EXISTE", "objective": "NO_EXISTE", "windows": [99999]},
        ),
        (
            "8_🛡️_Robustez.py",
            {"strategy": "Mean Reversion", "t1": "BBB", "n_sims": 2000},
            "Estrategia", "Mean Reversion",
            {"t1": "ZZZ_NO_EXISTE", "n_sims": 99999},
        ),
        (
            "14_💼_Portfolio.py",
            {"method": "Risk Parity (ERC)", "initial_capital": 50000},
            "Método", "Risk Parity (ERC)",
            {"method": "NO_EXISTE", "initial_capital": -5},
        ),
    ],
)
def test_config_upload_roundtrip_extended(page_path, valid_config, label, expected, stale_config):
    """Misma regresión que test_config_upload_roundtrip, para las 4
    páginas con parámetros condicionales según la estrategia/método
    elegido (WalkForward, Optimización, Robustez, Portfolio)."""
    import json

    full_path = str(APP_DIR / "pages" / page_path)

    at = AppTest.from_file(full_path, default_timeout=30)
    at.session_state["global_tickers"] = "AAA, BBB, CCC"
    at.run()
    assert not at.exception

    uploader = at.get("file_uploader")[0]
    uploader.upload("config.json", json.dumps(valid_config).encode("utf-8"), "application/json").run()
    assert not at.exception
    sel = [s for s in at.sidebar.selectbox if s.label == label][0]
    assert sel.value == expected

    at2 = AppTest.from_file(full_path, default_timeout=30)
    at2.session_state["global_tickers"] = "AAA, BBB, CCC"
    at2.run()
    uploader2 = at2.get("file_uploader")[0]
    uploader2.upload("mala.json", json.dumps(stale_config).encode("utf-8"), "application/json").run()
    assert not at2.exception


def test_famafrench_page_config_upload_roundtrip():
    """Regresión: Fama-French quedó fuera por accidente de la primera
    ronda de guardar/cargar configuración -- se añadió después al
    revisar el conteo final de páginas cubiertas."""
    import json

    at = AppTest.from_file(str(APP_DIR / "pages" / "3_📊_Fama_French.py"), default_timeout=30)
    at.session_state["global_tickers"] = "AAA, BBB"
    at.run()
    assert not at.exception

    valid_config = {"ticker": "BBB", "model": "5", "cov_type": "HC3", "maxlags": 10}
    uploader = at.get("file_uploader")[0]
    uploader.upload("config.json", json.dumps(valid_config).encode("utf-8"), "application/json").run()
    assert not at.exception
    ticker_select = [s for s in at.sidebar.selectbox if s.label == "Ticker"][0]
    model_select = [s for s in at.sidebar.selectbox if s.label == "Modelo"][0]
    assert ticker_select.value == "BBB"
    assert model_select.value == "5"

    at2 = AppTest.from_file(str(APP_DIR / "pages" / "3_📊_Fama_French.py"), default_timeout=30)
    at2.session_state["global_tickers"] = "AAA, BBB"
    at2.run()
    stale_config = {"ticker": "ZZZ_NO_EXISTE", "model": "NO_EXISTE", "maxlags": 9999}
    uploader2 = at2.get("file_uploader")[0]
    uploader2.upload("mala.json", json.dumps(stale_config).encode("utf-8"), "application/json").run()
    assert not at2.exception


@pytest.mark.parametrize(
    "page_path,select_label,value_before,value_after",
    [
        ("5_🧪_Backtest.py", "Estrategia", "Momentum", "Mean Reversion"),
        ("1_📈_GARCH.py", "Tipo de modelo", "EGARCH", "Garch"),
    ],
)
def test_named_config_manager_save_and_load_roundtrip(page_path, select_label, value_before, value_after):
    """Regresión: guardar una configuración con nombre en la sesión
    (named_config_manager) y volver a cargarla debe restaurar el
    valor guardado -- sin depender de subir/descargar ningún archivo.
    """
    full_path = str(APP_DIR / "pages" / page_path)
    at = AppTest.from_file(full_path, default_timeout=30)
    at.session_state["global_tickers"] = "AAA, BBB"
    at.run()
    assert not at.exception

    sel = [s for s in at.sidebar.selectbox if s.label == select_label][0]
    sel.select(value_before).run()
    assert not at.exception

    name_input = [t for t in at.sidebar.text_input if "Nombre para guardar" in (t.label or "")][0]
    name_input.set_value("config_de_test").run()
    save_btn = [b for b in at.sidebar.button if "Guardar con este nombre" in (b.label or "")][0]
    save_btn.click().run()
    assert not at.exception

    sel2 = [s for s in at.sidebar.selectbox if s.label == select_label][0]
    sel2.select(value_after).run()
    assert not at.exception

    load_btn = [b for b in at.sidebar.button if b.label == "📂 Cargar"][0]
    load_btn.click().run()
    assert not at.exception

    sel3 = [s for s in at.sidebar.selectbox if s.label == select_label][0]
    assert sel3.value == value_before

    delete_btn = [b for b in at.sidebar.button if b.label == "🗑️ Borrar"][0]
    delete_btn.click().run()
    assert not at.exception
    """Regresión: Estrategias es la más compleja (6 estrategias, cada
    una con su propio conjunto de parámetros, algunos con el mismo
    nombre pero rangos distintos entre estrategias) -- confirma que
    cargar una configuración válida selecciona la estrategia correcta,
    y que una con la estrategia inexistente o un ticker obsoleto no
    rompe la página."""
    import json

    at = AppTest.from_file(str(APP_DIR / "pages" / "15_🧭_Estrategias.py"), default_timeout=30)
    at.session_state["global_tickers"] = "AAA, BBB, CCC"
    at.run()
    assert not at.exception

    valid_config = {
        "strategy_name": "volatility_targeting",
        "params": {"target_vol": 0.25},
        "single_ticker": "BBB",
    }
    uploader = at.get("file_uploader")[0]
    uploader.upload("config.json", json.dumps(valid_config).encode("utf-8"), "application/json").run()
    assert not at.exception
    strategy_select = [s for s in at.sidebar.selectbox if s.key == "es_strategy_name"][0]
    assert strategy_select.value == "volatility_targeting"

    at2 = AppTest.from_file(str(APP_DIR / "pages" / "15_🧭_Estrategias.py"), default_timeout=30)
    at2.session_state["global_tickers"] = "AAA, BBB, CCC"
    at2.run()
    stale_config = {
        "strategy_name": "NO_EXISTE",
        "params": {"target_vol": 99999},
        "single_ticker": "ZZZ_NO_EXISTE",
    }
    uploader2 = at2.get("file_uploader")[0]
    uploader2.upload("mala.json", json.dumps(stale_config).encode("utf-8"), "application/json").run()
    assert not at2.exception


@pytest.mark.parametrize(
    "page_path",
    ["2_🔗_Cointegración.py", "1_📈_GARCH.py", "5_🧪_Backtest.py"],
)
def test_plain_language_conclusion_pages_load_without_exception(page_path):
    """Regresión: las 3 páginas con conclusión en lenguaje llano
    (plain_language_summary + app.styles.conclusion()) deben seguir
    cargando sin excepción. No ejercita la rama que llama a
    conclusion() de verdad (necesita red real para cargar precios),
    pero confirma que el cableado de imports no rompe la página --
    la lógica de cada plain_language_summary() ya está cubierta con
    tests deterministas en test_cointegration.py, test_garch.py y
    test_backtest.py."""
    at = AppTest.from_file(str(APP_DIR / "pages" / page_path), default_timeout=30)
    at.session_state["global_tickers"] = "AAA, BBB"
    at.run()
    assert not at.exception
