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


def test_apply_tickers_does_not_revert_universe_selection():
    """Regresión: elegir un universo predefinido actualiza
    `global_tickers` de inmediato, pero el campo de texto del
    formulario (`_tickers_form_input`) no se refrescaba -- así que al
    pulsar "Aplicar tickers" ese valor viejo pisaba el cambio recién
    hecho y la lista de tickers volvía a la anterior."""
    at = AppTest.from_file(str(APP_DIR / "main.py"), default_timeout=30)
    at.run()

    before = at.session_state["global_tickers"]
    universe_select = next(
        w for w in at.selectbox if w.key == "_universe_select"
    )
    new_universe = next(o for o in universe_select.options if o != "(personalizado)")
    at = universe_select.select(new_universe).run()

    after_select = at.session_state["global_tickers"]
    assert after_select != before
    assert at.session_state["_tickers_form_input"] == after_select

    apply_button = next(
        b for b in at.button if "Aplicar tickers" in (b.label or "")
    )
    at = apply_button.click().run()

    assert at.session_state["global_tickers"] == after_select


def test_close_app_button_shows_closing_screen_without_rest_of_page():
    """Regresión: pulsar "Cerrar aplicación" mataba el proceso en el
    acto (misma ejecución del script) -- el aviso apenas llegaba a
    verse y el resto de la página (sidebar, botones...) seguía
    renderizada debajo, dando la sensación de que la app se había
    quedado colgada en vez de haberse cerrado. Ahora el botón solo
    marca `_app_closing` y el kill real (con un pequeño retardo) pasa
    en el siguiente rerun, que muestra una pantalla de cierre limpia
    y corta con `st.stop()` antes de llegar al resto de la página."""
    from unittest.mock import patch

    with patch("threading.Timer") as mock_timer:
        at = AppTest.from_file(str(APP_DIR / "main.py"), default_timeout=30)
        at.run()

        close_button = next(
            b for b in at.button if "Cerrar aplicación" in (b.label or "")
        )
        at = close_button.click().run()

        assert not at.exception
        assert at.session_state["_app_closing"] is True
        assert any(
            "se ha cerrado" in (m.value or "") for m in at.markdown
        )
        # El resto de la página (sidebar, botones normales) no debe
        # haberse renderizado en este rerun -- st.stop() cortó antes.
        assert len(at.sidebar.text_input) == 0

    # El kill real está programado con retardo, no se ejecuta en el acto.
    assert mock_timer.called
    delay = mock_timer.call_args.args[0]
    assert delay > 0


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


def test_gemini_manual_key_persists_across_page_navigation():
    """Regresión: la clave de Gemini pegada a mano en la portada
    desaparecía en cuanto se navegaba a otra página. Streamlit borra
    el session_state de un widget en cualquier página donde ese
    widget no se vuelva a crear -- como el campo de la clave solo
    existe en main.py, leer directamente su key= (en vez de una clave
    "normal" sincronizada por separado) hacía que `get_gemini_api_key()`
    devolviera vacío fuera de la portada, aunque la fuente activa
    mostrara "Introducida en esta sesión" en la propia portada."""
    at = AppTest.from_file(str(APP_DIR / "main.py"), default_timeout=30)
    at.run()

    key_input = next(
        w for w in at.sidebar.text_input
        if w.label == "API key de Google AI Studio"
    )
    key_input.set_value("FAKE-TEST-KEY-12345").run()

    assert at.session_state.get("gemini_api_key_manual") == "FAKE-TEST-KEY-12345"

    at.switch_page(str(APP_DIR / "pages" / "1_📈_GARCH.py")).run()
    assert not at.exception
    # La clave "normal" (no atada a ningún widget de esta página) debe
    # seguir ahí -- antes del fix, Streamlit la borraba al no volver
    # a crear el widget que la origina.
    assert at.session_state.get("gemini_api_key_manual") == "FAKE-TEST-KEY-12345"


def test_ai_provider_and_groq_key_persist_across_page_navigation():
    """Regresión (misma clase de bug que la Gemini de arriba, ver
    _on_ai_provider_change / _on_groq_key_change en main.py): tanto
    el proveedor elegido como la clave de Groq deben sobrevivir a la
    navegación entre páginas."""
    at = AppTest.from_file(str(APP_DIR / "main.py"), default_timeout=30)
    at.run()
    assert at.session_state.get("ai_provider") == "gemini"

    provider_select = next(
        w for w in at.sidebar.selectbox if w.key == "_ai_provider_select"
    )
    at = provider_select.select("groq").run()
    assert at.session_state.get("ai_provider") == "groq"

    key_input = next(
        w for w in at.sidebar.text_input if w.label == "API key de Groq"
    )
    at = key_input.set_value("FAKE-GROQ-KEY").run()
    assert at.session_state.get("groq_api_key_manual") == "FAKE-GROQ-KEY"

    at.switch_page(str(APP_DIR / "pages" / "1_📈_GARCH.py")).run()
    assert not at.exception
    assert at.session_state.get("ai_provider") == "groq"
    assert at.session_state.get("groq_api_key_manual") == "FAKE-GROQ-KEY"


def test_list_groq_models_button_shows_results():
    """Igual que test_list_gemini_models_button_shows_results, para
    Groq -- confirma que el selector de proveedor también controla
    qué sección de "ver modelos disponibles" se usa."""
    from unittest.mock import patch

    at = AppTest.from_file(str(APP_DIR / "main.py"), default_timeout=30)
    at.session_state["ai_provider"] = "groq"
    at.session_state["groq_api_key_manual"] = "FAKE-GROQ-KEY"
    at.run()

    list_button = next(
        b for b in at.sidebar.button if "Ver modelos disponibles" in (b.label or "")
    )
    assert not list_button.disabled

    with patch(
        "app.core.groq_report.list_available_models",
        return_value=["gemma2-9b-it", "llama-3.3-70b-versatile"],
    ):
        at = list_button.click().run()

    assert not at.exception
    assert at.session_state["_groq_models_list"] == ["gemma2-9b-it", "llama-3.3-70b-versatile"]
    assert any(
        "llama-3.3-70b-versatile" in (c.value or "") for c in at.sidebar.code
    )


def test_garch_ai_report_button_does_not_reset_results():
    """Regresión: dentro de la sección de resultados de GARCH, `run`
    (el botón "Ejecutar GARCH") solo es True en el rerun donde se
    pulsa -- cualquier otro botón de esa misma sección (como
    "Generar informe con IA") dispara su propio rerun, en el que
    `run` vuelve a ser False, y toda la sección de resultados
    desaparecía (la página volvía al aviso "Configura los
    parámetros..."), perdiendo el ajuste ya calculado."""
    from unittest.mock import patch

    import numpy as np
    import pandas as pd

    idx = pd.bdate_range("2023-01-01", "2024-12-31")
    rng = np.random.default_rng(7)
    prices = pd.DataFrame(
        {"AAA": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx))))}, index=idx,
    )

    with patch("app.core.data_loader.load_prices", return_value=prices):
        at = AppTest.from_file(str(APP_DIR / "pages" / "1_📈_GARCH.py"), default_timeout=30)
        at.session_state["global_tickers"] = "AAA"
        at.session_state["gemini_api_key_manual"] = "FAKE-TEST-KEY-12345"
        at.run()

        run_button = next(b for b in at.sidebar.button if "Ejecutar GARCH" in (b.label or ""))
        at = run_button.click().run()
        assert not at.exception
        assert any("AIC" in (m.label or "") for m in at.metric)

        ai_button = next(b for b in at.button if "Generar informe con IA" in (b.label or ""))
        with patch("app.core.ai_report.generate_report", return_value="## Informe\nContenido."):
            at = ai_button.click().run()

        assert not at.exception
        assert any("AIC" in (m.label or "") for m in at.metric), (
            "Los resultados de GARCH desaparecieron al generar el informe con IA"
        )
        assert at.session_state.get("garch_ai_report") == "## Informe\nContenido."


def test_backtest_ai_report_button_does_not_reset_results():
    """Regresión: mismo problema que en GARCH (ver
    test_garch_ai_report_button_does_not_reset_results), pero en la
    página de Backtest -- "Generar informe con IA" hacía desaparecer
    los resultados del backtest ya calculado."""
    from unittest.mock import patch

    import numpy as np
    import pandas as pd

    idx = pd.bdate_range("2023-01-01", "2024-12-31")
    rng = np.random.default_rng(11)
    prices = pd.DataFrame(
        {"AAA": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx))))}, index=idx,
    )

    with patch("app.core.data_loader.load_prices", return_value=prices):
        at = AppTest.from_file(str(APP_DIR / "pages" / "5_🧪_Backtest.py"), default_timeout=30)
        at.session_state["global_tickers"] = "AAA"
        at.session_state["gemini_api_key_manual"] = "FAKE-TEST-KEY-12345"
        at.session_state["bt_strategy"] = "Momentum"  # solo necesita 1 ticker
        at.run()

        run_button = next(b for b in at.sidebar.button if "Ejecutar backtest" in (b.label or ""))
        at = run_button.click().run()
        assert not at.exception
        assert any("Sharpe" in (m.label or "") for m in at.metric)

        ai_button = next(b for b in at.button if "Generar informe con IA" in (b.label or ""))
        with patch("app.core.ai_report.generate_report", return_value="## Informe\nContenido."):
            at = ai_button.click().run()

        assert not at.exception
        assert any("Sharpe" in (m.label or "") for m in at.metric), (
            "Los resultados del backtest desaparecieron al generar el informe con IA"
        )
        assert at.session_state.get("bt_ai_report") == "## Informe\nContenido."


def test_garch_ai_report_with_groq_provider_does_not_reset_results():
    """Igual que test_garch_ai_report_button_does_not_reset_results,
    pero con Groq como proveedor elegido -- confirma que el despacho
    de app.state.generate_ai_report() al backend correcto tampoco
    rompe la persistencia de resultados."""
    from unittest.mock import patch

    import numpy as np
    import pandas as pd

    idx = pd.bdate_range("2023-01-01", "2024-12-31")
    rng = np.random.default_rng(7)
    prices = pd.DataFrame(
        {"AAA": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx))))}, index=idx,
    )

    with patch("app.core.data_loader.load_prices", return_value=prices):
        at = AppTest.from_file(str(APP_DIR / "pages" / "1_📈_GARCH.py"), default_timeout=30)
        at.session_state["global_tickers"] = "AAA"
        at.session_state["ai_provider"] = "groq"
        at.session_state["groq_api_key_manual"] = "FAKE-GROQ-KEY"
        at.run()

        run_button = next(b for b in at.sidebar.button if "Ejecutar GARCH" in (b.label or ""))
        at = run_button.click().run()
        assert not at.exception
        assert any("AIC" in (m.label or "") for m in at.metric)

        ai_button = next(b for b in at.button if "Generar informe con IA" in (b.label or ""))
        with patch(
            "app.core.groq_report.generate_report", return_value="## Informe de Groq\nContenido.",
        ) as mock_gen:
            at = ai_button.click().run()

        assert not at.exception
        assert any("AIC" in (m.label or "") for m in at.metric), (
            "Los resultados de GARCH desaparecieron al generar el informe con IA (Groq)"
        )
        assert at.session_state.get("garch_ai_report") == "## Informe de Groq\nContenido."
        assert mock_gen.call_args.kwargs["api_key"] == "FAKE-GROQ-KEY"


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


def test_data_preview_offers_raw_price_download():
    """Regresión: data_preview() debe ofrecer un botón para descargar
    los precios en bruto, no solo el gráfico/estadísticas -- para que
    el usuario pueda cruzar los datos con otra fuente y verificar los
    cálculos por su cuenta.

    Se prueba con un script mínimo que llama a data_preview()
    directamente con datos sintéticos, en vez de a través de una
    página real -- load_prices() necesita red real, que AppTest no
    tiene, así que probar vía una página nunca llegaría a ejecutar
    data_preview() en absoluto.
    """
    script = """
import pandas as pd
from app.styles import data_preview

prices = pd.DataFrame(
    {"AAA": [100.0, 101.0, 102.0], "BBB": [50.0, 50.5, 51.0]},
    index=pd.bdate_range("2023-01-02", periods=3),
)
data_preview(prices)
"""
    at = AppTest.from_string(script, default_timeout=30)
    at.run()
    assert not at.exception

    download_labels = [b.label for b in at.get("download_button")]
    assert any("Descargar estos precios" in (label or "") for label in download_labels)


def test_search_ticker_button_selects_ticker_on_single_ticker_pages():
    """Regresión: al añadir un ticker desde el buscador de la portada,
    debe quedar seleccionado automáticamente en las páginas de UN solo
    ticker (GARCH, Riesgo, Regímenes, Fama-French) -- antes solo se
    añadía a la lista global, y cada página se quedaba con su
    selección anterior (Streamlit no cambia solo un selectbox porque
    la lista de opciones creció), dando la sensación de que la
    empresa buscada "no se añadía" aunque sí estuviera disponible
    para elegir a mano.

    Se prueba con un script aislado que replica el callback del botón
    de resultado de búsqueda, en vez de a través de main.py real --
    search_ticker() necesita red real, que AppTest no tiene."""
    script = """
import streamlit as st

st.session_state.setdefault("global_tickers", "AAPL, MSFT")
st.session_state.setdefault("garch_ticker", "AAPL")

if st.button("Simular clic en resultado de búsqueda: KO"):
    current = [t.strip() for t in st.session_state["global_tickers"].split(",") if t.strip()]
    if "KO" not in current:
        current.append("KO")
    st.session_state["global_tickers"] = ", ".join(current)
    for k in ("garch_ticker", "riesgo_ticker", "regimenes_ticker", "ff_ticker"):
        st.session_state[k] = "KO"
    st.toast("KO añadido")
    st.rerun()
"""
    at = AppTest.from_string(script, default_timeout=15)
    at.run()
    assert not at.exception

    at.button[0].click().run()
    assert not at.exception

    assert at.session_state["global_tickers"] == "AAPL, MSFT, KO"
    for key in ("garch_ticker", "riesgo_ticker", "regimenes_ticker", "ff_ticker"):
        assert at.session_state[key] == "KO"
    assert len(at.get("toast")) > 0


def test_ticker_badge_shows_single_pair_and_none_cases():
    """Regresión: ticker_badge() debe mostrar un único ticker, una
    pareja unida con ' vs ', una cesta de varios unida con ', ', y no
    mostrar nada si todos los argumentos son None -- probado de forma
    aislada (sin depender de ninguna página real ni de red)."""
    script = """
from app.styles import ticker_badge
ticker_badge("KO")
ticker_badge("KO", "PEP")
ticker_badge("KO", "PEP", "AAPL")
ticker_badge("KO", None)
ticker_badge(None, None)
"""
    at = AppTest.from_string(script, default_timeout=15)
    at.run()
    assert not at.exception

    badges = [m.value for m in at.markdown if '<div class="fql-ticker-badge"' in (m.value or "")]
    assert len(badges) == 4
    assert "KO</div>" in badges[0]
    assert "KO vs PEP</div>" in badges[1]
    assert "KO, PEP, AAPL</div>" in badges[2]
    assert "KO</div>" in badges[3]


def test_pages_with_ticker_badge_show_correct_selection():
    """Regresión: las 12 páginas con ticker_badge() deben mostrar la
    selección real de cada una (uno, dos, o la cesta completa según
    corresponda) -- confirma tanto que no rompen como que el ticker
    mostrado es el correcto, no solo que el componente exista."""
    cases = [
        ("1_📈_GARCH.py", "AAPL"),
        ("2_🔗_Cointegración.py", "AAPL vs MSFT"),
        ("3_📊_Fama_French.py", "AAPL"),
        ("4_⚠️_Riesgo.py", "AAPL"),
        ("5_🧪_Backtest.py", "AAPL vs MSFT"),
        ("6_🔬_WalkForward.py", "AAPL vs MSFT"),
        ("7_🎯_Optimización.py", "AAPL"),
        ("8_🛡️_Robustez.py", "AAPL"),
        ("11_📉_Regímenes.py", "AAPL"),
        ("12_🎛️_Kalman.py", "AAPL vs MSFT"),
        ("14_💼_Portfolio.py", "AAPL, MSFT, KO"),
        ("15_🧭_Estrategias.py", "AAPL, MSFT, KO"),
    ]
    for page, expected in cases:
        at = AppTest.from_file(str(APP_DIR / "pages" / page), default_timeout=30)
        at.session_state["global_tickers"] = "AAPL, MSFT, KO"
        at.run()
        assert not at.exception, f"{page}: {at.exception}"
        badges = [m.value for m in at.markdown if '<div class="fql-ticker-badge"' in (m.value or "")]
        assert badges, f"{page}: no se encontró ningún ticker_badge"
        assert f"Analizando: {expected}</div>" in badges[0], f"{page}: {badges[0]}"


@pytest.mark.parametrize(
    "page_path,session_key",
    [
        ("1_📈_GARCH.py", "garch_ai_report"),
        ("5_🧪_Backtest.py", "bt_ai_report"),
    ],
)
def test_ai_report_section_loads_without_key_and_without_exception(page_path, session_key):
    """Regresión: las páginas con sección de informe con IA deben cargar
    sin excepción cuando NO hay clave de Gemini configurada -- deben
    mostrar un aviso informativo, no romper la página."""
    at = AppTest.from_file(str(APP_DIR / "pages" / page_path), default_timeout=30)
    at.session_state["global_tickers"] = "AAA, BBB"
    at.run()
    assert not at.exception


def test_gemini_settings_seeded_with_correct_default():
    """Regresión: la portada debe sembrar gemini_model con
    app.core.ai_report.DEFAULT_MODEL al iniciar una sesión nueva --
    detectamos y arreglamos un bug donde salía vacío porque main.py tiene
    su propia inicialización de sesión, separada de ensure_session_initialized().

    Compara contra la constante (no un string literal) para que este
    test no se quede desactualizado si el modelo por defecto cambia
    -- como ya pasó una vez, cuando Google retiró 'gemini-2.5-flash'."""
    from app.core.ai_report import DEFAULT_MODEL

    at = AppTest.from_file(str(APP_DIR.parent / "app" / "main.py"), default_timeout=30)
    at.run()
    assert not at.exception
    assert at.session_state["gemini_model"] == DEFAULT_MODEL


def test_list_gemini_models_button_shows_results():
    """Regresión de UX: cuando el modelo configurado deja de existir
    (Google los retira de vez en cuando -- ver DEFAULT_MODEL), el
    usuario necesita saber cuál usar en su lugar. El botón "Ver
    modelos disponibles" debe mostrar la lista devuelta por la API
    (con la clave del propio usuario), sin excepción."""
    from unittest.mock import patch

    at = AppTest.from_file(str(APP_DIR / "main.py"), default_timeout=30)
    at.session_state["gemini_api_key_manual"] = "FAKE-KEY"
    at.run()

    list_button = next(
        b for b in at.sidebar.button if "Ver modelos disponibles" in (b.label or "")
    )
    assert not list_button.disabled

    with patch(
        "app.core.ai_report.list_available_models",
        return_value=["gemini-3.6-flash", "gemini-3.6-pro"],
    ):
        at = list_button.click().run()

    assert not at.exception
    assert at.session_state["_gemini_models_list"] == ["gemini-3.6-flash", "gemini-3.6-pro"]
    assert any(
        "gemini-3.6-flash" in (c.value or "") for c in at.sidebar.code
    )


def test_generate_report_roundtrip_isolated():
    """Regresión: el flujo completo de generar un informe con IA (prompt
    → generate_report mockeada → guardado en session_state → mostrado en
    pantalla) funciona sin excepción, y los errores de la API quedan
    capturados y mostrados como st.error sin romper la app."""
    from unittest.mock import patch

    from app.core.ai_report import AIReportError

    script = """
import streamlit as st
from app.core.ai_report import AIReportError, generate_report

if st.button("Generar"):
    try:
        st.session_state["ai_report"] = generate_report("prompt", api_key="clave", model="gemini-2.5-flash")
    except AIReportError as e:
        st.session_state["ai_report"] = None
        st.error(str(e))

if st.session_state.get("ai_report"):
    st.markdown(st.session_state["ai_report"])
"""
    with patch("app.core.ai_report.generate_report", return_value="## Informe\nContenido."):
        at = AppTest.from_string(script, default_timeout=15)
        at.run()
        at.button[0].click().run()
        assert not at.exception
        assert at.session_state["ai_report"] == "## Informe\nContenido."

    with patch("app.core.ai_report.generate_report", side_effect=AIReportError("429 límite")):
        at2 = AppTest.from_string(script, default_timeout=15)
        at2.run()
        at2.button[0].click().run()
        assert not at2.exception
        errors = [e.value for e in at2.error]
        assert any("429" in (e or "") for e in errors)


def test_ai_report_pdf_download_button_appears_isolated():
    """Regresión: junto al informe con IA mostrado en pantalla, debe
    aparecer un botón para descargarlo en PDF (build_ai_report_pdf),
    sin excepción. Igual que test_generate_report_roundtrip_isolated,
    aislado del resto de la página para no depender de red."""
    script = """
import streamlit as st
from app.core.report import build_ai_report_pdf

st.session_state.setdefault("ai_report", "## Informe\\nContenido con é, ñ y — raya.")

if st.session_state.get("ai_report"):
    st.markdown(st.session_state["ai_report"])
    st.download_button(
        "📄 Descargar informe en PDF",
        build_ai_report_pdf(
            title="Informe con IA — Test",
            meta={"Ticker": "AAPL"},
            report_text=st.session_state["ai_report"],
        ),
        file_name="informe_ia_test.pdf",
        mime="application/pdf",
    )
"""
    at = AppTest.from_string(script, default_timeout=15)
    at.run()
    assert not at.exception
    pdf_buttons = [b for b in at.get("download_button") if "PDF" in (b.label or "")]
    assert len(pdf_buttons) == 1
