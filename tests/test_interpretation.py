"""Tests de app/core/interpretation/*: interpretación determinista
(sin IA, en lenguaje llano) de cada módulo cuantitativo. Cada función
es pura (recibe un resultado ya calculado, devuelve una lista de
strings), así que se construyen los dataclasses directamente en vez
de correr los modelos reales."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.core.backtest import BacktestResult
from app.core.cointegration import CointegrationResult
from app.core.fama_french import FamaFrenchResult
from app.core.garch import GarchResult
from app.core.interpretation import (
    interpret_backtest,
    interpret_cointegration,
    interpret_fama_french,
    interpret_garch,
    interpret_optimization,
    interpret_risk,
    interpret_robustness,
    interpret_walkforward,
)
from app.core.optimization import WalkForwardOptimizationResult
from app.core.robustness import MonteCarloResult, RobustnessReport, SensitivityResult
from app.core.walkforward import WalkForwardResult, WalkForwardWindow


# ============================================================
#  GARCH
# ============================================================
def _garch_result(converged=True, stationary=True) -> GarchResult:
    idx = pd.bdate_range("2024-01-01", periods=50)
    vol = pd.Series(np.linspace(1.0, 1.5, 50), index=idx)
    alpha_beta_sum = 0.8 if stationary else 1.2
    params = pd.Series({"omega": 0.01, "alpha[1]": alpha_beta_sum / 2, "beta[1]": alpha_beta_sum / 2})
    return GarchResult(
        model_result=None,  # type: ignore[arg-type]
        conditional_volatility=vol,
        standardized_residuals=pd.Series(np.zeros(50), index=idx),
        params=params,
        pvalues=pd.Series({"omega": 0.01, "alpha[1]": 0.02, "beta[1]": 0.01}),
        aic=100.0, bic=110.0, model_type="Garch",
        converged=converged, rescale_factor=1.0,
    )


def test_interpret_garch_covers_convergence_stationarity_and_forecast():
    result = _garch_result(converged=True, stationary=True)
    forecast = pd.Series([2.0], index=pd.bdate_range("2025-01-01", periods=1))
    diagnostics = {
        "ljung_box_pvalue": 0.5, "ljung_box_squared_pvalue": 0.5,
        "arch_lm_pvalue": 0.5, "jarque_bera_pvalue": 0.5,
    }
    bullets = interpret_garch(result, diagnostics, forecast)
    joined = " ".join(bullets)
    assert "terminó de calcularse correctamente" in joined
    assert "tiende a estabilizarse" in joined
    assert "subirá" in joined  # forecast (2.0) muy por encima de la vol actual (~1.5)


def test_interpret_garch_flags_non_convergence_and_non_stationarity():
    result = _garch_result(converged=False, stationary=False)
    forecast = pd.Series(dtype=float)
    bullets = interpret_garch(result, None, forecast)
    joined = " ".join(bullets)
    assert "no terminó de ajustarse bien" in joined
    assert "podría no estabilizarse" in joined


def test_interpret_garch_omits_fat_tails_bullet_for_the_common_case():
    """Regresión: las colas pesadas (jb < 0.05) son la norma en
    retornos diarios de bolsa -- casi cualquier empresa las tiene, así
    que no aporta nada distinguir un informe de otro. El caso
    informativo es el contrario (residuos que sí parecen normales)."""
    result = _garch_result()
    forecast = pd.Series([1.5], index=pd.bdate_range("2025-01-01", periods=1))
    diagnostics_common = {
        "ljung_box_pvalue": 0.5, "ljung_box_squared_pvalue": 0.5,
        "arch_lm_pvalue": 0.5, "jarque_bera_pvalue": 0.01,
    }
    bullets_common = interpret_garch(result, diagnostics_common, forecast)
    assert not any("movimientos más extremos" in b for b in bullets_common)

    diagnostics_rare = dict(diagnostics_common, jarque_bera_pvalue=0.5)
    bullets_rare = interpret_garch(result, diagnostics_rare, forecast)
    assert any("movimientos más extremos" in b for b in bullets_rare)


def test_interpret_garch_merges_healthy_fit_checks_into_one_bullet():
    """Regresión: converger y ser estacionario casi siempre van juntos
    en datos reales -- antes salían como dos frases casi calcadas en
    cada informe; ahora se fusionan en una sola cuando ambas van bien."""
    result = _garch_result(converged=True, stationary=True)
    forecast = pd.Series([1.5], index=pd.bdate_range("2025-01-01", periods=1))
    bullets = interpret_garch(result, None, forecast)
    fit_bullets = [b for b in bullets if "ajuste técnico es sólido" in b]
    assert len(fit_bullets) == 1


def test_interpret_garch_avoids_naming_the_underlying_statistical_tests():
    """El objetivo explícito de este módulo es no exponer jerga
    (nombres de tests estadísticos) a un lector no técnico."""
    result = _garch_result()
    forecast = pd.Series([1.5], index=pd.bdate_range("2025-01-01", periods=1))
    diagnostics = {
        "ljung_box_pvalue": 0.01, "ljung_box_squared_pvalue": 0.01,
        "arch_lm_pvalue": 0.01, "jarque_bera_pvalue": 0.01,
    }
    bullets = interpret_garch(result, diagnostics, forecast)
    joined = " ".join(bullets)
    for jargon in ("Ljung-Box", "ARCH-LM", "Jarque-Bera", "p-valor"):
        assert jargon not in joined


# ============================================================
#  Cointegración
# ============================================================
def _coint_result(is_cointegrated=True, adf_pvalue=0.01) -> CointegrationResult:
    idx = pd.bdate_range("2024-01-01", periods=30)
    return CointegrationResult(
        pvalue=0.01 if is_cointegrated else 0.5,
        alpha=1.0, beta=0.8,
        spread=pd.Series(np.zeros(30), index=idx),
        adf_spread_pvalue=adf_pvalue, adf_spread_stat=-3.5,
        adf_spread_crit={"5%": -2.9},
        adf_pvalue_1=0.3, adf_pvalue_2=0.4,
        is_cointegrated=is_cointegrated,
    )


def test_interpret_cointegration_cointegrated_fast_reversion_uses_ticker_names():
    bullets = interpret_cointegration(
        _coint_result(is_cointegrated=True), half_life_value=10.0,
        ticker_1="AAPL", ticker_2="MSFT",
    )
    joined = " ".join(bullets)
    assert "AAPL y MSFT se mueven juntos" in joined
    assert "favorable" in joined  # half-life corta


def test_interpret_cointegration_not_cointegrated_without_ticker_names():
    """Sin ticker_1/ticker_2, debe usar términos genéricos en vez de fallar."""
    bullets = interpret_cointegration(_coint_result(is_cointegrated=False), half_life_value=float("inf"))
    joined = " ".join(bullets)
    assert "no confirma" in joined
    assert "no se puede estimar" in joined.lower()


def test_interpret_cointegration_slow_reversion():
    bullets = interpret_cointegration(_coint_result(), half_life_value=90.0)
    assert any("lento" in b for b in bullets)


def test_interpret_cointegration_avoids_naming_the_underlying_statistical_tests():
    bullets = interpret_cointegration(_coint_result(), half_life_value=10.0)
    joined = " ".join(bullets)
    for jargon in ("Engle-Granger", "ADF", "p-valor"):
        assert jargon not in joined


# ============================================================
#  Fama-French
# ============================================================
def _ff_result(alpha_pvalue=0.01, all_betas_significant=False) -> FamaFrenchResult:
    betas = pd.Series({"Mkt-RF": 1.1, "SMB": 0.2, "HML": -0.1})
    betas_pvalues = (
        pd.Series({"Mkt-RF": 0.001, "SMB": 0.01, "HML": 0.02})
        if all_betas_significant
        else pd.Series({"Mkt-RF": 0.001, "SMB": 0.3, "HML": 0.4})
    )
    return FamaFrenchResult(
        alpha=0.0005, alpha_pvalue=alpha_pvalue, alpha_tstat=2.5,
        betas=betas,
        betas_pvalues=betas_pvalues,
        betas_tstats=pd.Series({"Mkt-RF": 5.0, "SMB": 1.0, "HML": 0.8}),
        r_squared=0.75, adj_r_squared=0.74, n_obs=500,
        cov_type="HAC", maxlags=5, summary="...",
    )


def test_interpret_fama_french_significant_alpha_and_all_factors():
    bullets = interpret_fama_french(_ff_result(alpha_pvalue=0.01, all_betas_significant=True))
    joined = " ".join(bullets)
    assert "no explicado" in joined
    assert "más que el mercado" in joined  # Mkt-RF beta 1.1 > 1
    assert "empresas pequeñas" in joined  # SMB beta positivo
    assert "crecimiento" in joined  # HML beta negativo
    assert "en gran parte" in joined  # r_squared 0.75


def test_interpret_fama_french_non_significant_alpha():
    bullets = interpret_fama_french(_ff_result(alpha_pvalue=0.5))
    assert any("se explica bien" in b for b in bullets)


def test_interpret_fama_french_no_significant_factors_says_so():
    result = _ff_result(alpha_pvalue=0.5)
    result.betas_pvalues = pd.Series({"Mkt-RF": 0.5, "SMB": 0.5, "HML": 0.5})
    bullets = interpret_fama_french(result)
    assert any("Ninguno de los factores" in b for b in bullets)


# ============================================================
#  Riesgo
# ============================================================
def test_interpret_risk_flags_non_normality_and_sharpe_bands():
    metrics = {
        "var_hist": 0.05, "var_param": 0.03, "var_cf": 0.06,
        "es_hist": 0.08, "es_param": 0.04, "var_fhs": 0.07,
        "sharpe": 2.5, "sortino": 4.0, "calmar": 1.2, "max_dd": -0.2,
    }
    bullets = interpret_risk(metrics, confidence=0.95)
    joined = " ".join(bullets)
    assert "resultados bastante distintos" in joined
    assert "muy buena" in joined


def test_interpret_risk_missing_keys_are_skipped_not_crashed():
    bullets = interpret_risk({"sharpe": -0.5}, confidence=0.95)
    assert any("ha perdido valor" in b for b in bullets)


def test_interpret_risk_avoids_naming_var_es_sharpe_as_such():
    metrics = {"var_hist": 0.05, "sharpe": 1.5}
    bullets = interpret_risk(metrics, confidence=0.95)
    joined = " ".join(bullets)
    for jargon in ("VaR", "Value at Risk", "Sharpe"):
        assert jargon not in joined


# ============================================================
#  Backtest
# ============================================================
def _bt_result(**metric_overrides) -> BacktestResult:
    metrics = {
        "n_trades": 50, "sharpe": 1.5, "win_rate": 0.5, "profit_factor": 1.3,
        "max_drawdown": -0.15, "total_return": 0.3, "exposure": 0.5,
    }
    metrics.update(metric_overrides)
    return BacktestResult(
        equity_curve=pd.Series(dtype=float), returns=pd.Series(dtype=float),
        positions=pd.Series(dtype=float), trades=pd.DataFrame(),
        metrics=metrics, params={},
    )


def test_interpret_backtest_few_trades_warns_unreliable():
    bullets = interpret_backtest(_bt_result(n_trades=3))
    assert any("puede deberse al azar" in b for b in bullets)


def test_interpret_backtest_suspiciously_high_sharpe():
    bullets = interpret_backtest(_bt_result(sharpe=5.0))
    assert any("demasiado bueno para ser cierto" in b for b in bullets)


def test_interpret_backtest_negative_sharpe():
    bullets = interpret_backtest(_bt_result(sharpe=-0.5))
    assert any("ha perdido dinero" in b for b in bullets)


# ============================================================
#  Walk-Forward
# ============================================================
def _wf_result(is_sharpe=2.0, oos_sharpe=1.8, embargo=0) -> WalkForwardResult:
    window = WalkForwardWindow(
        train_start=pd.Timestamp("2024-01-01"), train_end=pd.Timestamp("2024-06-01"),
        test_start=pd.Timestamp("2024-06-02"), test_end=pd.Timestamp("2024-09-01"),
        is_metrics={"sharpe": is_sharpe}, oos_metrics={"sharpe": oos_sharpe},
    )
    return WalkForwardResult(
        windows=[window],
        is_metrics_agg={"sharpe": is_sharpe, "total_return": 0.2, "max_drawdown": -0.1},
        oos_metrics_agg={"sharpe": oos_sharpe, "total_return": 0.15, "max_drawdown": -0.12},
        oos_equity_concat=pd.Series(dtype=float),
        params={"n_windows": 1, "n_windows_skipped": 0, "embargo": embargo},
    )


def test_interpret_walkforward_severe_degradation():
    bullets = interpret_walkforward(_wf_result(is_sharpe=2.0, oos_sharpe=0.5))
    assert any("empeora mucho" in b for b in bullets)


def test_interpret_walkforward_stable_result_and_no_margin_note():
    bullets = interpret_walkforward(_wf_result(is_sharpe=2.0, oos_sharpe=1.9, embargo=0))
    joined = " ".join(bullets)
    assert "razonablemente parecido" in joined
    assert "ningún margen" in joined


def test_interpret_walkforward_with_embargo_configured():
    bullets = interpret_walkforward(_wf_result(embargo=5))
    assert any("margen de **5 días**" in b for b in bullets)


# ============================================================
#  Optimización
# ============================================================
def _opt_result() -> WalkForwardOptimizationResult:
    grid = pd.DataFrame({"window": [10, 20], "oos_sharpe": [0.5, 1.2]})
    return WalkForwardOptimizationResult(
        grid=grid, best_params={"window": 20},
        best_oos_metrics={"sharpe": 1.2, "n_trades": 40},
        best_is_metrics={"sharpe": 1.5},
        objective="sharpe", param_names=["window"],
    )


def test_interpret_optimization_without_dsr():
    bullets = interpret_optimization(_opt_result(), dsr=None)
    assert any("configuraciones distintas" in b for b in bullets)
    assert all("Deflated" not in b for b in bullets)


def test_interpret_optimization_with_low_dsr_warns():
    dsr = {"dsr": 0.6, "n_trials": 2}
    bullets = interpret_optimization(_opt_result(), dsr=dsr)
    assert any("trátalo con cautela" in b for b in bullets)


def test_interpret_optimization_few_oos_trades_flagged():
    result = _opt_result()
    result.best_oos_metrics["n_trades"] = 5
    bullets = interpret_optimization(result, dsr=None)
    assert any("demasiado pocas para confiar" in b for b in bullets)


# ============================================================
#  Robustez
# ============================================================
def _mc_result(mean=1.0, positive_fraction=0.9) -> MonteCarloResult:
    n = 100
    n_pos = int(n * positive_fraction)
    dist = np.concatenate([np.full(n_pos, 1.0), np.full(n - n_pos, -1.0)])
    return MonteCarloResult(
        n_simulations=n, horizon=252, metric="sharpe", distribution=dist,
        mean=mean, std=0.5, percentiles={"p05": -1, "p50": 1, "p95": 2},
        var_95=1.0, cvar_95=1.2,
    )


def test_interpret_robustness_full_report():
    report = RobustnessReport(
        components={"degradacion_is_oos": 25.0, "monte_carlo_sharpe": 20.0},
        final_score=75.0, interpretation="Aceptable — vigilar la degradación OOS.",
    )
    sens = SensitivityResult(
        param_name="window", base_value=60,
        variations=pd.DataFrame({"value": [30, 60, 90], "sharpe": [1.0, 1.2, 1.1]}),
        stability_score=0.8,
    )
    bullets = interpret_robustness(report, _mc_result(positive_fraction=0.9), sens)
    joined = " ".join(bullets)
    assert "75 sobre 100" in joined
    assert "consistente" in joined
    assert "no depende demasiado" in joined


def test_interpret_robustness_low_probability_positive_and_low_stability():
    report = RobustnessReport(components={}, final_score=20.0, interpretation="Muy frágil.")
    sens = SensitivityResult(
        param_name="entry", base_value=2.0,
        variations=pd.DataFrame({"value": [1.5, 2.0, 2.5], "sharpe": [0.1, 1.0, -0.5]}),
        stability_score=0.1,
    )
    bullets = interpret_robustness(report, _mc_result(positive_fraction=0.3), sens)
    joined = " ".join(bullets)
    assert "podría perder" in joined
    assert "depende mucho" in joined
