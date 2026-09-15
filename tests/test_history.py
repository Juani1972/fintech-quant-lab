"""Tests del módulo de histórico SQLite."""
from pathlib import Path

import pandas as pd
import pytest

from app.core.history import (
    HIGHER_IS_BETTER,
    LOWER_IS_BETTER,
    clear_all,
    compare_runs,
    count_runs,
    delete_run,
    get_run,
    init_db,
    list_runs,
    save_run,
)


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Ruta temporal para la DB de test."""
    return tmp_path / "test_history.db"


@pytest.fixture(autouse=True)
def _setup_db(db_path: Path):
    """Inicializa la DB antes de cada test."""
    init_db(db_path)
    yield


def _make_entry(**overrides):
    """Helper para crear entradas con valores por defecto."""
    base = {
        "strategy": "Momentum",
        "tickers": ["KO", "PEP"],
        "start_date": "2020-01-01",
        "end_date": "2024-12-31",
        "params": {"window": 60},
        "metrics": {"sharpe": 1.21, "max_drawdown": -0.083, "n_trades": 42},
        "notes": None,
    }
    base.update(overrides)
    return base


# ============================================================
#  CRUD básico
# ============================================================
def test_init_db_idempotent(db_path: Path):
    init_db(db_path)
    init_db(db_path)
    assert count_runs(db_path) == 0


def test_save_and_get(db_path: Path):
    run_id = save_run(**_make_entry(), db_path=db_path)
    assert run_id > 0

    entry = get_run(run_id, db_path=db_path)
    assert entry is not None
    assert entry.id == run_id
    assert entry.strategy == "Momentum"
    assert entry.tickers == ["KO", "PEP"]
    assert entry.params["window"] == 60
    assert entry.metrics["sharpe"] == 1.21


def test_save_multiple_and_count(db_path: Path):
    for i in range(5):
        save_run(**_make_entry(notes=f"run {i}"), db_path=db_path)
    assert count_runs(db_path) == 5


def test_list_runs_order_desc(db_path: Path):
    id1 = save_run(**_make_entry(notes="primero"), db_path=db_path)
    id2 = save_run(**_make_entry(notes="segundo"), db_path=db_path)

    entries = list_runs(db_path=db_path)
    assert len(entries) == 2
    assert entries[0].id == id2
    assert entries[1].id == id1


def test_list_runs_filter_by_strategy(db_path: Path):
    save_run(**_make_entry(strategy="Momentum"), db_path=db_path)
    save_run(**_make_entry(strategy="Mean Reversion"), db_path=db_path)
    save_run(**_make_entry(strategy="Momentum"), db_path=db_path)

    momentum = list_runs(strategy="Momentum", db_path=db_path)
    assert len(momentum) == 2
    assert all(e.strategy == "Momentum" for e in momentum)


def test_list_runs_limit(db_path: Path):
    for i in range(10):
        save_run(**_make_entry(notes=f"run {i}"), db_path=db_path)

    entries = list_runs(limit=3, db_path=db_path)
    assert len(entries) == 3


def test_delete_run(db_path: Path):
    run_id = save_run(**_make_entry(), db_path=db_path)
    assert delete_run(run_id, db_path=db_path) is True
    assert get_run(run_id, db_path=db_path) is None
    assert count_runs(db_path) == 0


def test_delete_nonexistent(db_path: Path):
    assert delete_run(9999, db_path=db_path) is False


def test_clear_all(db_path: Path):
    for _ in range(3):
        save_run(**_make_entry(), db_path=db_path)
    n = clear_all(db_path=db_path)
    assert n == 3
    assert count_runs(db_path) == 0


def test_save_rejects_empty_strategy(db_path: Path):
    with pytest.raises(ValueError, match="strategy"):
        save_run(**_make_entry(strategy=""), db_path=db_path)


def test_save_rejects_empty_tickers(db_path: Path):
    with pytest.raises(ValueError, match="tickers"):
        save_run(**_make_entry(tickers=[]), db_path=db_path)


def test_params_json_roundtrip(db_path: Path):
    params = {
        "window": 60,
        "entry": 2.0,
        "nested": {"a": [1, 2, 3], "b": True},
    }
    run_id = save_run(**_make_entry(params=params), db_path=db_path)
    entry = get_run(run_id, db_path=db_path)
    assert entry.params == params


def test_get_nonexistent_returns_none(db_path: Path):
    assert get_run(9999, db_path=db_path) is None


# ============================================================
#  compare_runs
# ============================================================
def test_compare_runs_requires_two(db_path: Path):
    run_id = save_run(**_make_entry(), db_path=db_path)
    with pytest.raises(ValueError, match="al menos 2"):
        compare_runs([run_id], db_path=db_path)


def test_compare_runs_nonexistent(db_path: Path):
    id1 = save_run(**_make_entry(), db_path=db_path)
    with pytest.raises(ValueError, match="no existe"):
        compare_runs([id1, 9999], db_path=db_path)


def test_compare_runs_basic(db_path: Path):
    id1 = save_run(
        **_make_entry(metrics={"sharpe": 1.0, "max_drawdown": -0.10}),
        db_path=db_path,
    )
    id2 = save_run(
        **_make_entry(metrics={"sharpe": 1.5, "max_drawdown": -0.05}),
        db_path=db_path,
    )

    comp = compare_runs([id1, id2], db_path=db_path)

    assert len(comp.entries) == 2
    assert comp.metrics_table.shape[0] >= 2  # al menos sharpe y max_drawdown
    assert id1 in comp.metrics_table.columns
    assert id2 in comp.metrics_table.columns


def test_compare_runs_best_per_metric(db_path: Path):
    id1 = save_run(**_make_entry(metrics={"sharpe": 1.0}), db_path=db_path)
    id2 = save_run(**_make_entry(metrics={"sharpe": 1.5}), db_path=db_path)

    comp = compare_runs([id1, id2], db_path=db_path)

    # Sharpe más alto (id2) es el mejor
    assert comp.best_per_metric["sharpe"] == id2
    assert comp.worst_per_metric["sharpe"] == id1


def test_compare_runs_lower_is_better(db_path: Path):
    id1 = save_run(**_make_entry(metrics={"turnover": 5.0}), db_path=db_path)
    id2 = save_run(**_make_entry(metrics={"turnover": 10.0}), db_path=db_path)

    comp = compare_runs([id1, id2], db_path=db_path)

    # Turnover bajo (id1) es mejor
    assert comp.best_per_metric["turnover"] == id1
    assert comp.worst_per_metric["turnover"] == id2


def test_compare_runs_max_drawdown_direction(db_path: Path):
    """Menos negativo (-5%) es mejor que más negativo (-20%)."""
    id1 = save_run(**_make_entry(metrics={"max_drawdown": -0.20}), db_path=db_path)
    id2 = save_run(**_make_entry(metrics={"max_drawdown": -0.05}), db_path=db_path)

    comp = compare_runs([id1, id2], db_path=db_path)

    assert comp.best_per_metric["max_drawdown"] == id2
    assert comp.worst_per_metric["max_drawdown"] == id1


def test_compare_runs_params_table(db_path: Path):
    id1 = save_run(**_make_entry(params={"window": 30}), db_path=db_path)
    id2 = save_run(**_make_entry(params={"window": 60}), db_path=db_path)

    comp = compare_runs([id1, id2], db_path=db_path)

    assert "window" in comp.params_table.index
    assert comp.params_table.at["window", id1] == 30
    assert comp.params_table.at["window", id2] == 60


def test_compare_runs_handles_missing_metrics(db_path: Path):
    """Si un run tiene menos métricas, las ausentes deben ser NaN, no error."""
    id1 = save_run(
        **_make_entry(metrics={"sharpe": 1.0, "extra": 42}),
        db_path=db_path,
    )
    id2 = save_run(
        **_make_entry(metrics={"sharpe": 1.5}),
        db_path=db_path,
    )

    comp = compare_runs([id1, id2], db_path=db_path)

    # "extra" solo existe en id1 → id2 debe ser NaN
    assert "extra" in comp.metrics_table.index
    assert pd.isna(comp.metrics_table.at["extra", id2])


def test_constants_are_disjoint():
    """Un mismo metric no puede estar en ambas categorías."""
    assert HIGHER_IS_BETTER.isdisjoint(LOWER_IS_BETTER)
