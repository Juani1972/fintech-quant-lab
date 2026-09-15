"""Tests del módulo de histórico SQLite."""
from pathlib import Path

import pytest

from app.core.history import (
    clear_all,
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
    # Limpieza no necesaria: tmp_path se borra solo


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


def test_init_db_idempotent(db_path: Path):
    """Llamar init_db varias veces no debe fallar."""
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
    # El más reciente primero
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
    """Params complejos deben sobrevivir el roundtrip JSON."""
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
