"""Persistencia local de backtests ejecutados en SQLite.

Cada vez que el usuario pulsa "Guardar" en la página de Backtest, se
inserta una fila con: fecha, estrategia, tickers, rango de fechas,
parámetros y métricas (como JSON).

La base de datos vive en `data/history.db` (excluida de git).

Incluye funciones de comparación para contrastar varias corridas
lado a lado (ver `compare_runs`).

Uso:
    from app.core.history import init_db, save_run, list_runs, compare_runs

    init_db()
    run_id = save_run(
        strategy="Pairs Trading",
        tickers=["KO", "PEP"],
        start_date="2020-01-01",
        end_date="2024-12-31",
        params={"window": 60, "entry": 2.0},
        metrics={"sharpe": 1.21, "max_drawdown": -0.083},
        notes="Prueba inicial",
    )
    comparison = compare_runs([1, 2, 3])
"""
from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

# ============================================================
#  Configuración
# ============================================================
DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "history.db"


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT NOT NULL,
    strategy    TEXT NOT NULL,
    tickers     TEXT NOT NULL,
    start_date  TEXT,
    end_date    TEXT,
    params      TEXT,
    metrics     TEXT,
    notes       TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_runs_strategy   ON runs(strategy);
"""


# Métricas donde "más alto = mejor" (incluye max_drawdown porque es negativo:
# -5% es mejor que -20%).
HIGHER_IS_BETTER: set[str] = {
    "total_return",
    "annual_return",
    "sharpe",
    "sortino",
    "calmar",
    "max_drawdown",
    "win_rate",
    "profit_factor",
    "avg_trade_pnl",
    "exposure",
}

# Métricas donde "más bajo = mejor".
LOWER_IS_BETTER: set[str] = {
    "turnover",
    "avg_bars_held",
    "annual_volatility",
}


# ============================================================
#  Modelo
# ============================================================
@dataclass
class HistoryEntry:
    """Entrada del histórico de backtests."""
    id: int | None
    created_at: str
    strategy: str
    tickers: list[str]
    start_date: str | None
    end_date: str | None
    params: dict[str, Any]
    metrics: dict[str, Any]
    notes: str | None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> HistoryEntry:
        """Construye una entrada desde una fila de SQLite."""
        return cls(
            id=row["id"],
            created_at=row["created_at"],
            strategy=row["strategy"],
            tickers=json.loads(row["tickers"]),
            start_date=row["start_date"],
            end_date=row["end_date"],
            params=json.loads(row["params"]) if row["params"] else {},
            metrics=json.loads(row["metrics"]) if row["metrics"] else {},
            notes=row["notes"],
        )

    def to_dict(self) -> dict[str, Any]:
        """Devuelve la entrada como dict (útil para st.dataframe)."""
        return asdict(self)


@dataclass
class RunComparison:
    """Resultado de comparar varios runs."""
    run_ids: list[int]
    entries: list[HistoryEntry]
    metrics_table: pd.DataFrame     # index=metric, columns=run_id
    params_table: pd.DataFrame      # index=param, columns=run_id
    best_per_metric: dict[str, int]
    worst_per_metric: dict[str, int]


# ============================================================
#  Conexión
# ============================================================
@contextmanager
def _connect(db_path: Path | None = None) -> Iterator[sqlite3.Connection]:
    """Context manager que abre y cierra la conexión."""
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Path | None = None) -> None:
    """Crea las tablas e índices si no existen. Idempotente."""
    with _connect(db_path) as conn:
        conn.executescript(SCHEMA)


# ============================================================
#  CRUD
# ============================================================
def save_run(
    strategy: str,
    tickers: list[str],
    start_date: str | None,
    end_date: str | None,
    params: dict[str, Any],
    metrics: dict[str, Any],
    notes: str | None = None,
    db_path: Path | None = None,
) -> int:
    """Inserta una nueva entrada. Devuelve el `id` asignado."""
    if not strategy:
        raise ValueError("strategy no puede estar vacío.")
    if not tickers:
        raise ValueError("tickers no puede estar vacío.")

    created_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    with _connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO runs
                (created_at, strategy, tickers, start_date, end_date,
                 params, metrics, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                strategy,
                json.dumps(list(tickers)),
                start_date,
                end_date,
                json.dumps(params, default=str),
                json.dumps(metrics, default=str),
                notes,
            ),
        )
        assert cur.lastrowid is not None, "lastrowid tras un INSERT exitoso nunca es None"
        return int(cur.lastrowid)


def list_runs(
    limit: int = 100,
    strategy: str | None = None,
    db_path: Path | None = None,
) -> list[HistoryEntry]:
    """Lista las entradas más recientes, con filtro opcional por estrategia."""
    with _connect(db_path) as conn:
        if strategy:
            rows = conn.execute(
                "SELECT * FROM runs WHERE strategy = ? "
                "ORDER BY created_at DESC, id DESC LIMIT ?",
                (strategy, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM runs ORDER BY created_at DESC, id DESC LIMIT ?",
                (limit,),
            ).fetchall()
    return [HistoryEntry.from_row(r) for r in rows]


def get_run(run_id: int, db_path: Path | None = None) -> HistoryEntry | None:
    """Devuelve una entrada por id, o None si no existe."""
    with _connect(db_path) as conn:
        row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    return HistoryEntry.from_row(row) if row else None


def delete_run(run_id: int, db_path: Path | None = None) -> bool:
    """Elimina una entrada. Devuelve True si existía."""
    with _connect(db_path) as conn:
        cur = conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
        return cur.rowcount > 0


def clear_all(db_path: Path | None = None) -> int:
    """Borra todas las entradas. Devuelve cuántas se eliminaron."""
    with _connect(db_path) as conn:
        cur = conn.execute("DELETE FROM runs")
        return cur.rowcount


def count_runs(db_path: Path | None = None) -> int:
    """Cuenta total de entradas."""
    with _connect(db_path) as conn:
        row = conn.execute("SELECT COUNT(*) AS n FROM runs").fetchone()
    return int(row["n"])


# ============================================================
#  Comparación
# ============================================================
def compare_runs(
    run_ids: list[int],
    db_path: Path | None = None,
) -> RunComparison:
    """Compara varias corridas guardadas.

    Args:
        run_ids: Lista de IDs a comparar (mínimo 2).
        db_path: Ruta opcional a la DB.

    Returns:
        RunComparison con tablas de métricas y parámetros, y los mejores
        y peores por métrica.

    Raises:
        ValueError: Si se pasan menos de 2 IDs o alguno no existe.
    """
    if len(run_ids) < 2:
        raise ValueError("Se necesitan al menos 2 runs para comparar.")

    entries: list[HistoryEntry] = []
    for rid in run_ids:
        entry = get_run(rid, db_path=db_path)
        if entry is None:
            raise ValueError(f"Run #{rid} no existe.")
        entries.append(entry)

    # --- Tabla de métricas ---
    metrics_data: dict[int, dict[str, Any]] = {}
    for e in entries:
        assert e.id is not None, "entries viene de get_run(), siempre tiene id"
        metrics_data[e.id] = e.metrics or {}
    metrics_table = pd.DataFrame(metrics_data)
    metrics_table.index.name = "metric"

    # --- Tabla de parámetros ---
    params_data: dict[int, dict[str, Any]] = {}
    for e in entries:
        assert e.id is not None, "entries viene de get_run(), siempre tiene id"
        params_data[e.id] = e.params or {}
    params_table = pd.DataFrame(params_data)
    params_table.index.name = "param"

    # --- Mejor/peor por métrica ---
    best_per_metric: dict[str, int] = {}
    worst_per_metric: dict[str, int] = {}

    for metric in metrics_table.index:
        row = metrics_table.loc[metric].dropna()
        if len(row) == 0:
            continue
        if metric in HIGHER_IS_BETTER:
            best_per_metric[metric] = int(row.idxmax())
            worst_per_metric[metric] = int(row.idxmin())
        elif metric in LOWER_IS_BETTER:
            best_per_metric[metric] = int(row.idxmin())
            worst_per_metric[metric] = int(row.idxmax())
        # métricas neutras no aparecen en best/worst

    return RunComparison(
        run_ids=list(run_ids),
        entries=entries,
        metrics_table=metrics_table,
        params_table=params_table,
        best_per_metric=best_per_metric,
        worst_per_metric=worst_per_metric,
    )
