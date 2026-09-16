"""
app/core/experiments.py

Reproducibilidad total de experimentos cuantitativos.

Cada vez que se corre un backtest/análisis "serio", esta capa guarda:
    - la configuración exacta usada (dict serializable a JSON)
    - una foto de los datos de entrada (CSV, para poder repetir el
      experimento exactamente igual aunque el proveedor cambie después)
    - los resultados/métricas obtenidos
    - metadatos: timestamp y commit de git (si el repo está disponible)

Esto es lo mínimo que espera un quant researcher serio antes de confiar
en un resultado, y es un argumento de venta fuerte para clientes B2B.

Uso típico:

    from app.core.experiments import save_experiment, load_experiment

    exp = save_experiment(
        config={"strategy": "pairs", "tickers": ["AAPL", "MSFT"]},
        data=prices_df,
        results={"sharpe": 1.42, "max_drawdown": -0.18},
    )
    print(exp.id)

    exp2 = load_experiment(exp.id, base_dir=Path("experiments"))
"""

from __future__ import annotations

import json
import subprocess
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import pandas as pd


class ExperimentError(Exception):
    """Error genérico al guardar, cargar o comparar experimentos."""


@dataclass
class Experiment:
    """Representa un experimento guardado en disco."""

    id: str
    config: dict
    data_snapshot_path: Path
    results_path: Path
    created_at: str
    git_commit: str | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["data_snapshot_path"] = str(self.data_snapshot_path)
        d["results_path"] = str(self.results_path)
        return d

    @property
    def results(self) -> dict:
        """Carga y devuelve las métricas guardadas para este experimento."""
        with open(self.results_path, encoding="utf-8") as fh:
            return cast(dict, json.load(fh))

    @property
    def data(self) -> pd.DataFrame:
        """Carga y devuelve la foto de datos usada en este experimento."""
        return pd.read_csv(self.data_snapshot_path, index_col=0, parse_dates=True)


def _current_git_commit() -> str | None:
    """Devuelve el hash corto del commit actual, o None si no hay repo git."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def save_experiment(
    config: dict,
    data: pd.DataFrame,
    results: dict,
    base_dir: Path = Path("experiments"),
    experiment_id: str | None = None,
) -> Experiment:
    """
    Guarda un experimento completo (config + datos + resultados) en disco.

    Args:
        config: configuración usada para generar los resultados. Debe ser
            serializable a JSON (tickers, parámetros de estrategia, fechas...).
        data: DataFrame de datos de entrada (p. ej. precios) a congelar.
        results: métricas/resultados obtenidos. Debe ser serializable a JSON.
        base_dir: carpeta raíz donde se guardan los experimentos.
        experiment_id: identificador explícito. Si no se indica, se genera
            uno nuevo (uuid4 corto).

    Returns:
        El objeto Experiment creado, con las rutas ya escritas en disco.

    Raises:
        ExperimentError: si config o results no son serializables a JSON,
            o si data está vacío.
    """
    if data is None or data.empty:
        raise ExperimentError("No se puede guardar un experimento con datos vacíos.")

    exp_id = experiment_id or uuid.uuid4().hex[:12]
    exp_dir = Path(base_dir) / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    config_path = exp_dir / "config.json"
    data_path = exp_dir / "data_snapshot.csv"
    results_path = exp_dir / "results.json"

    try:
        config_path.write_text(
            json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except TypeError as exc:
        raise ExperimentError(f"config no es serializable a JSON: {exc}") from exc

    try:
        results_path.write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except TypeError as exc:
        raise ExperimentError(f"results no es serializable a JSON: {exc}") from exc

    data.to_csv(data_path)

    created_at = datetime.now(timezone.utc).isoformat()
    git_commit = _current_git_commit()

    metadata = {"created_at": created_at, "git_commit": git_commit}
    (exp_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return Experiment(
        id=exp_id,
        config=config,
        data_snapshot_path=data_path,
        results_path=results_path,
        created_at=created_at,
        git_commit=git_commit,
        metadata=metadata,
    )


def load_experiment(exp_id: str, base_dir: Path = Path("experiments")) -> Experiment:
    """
    Carga un experimento previamente guardado con `save_experiment`.

    Args:
        exp_id: identificador del experimento (nombre de su carpeta).
        base_dir: carpeta raíz donde se guardan los experimentos.

    Returns:
        El objeto Experiment reconstruido.

    Raises:
        ExperimentError: si el experimento no existe o está incompleto.
    """
    exp_dir = Path(base_dir) / exp_id
    config_path = exp_dir / "config.json"
    data_path = exp_dir / "data_snapshot.csv"
    results_path = exp_dir / "results.json"
    metadata_path = exp_dir / "metadata.json"

    for path in (config_path, data_path, results_path):
        if not path.exists():
            raise ExperimentError(f"Experimento '{exp_id}' incompleto: falta {path.name}.")

    config = json.loads(config_path.read_text(encoding="utf-8"))

    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    return Experiment(
        id=exp_id,
        config=config,
        data_snapshot_path=data_path,
        results_path=results_path,
        created_at=metadata.get("created_at", ""),
        git_commit=metadata.get("git_commit"),
        metadata=metadata,
    )


def diff_experiments(
    exp_id_a: str,
    exp_id_b: str,
    base_dir: Path = Path("experiments"),
) -> pd.DataFrame:
    """
    Compara la config y las métricas de dos experimentos guardados.

    Args:
        exp_id_a: identificador del primer experimento.
        exp_id_b: identificador del segundo experimento.
        base_dir: carpeta raíz donde se guardan los experimentos.

    Returns:
        DataFrame con columnas ['campo', 'grupo', 'A', 'B', 'cambio'],
        una fila por cada clave de config o de resultados que difiere
        (o que solo existe en uno de los dos experimentos). 'cambio' es
        True para todas las filas devueltas (solo se listan diferencias).

    Raises:
        ExperimentError: si alguno de los dos experimentos no existe.
    """
    exp_a = load_experiment(exp_id_a, base_dir=base_dir)
    exp_b = load_experiment(exp_id_b, base_dir=base_dir)

    rows = []
    for grupo, dict_a, dict_b in (
        ("config", exp_a.config, exp_b.config),
        ("results", exp_a.results, exp_b.results),
    ):
        claves = sorted(set(dict_a) | set(dict_b))
        for clave in claves:
            val_a = dict_a.get(clave, "<ausente>")
            val_b = dict_b.get(clave, "<ausente>")
            if val_a != val_b:
                rows.append(
                    {
                        "campo": clave,
                        "grupo": grupo,
                        "A": val_a,
                        "B": val_b,
                        "cambio": True,
                    }
                )

    return pd.DataFrame(rows, columns=["campo", "grupo", "A", "B", "cambio"])


def list_experiments(base_dir: Path = Path("experiments")) -> list[str]:
    """
    Lista los ids de todos los experimentos guardados bajo `base_dir`.

    Args:
        base_dir: carpeta raíz donde se guardan los experimentos.

    Returns:
        Lista de ids de experimento, ordenada alfabéticamente. Lista
        vacía si `base_dir` no existe todavía.
    """
    root = Path(base_dir)
    if not root.exists():
        return []

    return sorted(
        p.name
        for p in root.iterdir()
        if p.is_dir() and (p / "config.json").exists()
    )
