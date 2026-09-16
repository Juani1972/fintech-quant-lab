"""
tests/test_experiments.py

Tests unitarios de app/core/experiments.py.

Usan `tmp_path` de pytest como `base_dir`, así que no tocan el disco real
del repo. No requieren red ni git (git_commit puede salir None y es válido).
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.core.experiments import (
    Experiment,
    ExperimentError,
    diff_experiments,
    list_experiments,
    load_experiment,
    save_experiment,
)


@pytest.fixture
def sample_data() -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=5, freq="B")
    return pd.DataFrame({"AAPL": [100, 101, 102, 101, 103]}, index=idx)


@pytest.fixture
def sample_config() -> dict:
    return {"strategy": "pairs", "tickers": ["AAPL", "MSFT"], "lookback": 30}


@pytest.fixture
def sample_results() -> dict:
    return {"sharpe": 1.42, "max_drawdown": -0.18}


# ---------------------------------------------------------------------------
# save_experiment
# ---------------------------------------------------------------------------

class TestSaveExperiment:
    def test_returns_experiment_with_generated_id(self, tmp_path, sample_config, sample_data, sample_results):
        exp = save_experiment(sample_config, sample_data, sample_results, base_dir=tmp_path)

        assert isinstance(exp, Experiment)
        assert exp.id  # no vacío
        assert exp.config == sample_config

    def test_writes_files_to_disk(self, tmp_path, sample_config, sample_data, sample_results):
        exp = save_experiment(sample_config, sample_data, sample_results, base_dir=tmp_path)

        exp_dir = tmp_path / exp.id
        assert (exp_dir / "config.json").exists()
        assert (exp_dir / "data_snapshot.csv").exists()
        assert (exp_dir / "results.json").exists()
        assert (exp_dir / "metadata.json").exists()

    def test_custom_experiment_id_is_used(self, tmp_path, sample_config, sample_data, sample_results):
        exp = save_experiment(
            sample_config, sample_data, sample_results, base_dir=tmp_path, experiment_id="mi-experimento"
        )
        assert exp.id == "mi-experimento"
        assert (tmp_path / "mi-experimento").exists()

    def test_empty_data_raises(self, tmp_path, sample_config, sample_results):
        with pytest.raises(ExperimentError, match="datos vacíos"):
            save_experiment(sample_config, pd.DataFrame(), sample_results, base_dir=tmp_path)

    def test_non_serializable_config_raises(self, tmp_path, sample_data, sample_results):
        config = {"objeto_raro": object()}
        with pytest.raises(ExperimentError, match="config no es serializable"):
            save_experiment(config, sample_data, sample_results, base_dir=tmp_path)

    def test_non_serializable_results_raises(self, tmp_path, sample_config, sample_data):
        results = {"objeto_raro": object()}
        with pytest.raises(ExperimentError, match="results no es serializable"):
            save_experiment(sample_config, sample_data, results, base_dir=tmp_path)

    def test_results_and_data_roundtrip(self, tmp_path, sample_config, sample_data, sample_results):
        exp = save_experiment(sample_config, sample_data, sample_results, base_dir=tmp_path)

        assert exp.results == sample_results
        pd.testing.assert_frame_equal(exp.data, sample_data, check_freq=False)


# ---------------------------------------------------------------------------
# load_experiment
# ---------------------------------------------------------------------------

class TestLoadExperiment:
    def test_roundtrip_matches_saved_experiment(self, tmp_path, sample_config, sample_data, sample_results):
        saved = save_experiment(sample_config, sample_data, sample_results, base_dir=tmp_path)
        loaded = load_experiment(saved.id, base_dir=tmp_path)

        assert loaded.id == saved.id
        assert loaded.config == sample_config
        assert loaded.results == sample_results

    def test_missing_experiment_raises(self, tmp_path):
        with pytest.raises(ExperimentError, match="incompleto"):
            load_experiment("no-existe", base_dir=tmp_path)

    def test_incomplete_experiment_raises(self, tmp_path):
        exp_dir = tmp_path / "roto"
        exp_dir.mkdir()
        (exp_dir / "config.json").write_text("{}", encoding="utf-8")
        # Faltan data_snapshot.csv y results.json.

        with pytest.raises(ExperimentError, match="incompleto"):
            load_experiment("roto", base_dir=tmp_path)


# ---------------------------------------------------------------------------
# diff_experiments
# ---------------------------------------------------------------------------

class TestDiffExperiments:
    def test_detects_config_and_results_changes(self, tmp_path, sample_data):
        save_experiment(
            {"strategy": "pairs", "lookback": 30},
            sample_data,
            {"sharpe": 1.0},
            base_dir=tmp_path,
            experiment_id="exp_a",
        )
        save_experiment(
            {"strategy": "pairs", "lookback": 60},
            sample_data,
            {"sharpe": 1.5},
            base_dir=tmp_path,
            experiment_id="exp_b",
        )

        diff = diff_experiments("exp_a", "exp_b", base_dir=tmp_path)

        campos = set(diff["campo"])
        assert "lookback" in campos
        assert "sharpe" in campos
        assert "strategy" not in campos  # no cambió

    def test_identical_experiments_have_empty_diff(self, tmp_path, sample_config, sample_data, sample_results):
        save_experiment(sample_config, sample_data, sample_results, base_dir=tmp_path, experiment_id="a")
        save_experiment(sample_config, sample_data, sample_results, base_dir=tmp_path, experiment_id="b")

        diff = diff_experiments("a", "b", base_dir=tmp_path)

        assert diff.empty

    def test_key_only_in_one_experiment_is_flagged(self, tmp_path, sample_data):
        save_experiment(
            {"strategy": "pairs"}, sample_data, {"sharpe": 1.0}, base_dir=tmp_path, experiment_id="a"
        )
        save_experiment(
            {"strategy": "pairs", "extra_param": True},
            sample_data,
            {"sharpe": 1.0},
            base_dir=tmp_path,
            experiment_id="b",
        )

        diff = diff_experiments("a", "b", base_dir=tmp_path)
        fila = diff[diff["campo"] == "extra_param"].iloc[0]

        assert fila["A"] == "<ausente>"
        assert bool(fila["B"]) is True

    def test_missing_experiment_raises(self, tmp_path, sample_config, sample_data, sample_results):
        save_experiment(sample_config, sample_data, sample_results, base_dir=tmp_path, experiment_id="a")

        with pytest.raises(ExperimentError):
            diff_experiments("a", "no-existe", base_dir=tmp_path)


# ---------------------------------------------------------------------------
# list_experiments
# ---------------------------------------------------------------------------

class TestListExperiments:
    def test_empty_base_dir_returns_empty_list(self, tmp_path):
        assert list_experiments(base_dir=tmp_path / "no-creado") == []

    def test_lists_saved_experiment_ids_sorted(self, tmp_path, sample_config, sample_data, sample_results):
        save_experiment(sample_config, sample_data, sample_results, base_dir=tmp_path, experiment_id="zeta")
        save_experiment(sample_config, sample_data, sample_results, base_dir=tmp_path, experiment_id="alfa")

        assert list_experiments(base_dir=tmp_path) == ["alfa", "zeta"]

    def test_ignores_directories_without_config(self, tmp_path, sample_config, sample_data, sample_results):
        save_experiment(sample_config, sample_data, sample_results, base_dir=tmp_path, experiment_id="valido")
        (tmp_path / "carpeta_suelta").mkdir()

        assert list_experiments(base_dir=tmp_path) == ["valido"]
