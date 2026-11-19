import subprocess

import pandas as pd
import pytest

import run_pipeline


def config():
    return {
        "seed": 42,
        "input": "data/raw/source.csv",
        "output_root": "runs",
        "prioritisation": {"budget": 1, "generations": 2},
    }


def test_config_validation(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "seed: 42\ninput: data.csv\noutput_root: runs\n"
        "prioritisation:\n  budget: 10\n  generations: 200\n",
        encoding="utf-8",
    )
    assert run_pipeline.read_config(path)["seed"] == 42
    path.write_text("seed: false\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Config"):
        run_pipeline.read_config(path)
    path.write_text(
        "seed: false\ninput: data.csv\noutput_root: runs\n"
        "prioritisation:\n  budget: 10\n  generations: 200\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="seed"):
        run_pipeline.read_config(path)


def test_pipeline_orchestrates_and_keeps_source_warnings(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(run_pipeline, "ROOT", tmp_path)
    source = tmp_path / "data" / "raw" / "source.csv"
    source.parent.mkdir(parents=True)
    source.write_text("source", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("test", encoding="utf-8")
    profiles = pd.DataFrame({"admin1_code": ["KE001"], "admin1_name": ["County"]})
    checks = pd.DataFrame({"status": ["inconsistent", "consistent", "incomplete"]})
    monkeypatch.setattr(run_pipeline, "load_kenya", lambda path: profiles)
    monkeypatch.setattr(run_pipeline, "clean_population", lambda rows: rows)
    monkeypatch.setattr(run_pipeline, "county_indicators", lambda rows: rows)
    monkeypatch.setattr(run_pipeline, "validate_profiles", lambda rows: checks)
    calls = []
    monkeypatch.setattr(run_pipeline, "command", lambda module, *args: calls.append((module, args)))
    output = tmp_path / "runs" / "first"
    run_pipeline.run_pipeline(config(), config_path, output)
    assert [name for name, _ in calls] == [
        "src.features.prepare", "src.models.baselines", "src.models.autoencoder",
        "src.models.compare_models", "src.optimization.compare",
    ]
    assert "1 inconsistent and 1 incomplete" in caplog.text
    manifest = pd.read_json(output / "manifest.json", typ="series")
    assert manifest["inconsistent_checks"] == 1
    assert manifest["test_evaluated"] is False
    with pytest.raises(FileExistsError):
        run_pipeline.run_pipeline(config(), config_path, output)


def test_failure_does_not_report_completion(tmp_path, monkeypatch):
    monkeypatch.setattr(run_pipeline, "ROOT", tmp_path)
    source = tmp_path / "data" / "raw" / "source.csv"
    source.parent.mkdir(parents=True)
    source.write_text("source", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("test", encoding="utf-8")
    profiles = pd.DataFrame({"admin1_code": ["KE001"]})
    monkeypatch.setattr(run_pipeline, "load_kenya", lambda path: profiles)
    monkeypatch.setattr(run_pipeline, "clean_population", lambda rows: rows)
    monkeypatch.setattr(run_pipeline, "county_indicators", lambda rows: rows)
    monkeypatch.setattr(
        run_pipeline, "validate_profiles", lambda rows: pd.DataFrame({"status": ["consistent"]})
    )

    def fail(*args):
        raise subprocess.CalledProcessError(1, args)

    monkeypatch.setattr(run_pipeline, "command", fail)
    output = tmp_path / "runs" / "failed"
    with pytest.raises(subprocess.CalledProcessError):
        run_pipeline.run_pipeline(config(), config_path, output)
    assert not (output / "manifest.json").exists()


def test_missing_input_fails_without_creating_output(tmp_path, monkeypatch):
    monkeypatch.setattr(run_pipeline, "ROOT", tmp_path)
    output = tmp_path / "runs" / "missing"
    with pytest.raises(FileNotFoundError, match="Population CSV not found"):
        run_pipeline.run_pipeline(config(), tmp_path / "config.yaml", output)
    assert not output.exists()
