"""Train masked autoencoders with county-grouped tuning and consistency supervision."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from torch import nn

from src.data.clean import PROCESSED_PATH, population_column
from src.features.audit import audit_masked_inputs, audit_splits
from src.features.engineer import AGES, ATOMIC_COLUMNS, engineer_features
from src.features.masking import PATTERNS, RATES, simulate_mask
from src.features.split import split_counties
from src.models.evaluation import mask_fingerprint
from src.models.metrics import consistency_metrics, reconstruction_metrics

CONFIGS = [
    {"hidden": hidden, "consistency_weight": weight, "epochs": 180, "learning_rate": 0.01}
    for hidden in (32, 64) for weight in (0.0, 0.1)
]


class MaskedAutoencoder(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(84, hidden), nn.ReLU(), nn.Linear(hidden, 16), nn.ReLU(),
            nn.Linear(16, hidden), nn.ReLU(), nn.Linear(hidden, 42), nn.Softplus(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


def input_tensor(inputs: pd.DataFrame, scale: float) -> torch.Tensor:
    if list(inputs.columns) != ATOMIC_COLUMNS or not np.isfinite(scale) or scale <= 0:
        raise ValueError("Expected atomic population columns and a positive finite scale")
    values = inputs.to_numpy(dtype=float)
    if np.isinf(values).any() or np.any(values[np.isfinite(values)] < 0):
        raise ValueError("Inputs must be non-negative, finite or missing")
    observed = np.isfinite(values)
    features = np.concatenate([np.where(observed, values / scale, 0), observed], axis=1)
    return torch.tensor(features, dtype=torch.float32)


def consistency_data(
    profiles: pd.DataFrame, scale: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    constraints = []
    for age_index, age in enumerate(AGES):
        constraints.append(([2 * age_index, 2 * age_index + 1], population_column("all", age)))
    for offset, gender in enumerate(("f", "m")):
        constraints.append((list(range(offset, 42, 2)), population_column(gender, "all")))
    constraints.append((list(range(42)), population_column("all", "all")))
    true_atomic = profiles[ATOMIC_COLUMNS].to_numpy(dtype=float)
    coefficients = np.zeros((42, len(constraints)), dtype=np.float32)
    labels = np.zeros((len(profiles), len(constraints)), dtype=np.float32)
    valid = np.zeros((len(profiles), len(constraints)), dtype=bool)
    for constraint_index, (indices, column) in enumerate(constraints):
        reported = pd.to_numeric(profiles[column], errors="raise").to_numpy(dtype=float)
        usable = np.isfinite(reported) & np.isfinite(true_atomic[:, indices]).all(axis=1)
        coefficients[indices, constraint_index] = 1 / len(indices)
        labels[usable, constraint_index] = reported[usable] / scale / len(indices)
        valid[:, constraint_index] = usable
    if not valid.any():
        raise ValueError("No complete aggregate labels available for consistency supervision")
    return torch.from_numpy(coefficients), torch.from_numpy(labels), torch.from_numpy(valid)


def consistency_loss(
    completed: torch.Tensor, profiles: pd.DataFrame, scale: float
) -> torch.Tensor:
    """Compare completed components against available reported aggregate training labels."""
    data = consistency_data(profiles, scale)
    return _consistency_loss(completed, data)


def _consistency_loss(
    completed: torch.Tensor, data: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
) -> torch.Tensor:
    coefficients, labels, valid = data
    difference = completed @ coefficients - labels
    counts = valid.sum(dim=0)
    per_constraint = (difference.square() * valid).sum(dim=0) / counts.clamp(min=1)
    return per_constraint[counts > 0].mean()


def train_model(
    profiles: pd.DataFrame,
    inputs: pd.DataFrame,
    mask: pd.DataFrame,
    targets: pd.DataFrame,
    config: dict,
    seed: int,
) -> tuple[MaskedAutoencoder, float]:
    # Scaling and supervision use only the counties passed to this training fold.
    population = profiles[ATOMIC_COLUMNS].to_numpy(dtype=float)
    observed = population[np.isfinite(population)]
    if not len(observed) or np.any(observed < 0):
        raise ValueError("Training populations must contain non-negative observed values")
    scale = float(np.mean(observed))
    if scale <= 0:
        raise ValueError("Training population scale must be positive")
    torch.manual_seed(seed)
    model = MaskedAutoencoder(config["hidden"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    x = input_tensor(inputs, scale)
    missing = torch.tensor(mask.to_numpy(dtype=bool))
    expected = torch.tensor(
        targets.fillna(0).to_numpy(dtype=np.float32) / scale, dtype=torch.float32
    )
    if not missing.any():
        raise ValueError("Training mask contains no held-out values")
    observed_input = x[:, :42]
    constraints = (
        consistency_data(profiles, scale) if config["consistency_weight"] else None
    )
    for _ in range(config["epochs"]):
        prediction = model(x)
        completed = torch.where(missing, prediction, observed_input)
        loss = (prediction[missing] - expected[missing]).square().mean()
        if constraints is not None:
            loss = loss + config["consistency_weight"] * _consistency_loss(completed, constraints)
        if not torch.isfinite(loss):
            raise ValueError("Autoencoder training produced a non-finite loss")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    return model, scale


def predict(model: MaskedAutoencoder, scale: float, inputs: pd.DataFrame) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        result = model(input_tensor(inputs, scale)).numpy().astype(float) * scale
    if not np.isfinite(result).all():
        raise ValueError("Autoencoder produced non-finite predictions")
    return result


def load_autoencoder(path: Path) -> tuple[MaskedAutoencoder, float]:
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if artifact["columns"] != ATOMIC_COLUMNS:
        raise ValueError("Artifact demographic columns do not match the current schema")
    model = MaskedAutoencoder(artifact["config"]["hidden"])
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    return model, float(artifact["scale"])


def tune(
    profiles: pd.DataFrame, inputs: pd.DataFrame, mask: pd.DataFrame,
    targets: pd.DataFrame, seed: int,
) -> tuple[dict, float]:
    groups = profiles["admin1_code"].to_numpy()
    eligible = mask.any(axis=1).to_numpy()
    indices = np.flatnonzero(eligible)
    if len(np.unique(groups[indices])) < 3:
        raise ValueError("At least three masked training counties are needed for grouped CV")
    folds = list(GroupKFold(n_splits=3).split(indices, groups=groups[indices]))
    scores = []
    for config in CONFIGS:
        fold_scores = []
        for train_pos, valid_pos in folds:
            train, valid = indices[train_pos], indices[valid_pos]
            model, scale = train_model(
                profiles.iloc[train], inputs.iloc[train], mask.iloc[train],
                targets.iloc[train], config, seed,
            )
            predicted = predict(model, scale, inputs.iloc[valid])
            heldout = mask.iloc[valid].to_numpy(dtype=bool)
            actual = targets.iloc[valid].to_numpy(dtype=float)[heldout]
            fold_scores.append(reconstruction_metrics(actual, predicted[heldout])["mae"])
        scores.append(float(np.mean(fold_scores)))
    best = int(np.argmin(scores))
    return CONFIGS[best].copy(), scores[best]


def run_autoencoders(
    profiles: pd.DataFrame, model_dir: Path, report_dir: Path, seed: int = 42
) -> pd.DataFrame:
    if profiles["admin1_code"].duplicated().any():
        raise ValueError("Expected one profile per county")
    splits = split_counties(profiles, seed)
    audit_splits(splits)
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for pattern in PATTERNS:
        for rate in RATES:
            data = {}
            for name in ("train", "validation"):
                split = splits[name]
                inputs, mask, targets = simulate_mask(split, pattern, rate, seed)
                audit_masked_inputs(split, inputs, mask, targets, engineer_features(inputs))
                data[name] = (inputs, mask, targets)
            inputs, mask, targets = data["train"]
            config, cv_mae = tune(splits["train"], inputs, mask, targets, seed)
            model, scale = train_model(splits["train"], inputs, mask, targets, config, seed)
            inputs, mask, targets = data["validation"]
            prediction = predict(model, scale, inputs)
            selected = mask.to_numpy(dtype=bool)
            actual = targets.to_numpy(dtype=float)[selected]
            complete = inputs.to_numpy(dtype=float, copy=True)
            complete[selected] = prediction[selected]
            reconstructed = pd.concat([
                splits["validation"].drop(columns=ATOMIC_COLUMNS),
                pd.DataFrame(complete, columns=ATOMIC_COLUMNS, index=inputs.index),
            ], axis=1)
            results.append({
                "pattern": pattern, "rate": rate, "model": "autoencoder", "split": "validation",
                "train_counties_with_targets": int(data["train"][1].any(axis=1).sum()),
                "train_targets": int(data["train"][1].to_numpy().sum()),
                "validation_targets": int(selected.sum()), "cv_mae": cv_mae,
                "mask_sha256": mask_fingerprint(splits["validation"]["admin1_code"], mask),
                "best_params": json.dumps(config),
                **reconstruction_metrics(actual, prediction[selected]),
                **consistency_metrics(reconstructed),
            })
            label = f"{pattern}_{int(rate * 100)}_autoencoder"
            torch.save({
                "state_dict": model.state_dict(), "scale": scale, "config": config,
                "pattern": pattern, "rate": rate, "seed": seed, "columns": ATOMIC_COLUMNS,
            }, model_dir / f"{label}.pt")
            print(f"{label}: validation MAE={results[-1]['mae']:.2f}", flush=True)
    table = pd.DataFrame(results)
    table.to_csv(report_dir / "autoencoder_validation.csv", index=False)
    return table


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROCESSED_PATH)
    parser.add_argument("--model-dir", type=Path, default=Path("models/autoencoders"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/autoencoders"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.set_num_threads(1)
    profiles = pd.read_csv(args.input)
    table = run_autoencoders(profiles, args.model_dir, args.report_dir, args.seed)
    metadata = {
        "input_sha256": sha256_file(args.input), "seed": args.seed, "rows": len(profiles),
        "results": len(table), "split": "validation",
    }
    (args.report_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
