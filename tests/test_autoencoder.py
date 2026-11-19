import numpy as np
import pandas as pd
import pytest
import torch

from src.data.clean import POPULATION_COLUMNS
from src.features.engineer import ATOMIC_COLUMNS
from src.features.masking import simulate_mask
from src.models import autoencoder
from src.models.autoencoder import (
    MaskedAutoencoder,
    consistency_loss,
    input_tensor,
    load_autoencoder,
    predict,
    train_model,
)


def profiles():
    rows = []
    for i in range(10):
        row = dict.fromkeys(POPULATION_COLUMNS, 1)
        row.update({"admin1_code": f"KE{i + 1:03}", "admin1_name": f"County {i}"})
        rows.append(row)
    return pd.DataFrame(rows)


def test_inputs_do_not_reveal_masked_values():
    original = profiles()
    inputs, mask, _ = simulate_mask(original, "random", 0.2)
    tensor = input_tensor(inputs, 1).numpy()
    assert (tensor[:, :42][mask.to_numpy()] == 0).all()
    assert (tensor[:, 42:][mask.to_numpy()] == 0).all()
    assert (tensor[:, 42:][~mask.to_numpy()] == 1).all()


def test_consistency_penalty_reacts_to_reported_discrepancies():
    data = profiles()
    data.loc[:, "population_all_0_4"] = 2
    predicted = torch.ones((10, 42), requires_grad=True)
    loss = consistency_loss(predicted, data, 1.0)
    assert loss.item() > 0
    loss.backward()
    assert predicted.grad is not None
    assert torch.count_nonzero(predicted.grad) > 0


def test_training_and_safe_artifact_roundtrip(tmp_path):
    data = profiles()
    inputs, mask, targets = simulate_mask(data, "random", 0.2)
    config = {"hidden": 8, "epochs": 4, "consistency_weight": 0.1, "learning_rate": 0.01}
    torch.set_num_threads(1)
    model, scale = train_model(data, inputs, mask, targets, config, 42)
    first = predict(model, scale, inputs)
    second, scale2 = train_model(data, inputs, mask, targets, config, 42)
    np.testing.assert_array_equal(first, predict(second, scale2, inputs))
    path = tmp_path / "model.pt"
    torch.save({
        "state_dict": model.state_dict(), "scale": scale,
        "config": config, "columns": ATOMIC_COLUMNS,
    }, path)
    restored, restored_scale = load_autoencoder(path)
    np.testing.assert_array_equal(first, predict(restored, restored_scale, inputs))
    assert np.isfinite(first).all() and (first >= 0).all()
    with pytest.raises(ValueError, match="columns"):
        input_tensor(inputs.drop(columns=ATOMIC_COLUMNS[0]), 1)


def test_model_input_shape():
    assert MaskedAutoencoder(8)(torch.zeros(3, 84)).shape == (3, 42)


def test_end_to_end_autoencoder_training(tmp_path, monkeypatch):
    monkeypatch.setattr(autoencoder, "PATTERNS", ("random",))
    monkeypatch.setattr(autoencoder, "RATES", (0.2,))
    monkeypatch.setattr(autoencoder, "CONFIGS", [
        {"hidden": 8, "epochs": 4, "consistency_weight": 0.1, "learning_rate": 0.01}
    ])
    torch.set_num_threads(1)
    result = autoencoder.run_autoencoders(profiles(), tmp_path / "models", tmp_path / "reports")
    assert len(result) == 1
    assert result.loc[0, "split"] == "validation"
    assert result.loc[0, "validation_targets"] > 0
    assert (tmp_path / "models" / "random_20_autoencoder.pt").is_file()
    assert (tmp_path / "reports" / "autoencoder_validation.csv").is_file()
