"""Train a neural surrogate for the unified spectral energy on a CNF instance."""
from __future__ import annotations

import math
import pathlib
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cnf_partition import CNFPartitioner, build_conflict_graph, load_dimacs


def features_from_alphas(alphas: np.ndarray) -> np.ndarray:
    """Convert alpha angles with shape (N, segments) or (segments,) to cos/sin features."""
    alphas = np.array(alphas, dtype=np.float32)
    if alphas.ndim == 1:
        alphas = alphas[None, :]
    cos_part = np.cos(alphas)
    sin_part = np.sin(alphas)
    return np.concatenate([cos_part, sin_part], axis=-1)


def sample_dataset(
    partitioner: CNFPartitioner,
    *,
    samples: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    alphas = rng.uniform(0.0, 2.0 * math.pi, size=(samples, partitioner.num_segments))
    energies = np.array([partitioner.unified_energy(alpha) for alpha in alphas], dtype=np.float32)
    features = features_from_alphas(alphas).astype(np.float32)
    return features, energies


@dataclass
class TrainConfig:
    cnf_path: pathlib.Path
    blocks: int
    samples: int = 4000
    train_split: float = 0.8
    seed: int = 2025
    batch_size: int = 128
    epochs: int = 200
    hidden: int = 128
    lr: float = 1e-3


class EnergyMLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_model(config: TrainConfig) -> None:
    formula = load_dimacs(config.cnf_path)
    graph = build_conflict_graph(formula)
    partitioner = CNFPartitioner(graph, num_segments=config.blocks)

    features, energies = sample_dataset(partitioner, samples=config.samples, seed=config.seed)
    n_train = int(config.train_split * config.samples)
    train_x, val_x = features[:n_train], features[n_train:]
    train_y, val_y = energies[:n_train], energies[n_train:]

    train_ds = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    val_ds = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnergyMLP(train_x.shape[1], config.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_fn = nn.MSELoss()

    def evaluate(loader: DataLoader) -> Tuple[float, float]:
        model.eval()
        total_loss = 0.0
        preds, targets = [], []
        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                pred = model(x_batch)
                total_loss += loss_fn(pred, y_batch).item() * x_batch.size(0)
                preds.append(pred.cpu().numpy())
                targets.append(y_batch.cpu().numpy())
        preds_arr = np.concatenate(preds)
        targets_arr = np.concatenate(targets)
        corr = np.corrcoef(preds_arr, targets_arr)[0, 1]
        return total_loss / len(loader.dataset), float(corr)

    best_val = float("inf")
    best_state = None
    for epoch in range(1, config.epochs + 1):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
        val_loss, val_corr = evaluate(val_loader)
        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict()
        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}: val_loss={val_loss:.4f}, val_corr={val_corr:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    val_loss, val_corr = evaluate(val_loader)
    print(f"Final validation: loss={val_loss:.4f}, corr={val_corr:.4f}")

    # Gradient-based optimization using the learned surrogate
    with torch.no_grad():
        base_energy = float(np.min(energies))
        print(f"Best sampled energy: {base_energy:.4f}")

    model.eval()
    phi = torch.randn(config.blocks, device=device, requires_grad=True)
    optimizer_phi = torch.optim.Adam([phi], lr=1e-2)
    for step in range(200):
        optimizer_phi.zero_grad()
        features_phi = torch.stack((torch.cos(phi), torch.sin(phi)), dim=-1).reshape(1, -1)
        energy_pred = model(features_phi)
        energy_pred.backward()
        optimizer_phi.step()
    with torch.no_grad():
        alpha = torch.remainder(phi, 2 * math.pi).cpu().numpy()
        feature_np = features_from_alphas(alpha).astype(np.float32)
    feature_tensor = torch.from_numpy(feature_np).to(device)
    with torch.no_grad():
        surrogate_energy = float(model(feature_tensor).cpu().item())
    true_energy = partitioner.unified_energy(alpha.squeeze())

    print(f"Surrogate-optimized energy (model): {surrogate_energy:.4f}")
    print(f"Surrogate-optimized energy (true): {true_energy:.4f}")


if __name__ == "__main__":
    config = TrainConfig(
        cnf_path=pathlib.Path("samples/example.cnf"),
        blocks=3,
        samples=6000,
        epochs=300,
        hidden=128,
        lr=5e-4,
    )
    train_model(config)
