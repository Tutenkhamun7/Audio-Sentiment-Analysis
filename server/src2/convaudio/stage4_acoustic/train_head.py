"""Training script for WavLM frame-level affect head with corpus-licence manifest check."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset

from convaudio.licences import assert_licence_allowed, load_licence_policy
from convaudio.stage4_acoustic.encoder_wavlm import FrameAffectHead


class SyntheticAffectDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Synthetic dataset for training demonstration and verification."""

    def __init__(self, n_samples: int = 100, hidden_dim: int = 768, seq_len: int = 50) -> None:
        self.features = torch.randn(n_samples, seq_len, hidden_dim)
        # Targets: valence and arousal in [-1, 1]
        self.targets = torch.tanh(torch.randn(n_samples, seq_len, 2))

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]


def verify_corpus_licence(manifest_path: Path | str) -> dict[str, Any]:
    """Verify that training corpus licence complies with strict allowlist policy.

    Raises LicenceViolation on any non-commercial or denied licence.
    """
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Corpus licence manifest not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f)

    if not isinstance(manifest, dict):
        raise ValueError("Manifest must be a YAML dictionary")

    corpus_name = manifest.get("corpus_name", "unknown")
    corpus_licence = manifest.get("corpus_licence", "UNKNOWN")
    code_licence = manifest.get("code_licence", "MIT")

    # Enforce licence policy
    policy = load_licence_policy()
    assert_licence_allowed(code_licence, corpus_licence, policy=policy)

    print(f"[Licence Check] Corpus '{corpus_name}' with licence '{corpus_licence}' approved for training.")
    return manifest


def train(
    manifest_path: Path | str,
    output_head_path: Path | str,
    epochs: int = 2,
    batch_size: int = 4,
    lr: float = 1e-4,
    hidden_dim: int = 768,
) -> None:
    """Train FrameAffectHead after verifying corpus licence."""
    # 1. Enforce corpus licence check BEFORE any training begins
    verify_corpus_licence(manifest_path)

    # 2. Setup model and optimizer
    head = FrameAffectHead(hidden_dim=hidden_dim)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # 3. Dataloader
    dataset = SyntheticAffectDataset(n_samples=32, hidden_dim=hidden_dim, seq_len=50)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    head.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for features, targets in dataloader:
            optimizer.zero_grad()
            preds = head(features)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

    # 4. Save trained head weights
    out_p = Path(output_head_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), str(out_p))
    print(f"Trained head weights saved to {out_p}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train WavLM Frame Affect Head")
    parser.add_argument("--manifest", required=True, help="Path to corpus licence manifest YAML")
    parser.add_argument("--output", required=True, help="Output path for head weights .pt")
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()

    train(manifest_path=args.manifest, output_head_path=args.output, epochs=args.epochs)
