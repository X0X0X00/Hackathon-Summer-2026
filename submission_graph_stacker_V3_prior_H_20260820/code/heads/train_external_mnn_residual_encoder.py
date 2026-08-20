from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


DEFAULT_DATA_DIR = Path(
    r"C:\Users\lizhi\Hackathon-Summer-2026\Hackathon-Summer-2026"
    r"\other_model\Hackathon-Summer-2026\work\cache_ext\gene_token"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "external_mnn_residual_encoder"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    probabilities = np.clip(probabilities.astype(np.float64), 1e-12, None)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    predictions = probabilities.argmax(axis=1)
    n_classes = probabilities.shape[1]
    f1_values = []
    supports = []
    for class_id in range(n_classes):
        truth = y_true == class_id
        pred = predictions == class_id
        tp = int(np.sum(truth & pred))
        fp = int(np.sum(~truth & pred))
        fn = int(np.sum(truth & ~pred))
        denominator = 2 * tp + fp + fn
        f1_values.append(0.0 if denominator == 0 else (2.0 * tp) / denominator)
        supports.append(int(truth.sum()))
    f1 = np.asarray(f1_values, dtype=np.float64)
    support = np.asarray(supports, dtype=np.float64)
    return {
        "accuracy": float(np.mean(predictions == y_true)),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.sum(f1 * support) / max(float(support.sum()), 1.0)),
        "log_loss": float(-np.log(probabilities[np.arange(len(y_true)), y_true]).mean()),
    }


def save_probabilities(path: Path, ids: np.ndarray, probabilities: np.ndarray, classes: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Cell_ID", *classes.tolist()])
        for cell_id, row in zip(ids, probabilities):
            writer.writerow([cell_id, *[f"{float(value):.9g}" for value in row]])


class MNNResidualEncoder(nn.Module):
    def __init__(
        self,
        n_genes: int,
        auxiliary_dim: int,
        n_classes: int,
        latent_dim: int,
        n_heads: int,
        dropout: float,
        encoder_hidden_dim: int = 192,
        decoder_hidden_dim: int = 160,
        edge_hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        input_dim = n_genes + auxiliary_dim
        self.current_encoder = nn.Sequential(
            nn.Linear(input_dim, encoder_hidden_dim), nn.GELU(), nn.LayerNorm(encoder_hidden_dim), nn.Dropout(dropout),
            nn.Linear(encoder_hidden_dim, latent_dim), nn.GELU(),
        )
        self.neighbor_encoder = nn.Sequential(
            nn.Linear(input_dim, encoder_hidden_dim), nn.GELU(), nn.LayerNorm(encoder_hidden_dim), nn.Dropout(dropout),
            nn.Linear(encoder_hidden_dim, latent_dim), nn.GELU(),
        )
        self.reconstruction_head = nn.Sequential(
            nn.Linear(latent_dim, encoder_hidden_dim), nn.GELU(), nn.Linear(encoder_hidden_dim, n_genes)
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(5, edge_hidden_dim), nn.GELU(), nn.Linear(edge_hidden_dim, latent_dim)
        )
        self.attention = nn.MultiheadAttention(latent_dim, n_heads, dropout=dropout, batch_first=True)
        self.gate = nn.Sequential(
            nn.Linear(2 * latent_dim, decoder_hidden_dim), nn.GELU(),
            nn.Linear(decoder_hidden_dim, latent_dim), nn.Sigmoid()
        )
        self.fusion_norm = nn.LayerNorm(latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, decoder_hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(decoder_hidden_dim, n_classes)
        )

    def encode_neighbors(self, expression_residual: torch.Tensor, auxiliary: torch.Tensor) -> torch.Tensor:
        auxiliary = auxiliary.clone()
        auxiliary[..., :2] = 0.0
        return self.neighbor_encoder(torch.cat([expression_residual, auxiliary], dim=-1))

    def forward(
        self,
        current_expression: torch.Tensor,
        current_auxiliary: torch.Tensor,
        neighbor_residual: torch.Tensor,
        neighbor_auxiliary: torch.Tensor,
        edge_features: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        current = self.current_encoder(torch.cat([current_expression, current_auxiliary], dim=-1))
        neighbor = self.encode_neighbors(neighbor_residual, neighbor_auxiliary)
        reconstruction = self.reconstruction_head(neighbor)
        tokens = (neighbor + self.edge_encoder(edge_features)).masked_fill(padding_mask.unsqueeze(-1), 0.0)
        attention_mask = padding_mask.clone()
        all_padded = attention_mask.all(dim=1)
        if all_padded.any():
            attention_mask[all_padded, 0] = False
        context, attention = self.attention(
            current.unsqueeze(1), tokens, tokens, key_padding_mask=attention_mask,
            need_weights=True, average_attn_weights=False,
        )
        context = context.squeeze(1)
        if all_padded.any():
            context = context.masked_fill(all_padded.unsqueeze(-1), 0.0)
            attention = attention.masked_fill(all_padded[:, None, None, None], 0.0)
        gate = self.gate(torch.cat([current, context], dim=-1))
        logits = self.decoder(self.fusion_norm(current + gate * context))
        return logits, reconstruction, gate, attention


def gather(indices: torch.Tensor, data: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
    neighbor_indices = data["neighbors"][indices]
    padding = neighbor_indices < 0
    safe = neighbor_indices.clamp_min(0)
    return (
        data["expression"][indices], data["auxiliary"][indices],
        data["residual"][safe], data["auxiliary"][safe],
        data["edges"][indices], padding,
    )


def pretrain(
    model: MNNResidualEncoder,
    data: dict[str, torch.Tensor],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    mask_rate: float,
) -> list[dict[str, float | int | str]]:
    parameters = list(model.neighbor_encoder.parameters()) + list(model.reconstruction_head.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    pool = torch.arange(len(data["residual"]), device=data["residual"].device)
    history = []
    model.train()
    for epoch in range(epochs):
        permutation = pool[torch.randperm(len(pool), device=pool.device)]
        total_loss = 0.0
        seen = 0
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size]
            original = data["residual"][indices]
            mask = torch.rand_like(original) < mask_rate
            masked = original.masked_fill(mask, 0.0)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                latent = model.encode_neighbors(masked.unsqueeze(1), data["auxiliary"][indices].unsqueeze(1))
                reconstruction = model.reconstruction_head(latent).squeeze(1)
                loss = nn.functional.smooth_l1_loss(reconstruction[mask], original[mask])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(parameters, 5.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item()) * len(indices)
            seen += len(indices)
        row = {"stage": "segment_centered_neighbor_pretrain", "epoch": epoch + 1, "loss": total_loss / seen}
        history.append(row)
        print(f"pretrain epoch={epoch + 1}/{epochs} loss={row['loss']:.5f}", flush=True)
    return history


def train_fold(
    model: MNNResidualEncoder,
    data: dict[str, torch.Tensor],
    fit_positions: np.ndarray,
    fold: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    mask_rate: float,
    reconstruction_weight: float,
) -> list[dict[str, float | int | str]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    fit = torch.as_tensor(fit_positions, dtype=torch.long, device=data["expression"].device)
    history = []
    for epoch in range(epochs):
        model.train()
        permutation = fit[torch.randperm(len(fit), device=fit.device)]
        total_loss = total_ce = total_reconstruction = 0.0
        seen = 0
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size]
            inputs = list(gather(indices, data))
            original_neighbors = inputs[2]
            padding = inputs[5]
            mask = (torch.rand_like(original_neighbors) < mask_rate) & ~padding.unsqueeze(-1)
            inputs[2] = original_neighbors.masked_fill(mask, 0.0)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits, reconstruction, _, _ = model(*inputs)
                ce = nn.functional.cross_entropy(logits, data["labels"][indices])
                recon = nn.functional.smooth_l1_loss(reconstruction[mask], original_neighbors[mask]) if mask.any() else torch.zeros((), device=logits.device)
                loss = ce + reconstruction_weight * recon
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            count = len(indices)
            total_loss += float(loss.item()) * count
            total_ce += float(ce.item()) * count
            total_reconstruction += float(recon.item()) * count
            seen += count
        row = {
            "stage": "joint_finetune", "fold": fold, "epoch": epoch + 1,
            "loss": total_loss / seen, "classification_loss": total_ce / seen,
            "neighbor_reconstruction_loss": total_reconstruction / seen,
        }
        history.append(row)
        print(
            f"fold={fold} epoch={epoch + 1}/{epochs} loss={row['loss']:.4f} "
            f"ce={row['classification_loss']:.4f} recon={row['neighbor_reconstruction_loss']:.4f}",
            flush=True,
        )
    return history


@torch.inference_mode()
def predict(model: MNNResidualEncoder, data: dict[str, torch.Tensor], positions: np.ndarray, batch_size: int) -> tuple[np.ndarray, float, float]:
    model.eval()
    probabilities = []
    gate_sum = attention_entropy_sum = 0.0
    seen = 0
    for start in range(0, len(positions), batch_size):
        batch = torch.as_tensor(positions[start : start + batch_size], dtype=torch.long, device=data["expression"].device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits, _, gate, attention = model(*gather(batch, data))
        probability = torch.softmax(logits.float(), dim=1)
        probabilities.append(probability.cpu().numpy())
        gate_sum += float(gate.float().mean(dim=1).sum().item())
        mean_attention = attention.float().mean(dim=1).squeeze(1)
        entropy = -(mean_attention * torch.log(mean_attention.clamp_min(1e-8))).sum(dim=1)
        attention_entropy_sum += float(entropy.sum().item())
        seen += len(batch)
    return np.concatenate(probabilities), gate_sum / seen, attention_entropy_sum / seen


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the large-reference Segment-MNN residual encoder.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--graph-dir", type=Path, default=None)
    parser.add_argument(
        "--centering-mode",
        choices=["auto", "graph", "segment"],
        default="auto",
        help="auto uses graph context_center when present; segment forces the original Segment mean",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--pretrain-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--prediction-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--mask-rate", type=float, default=0.30)
    parser.add_argument("--reconstruction-weight", type=float, default=0.25)
    parser.add_argument("--latent-dim", type=int, default=96)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--encoder-hidden-dim", type=int, default=192)
    parser.add_argument("--decoder-hidden-dim", type=int, default=160)
    parser.add_argument("--edge-hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is intentionally disabled")
    started = time.time()
    set_seed(args.seed)
    device = torch.device("cuda")
    data_dir = args.data_dir.resolve()
    graph_dir = args.graph_dir.resolve() if args.graph_dir is not None else data_dir / "mnn_residual"
    base_graph_dir = data_dir / "mnn_residual"
    output_dir = args.output_dir.resolve()
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    expression_source = graph_dir / "expression.npy"
    auxiliary_source = graph_dir / "auxiliary.npy"
    if not expression_source.exists():
        expression_source = base_graph_dir / "expression.npy"
    if not auxiliary_source.exists():
        auxiliary_source = base_graph_dir / "auxiliary.npy"
    expression_np = np.load(expression_source, mmap_mode="r", allow_pickle=False)
    auxiliary_np = np.load(auxiliary_source, mmap_mode="r", allow_pickle=False)
    neighbors_np = np.load(graph_dir / "neighbors.npy", mmap_mode="r", allow_pickle=False)
    edges_np = np.load(graph_dir / "edge_features.npy", mmap_mode="r", allow_pickle=False)
    labels_np = np.load(data_dir / "labels.npy", allow_pickle=False).astype(np.int64)
    reference_positions = np.load(data_dir / "reference_positions.npy", allow_pickle=False)
    train_positions = np.load(data_dir / "train_positions.npy", allow_pickle=False)
    test_positions = np.load(data_dir / "test_positions.npy", allow_pickle=False)
    folds_all = np.load(data_dir / "folds.npy", allow_pickle=False)
    ids = np.load(data_dir / "ids.npy", allow_pickle=False).astype(str)
    classes = np.load(data_dir / "class_names.npy", allow_pickle=False).astype(str)
    folds = folds_all[train_positions]

    expression_values = np.array(expression_np, dtype=np.float32, copy=True)
    context_center_path = graph_dir / "context_center.npy"
    use_graph_center = context_center_path.exists() and args.centering_mode in {"auto", "graph"}
    if args.centering_mode == "graph" and not context_center_path.exists():
        raise ValueError(f"graph centering requested but missing: {context_center_path}")
    if use_graph_center:
        context_center = np.load(context_center_path, mmap_mode="r", allow_pickle=False)
        residual_values = expression_values - np.asarray(context_center, dtype=np.float32)
        context_rule = "soft-assignment-weighted slot mean expression is subtracted before H aggregation"
        neighbor_rule = "soft-slot-overlap reference-neighbor graph"
    else:
        segment_codes = np.load(base_graph_dir / "segment_codes.npy", allow_pickle=False)
        segment_means = np.load(base_graph_dir / "segment_means.npy", allow_pickle=False)
        residual_values = expression_values - segment_means[segment_codes]
        context_rule = "all-cell Segment mean expression is subtracted before H aggregation"
        neighbor_rule = (
            "soft-slot-overlap reference-neighbor graph"
            if graph_dir != base_graph_dir
            else "exact reciprocal top-8 cosine neighbors within Segment"
        )
    data = {
        "expression": torch.from_numpy(expression_values).to(device),
        "residual": torch.from_numpy(residual_values.astype(np.float32)).to(device),
        "auxiliary": torch.from_numpy(np.array(auxiliary_np, dtype=np.float32, copy=True)).to(device),
        "neighbors": torch.from_numpy(np.array(neighbors_np, dtype=np.int64, copy=True)).to(device),
        "edges": torch.from_numpy(np.array(edges_np, dtype=np.float32, copy=True)).to(device),
        "labels": torch.from_numpy(labels_np).to(device),
    }
    del expression_values, residual_values
    print(
        f"device={torch.cuda.get_device_name(0)} reference={len(reference_positions)} "
        f"train={len(train_positions)} test={len(test_positions)} genes={data['expression'].shape[1]} "
        f"auxiliary={data['auxiliary'].shape[1]}", flush=True,
    )

    def make_model() -> MNNResidualEncoder:
        return MNNResidualEncoder(
            n_genes=data["expression"].shape[1], auxiliary_dim=data["auxiliary"].shape[1],
            n_classes=len(classes), latent_dim=args.latent_dim, n_heads=args.n_heads, dropout=args.dropout,
            encoder_hidden_dim=args.encoder_hidden_dim, decoder_hidden_dim=args.decoder_hidden_dim,
            edge_hidden_dim=args.edge_hidden_dim,
        ).to(device)

    history: list[dict[str, object]] = []
    set_seed(args.seed + 9000)
    pretrain_model = make_model()
    history.extend(pretrain(pretrain_model, data, args.pretrain_epochs, args.batch_size * 2, args.learning_rate, args.mask_rate))
    pretrained_state = {
        name: copy.deepcopy(value.detach().cpu())
        for name, value in pretrain_model.state_dict().items()
        if name.startswith("neighbor_encoder.") or name.startswith("reconstruction_head.")
    }
    del pretrain_model
    torch.cuda.empty_cache()

    oof = np.zeros((len(train_positions), len(classes)), dtype=np.float32)
    test_folds = []
    fold_metrics = []
    fold_diagnostics = []
    for fold in np.sort(np.unique(folds)):
        set_seed(args.seed + 12000 + int(fold))
        held_local = np.flatnonzero(folds == fold)
        fit_competition = train_positions[folds != fold]
        fit_positions = np.concatenate([reference_positions, fit_competition])
        valid_positions = train_positions[held_local]
        model = make_model()
        model.load_state_dict(pretrained_state, strict=False)
        history.extend(
            train_fold(
                model, data, fit_positions, int(fold), args.epochs, args.batch_size,
                args.learning_rate, args.mask_rate, args.reconstruction_weight,
            )
        )
        valid_probability, valid_gate, valid_entropy = predict(model, data, valid_positions, args.prediction_batch_size)
        test_probability, test_gate, test_entropy = predict(model, data, test_positions, args.prediction_batch_size)
        oof[held_local] = valid_probability
        test_folds.append(test_probability)
        result = {"fold": int(fold), "n_valid": int(len(valid_positions)), **metrics(labels_np[valid_positions], valid_probability)}
        fold_metrics.append(result)
        fold_diagnostics.append(
            {"fold": int(fold), "mean_gate_valid": valid_gate, "attention_entropy_valid": valid_entropy,
             "mean_gate_test": test_gate, "attention_entropy_test": test_entropy}
        )
        torch.save(
            {"fold": int(fold), "model_state_dict": model.state_dict(), "configuration": vars(args),
             "class_names": classes.tolist(), "context": context_rule},
            checkpoint_dir / f"mnn_residual_fold{int(fold)}.pt",
        )
        print(
            f"fold={fold} accuracy={result['accuracy']:.4f} macro_f1={result['macro_f1']:.4f} "
            f"log_loss={result['log_loss']:.4f} gate={valid_gate:.4f}", flush=True,
        )
        del model
        torch.cuda.empty_cache()

    test_probability = np.mean(np.stack(test_folds), axis=0).astype(np.float32)
    overall = metrics(labels_np[train_positions], oof)
    save_probabilities(output_dir / "oof_probabilities_mnn_residual_encoder.csv", ids[train_positions], oof, classes)
    save_probabilities(output_dir / "test_probabilities_mnn_residual_encoder.csv", ids[test_positions], test_probability, classes)
    with (output_dir / "submission_mnn_residual_encoder.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Cell_ID", "CellType"])
        for cell_id, prediction in zip(ids[test_positions], classes[test_probability.argmax(axis=1)]):
            writer.writerow([cell_id, prediction])

    configuration = {
        **{key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()},
        "device": "cuda", "device_name": torch.cuda.get_device_name(0), "torch_version": torch.__version__,
        "training_protocol": "three-fold OOF; all disjoint reference cells plus non-held competition cells",
        "neighbor_rule": neighbor_rule,
        "context_rule": context_rule,
    }
    report = {
        "configuration": configuration, "metrics": overall, "fold_metrics": fold_metrics,
        "fold_diagnostics": fold_diagnostics, "runtime_seconds": float(time.time() - started),
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        columns = sorted({key for row in history for key in row})
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(history)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
