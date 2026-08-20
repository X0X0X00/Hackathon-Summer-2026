from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = Path(
    r"C:\Users\lizhi\Hackathon-Summer-2026\Hackathon-Summer-2026"
    r"\other_model\Hackathon-Summer-2026\work\cache_ext\gene_token"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "external_gene_token_encoder"
TARGET_COLUMN = "MERFISH_cell_type_annotation"


class GeneTokenEncoder(nn.Module):
    def __init__(
        self,
        *,
        n_genes: int,
        auxiliary_dim: int,
        n_classes: int,
        latent_dim: int = 96,
        token_dim: int = 48,
        token_heads: int = 4,
        token_layers: int = 2,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.n_genes = n_genes
        self.gene_embedding = nn.Embedding(n_genes, token_dim)
        self.value_projection = nn.Sequential(
            nn.Linear(2, token_dim), nn.GELU(), nn.Linear(token_dim, token_dim)
        )
        self.quality_projection = nn.Sequential(
            nn.Linear(2, token_dim), nn.GELU(), nn.Linear(token_dim, token_dim)
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, token_dim))
        nn.init.normal_(self.cls_token, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=token_heads,
            dim_feedforward=2 * token_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=token_layers)
        self.token_to_latent = nn.Sequential(
            nn.LayerNorm(token_dim), nn.Linear(token_dim, latent_dim), nn.GELU()
        )
        self.auxiliary_to_latent = nn.Sequential(
            nn.Linear(auxiliary_dim, latent_dim), nn.GELU()
        )
        self.latent_norm = nn.LayerNorm(latent_dim)
        self.head = nn.Sequential(
            nn.Linear(latent_dim, 160),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(160, n_classes),
        )

    def forward(self, raw_counts: torch.Tensor, auxiliary: torch.Tensor) -> torch.Tensor:
        detected = (raw_counts > 0).sum(dim=1)
        max_detected = max(int(detected.max().item()), 1)
        values, gene_indices = torch.topk(raw_counts, k=max_detected, dim=1)
        padding = values <= 0
        totals = raw_counts.sum(dim=1, keepdim=True).clamp_min(1.0)
        normalized_values = torch.log1p(values / totals * 100.0)
        value_features = torch.stack(
            [torch.log1p(values), normalized_values], dim=-1
        )
        tokens = self.gene_embedding(gene_indices) + self.value_projection(
            value_features
        )
        quality = torch.stack(
            [torch.log1p(totals[:, 0]), detected.float() / float(self.n_genes)],
            dim=1,
        )
        cls = self.cls_token.expand(len(raw_counts), -1, -1) + self.quality_projection(
            quality
        ).unsqueeze(1)
        sequence = torch.cat([cls, tokens], dim=1)
        cls_padding = torch.zeros(
            (len(raw_counts), 1), dtype=torch.bool, device=raw_counts.device
        )
        encoded = self.transformer(
            sequence,
            src_key_padding_mask=torch.cat([cls_padding, padding], dim=1),
        )[:, 0]
        latent = self.latent_norm(
            self.token_to_latent(encoded) + self.auxiliary_to_latent(auxiliary)
        )
        return self.head(latent)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GPU gene-token encoder on the extended MERFISH reference universe."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--prediction-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--token-dim", type=int, default=48)
    parser.add_argument("--token-heads", type=int, default=4)
    parser.add_argument("--token-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--corrupt-ce-weight", type=float, default=0.5)
    parser.add_argument("--consistency-weight", type=float, default=0.2)
    parser.add_argument("--minimum-retention", type=float, default=0.60)
    parser.add_argument("--maximum-retention", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    prediction = probabilities.argmax(axis=1)
    n_classes = probabilities.shape[1]
    true_count = np.bincount(labels, minlength=n_classes).astype(np.float64)
    predicted_count = np.bincount(prediction, minlength=n_classes).astype(np.float64)
    true_positive = np.bincount(
        labels[prediction == labels], minlength=n_classes
    ).astype(np.float64)
    denominator = true_count + predicted_count
    class_f1 = np.divide(
        2.0 * true_positive,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    clipped = np.clip(probabilities, 1e-12, 1.0)
    return {
        "accuracy": float(np.mean(prediction == labels)),
        "macro_f1": float(class_f1.mean()),
        "weighted_f1": float(np.sum(class_f1 * true_count) / true_count.sum()),
        "log_loss": float(-np.log(clipped[np.arange(len(labels)), labels]).mean()),
    }


def standardize_auxiliary(
    values: np.ndarray, mean: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    result = (np.asarray(values, dtype=np.float32) - mean) / scale
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32, copy=False
    )


def batch_tensors(
    raw_counts: np.ndarray,
    auxiliary: np.ndarray,
    indices: np.ndarray,
    auxiliary_mean: np.ndarray,
    auxiliary_scale: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    raw = torch.as_tensor(
        np.asarray(raw_counts[indices], dtype=np.float32), dtype=torch.float32
    ).pin_memory().to(device, non_blocking=True)
    aux_values = standardize_auxiliary(
        auxiliary[indices], auxiliary_mean, auxiliary_scale
    )
    aux = torch.as_tensor(aux_values, dtype=torch.float32).pin_memory().to(
        device, non_blocking=True
    )
    return raw, aux


@torch.no_grad()
def predict(
    model: GeneTokenEncoder,
    raw_counts: np.ndarray,
    auxiliary: np.ndarray,
    indices: np.ndarray,
    auxiliary_mean: np.ndarray,
    auxiliary_scale: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    output: list[np.ndarray] = []
    for start in range(0, len(indices), batch_size):
        selected = indices[start : start + batch_size]
        raw, aux = batch_tensors(
            raw_counts,
            auxiliary,
            selected,
            auxiliary_mean,
            auxiliary_scale,
            device,
        )
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(raw, aux)
        output.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    return np.concatenate(output)


def train_fold(
    *,
    fold: int,
    model: GeneTokenEncoder,
    raw_counts: np.ndarray,
    auxiliary: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    auxiliary_mean: np.ndarray,
    auxiliary_scale: np.ndarray,
    device: torch.device,
    args: argparse.Namespace,
) -> list[dict[str, float | int]]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    steps_per_epoch = math.ceil(len(train_indices) / args.batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        total_steps=args.epochs * steps_per_epoch,
        pct_start=0.1,
    )
    scaler = torch.amp.GradScaler("cuda")
    rng = np.random.default_rng(args.seed + fold * 1000)
    history: list[dict[str, float | int]] = []
    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(train_indices)
        total_loss = 0.0
        total_clean = 0.0
        total_corrupt = 0.0
        total_consistency = 0.0
        seen = 0
        started = time.time()
        for start in range(0, len(order), args.batch_size):
            selected = order[start : start + args.batch_size]
            raw, aux = batch_tensors(
                raw_counts,
                auxiliary,
                selected,
                auxiliary_mean,
                auxiliary_scale,
                device,
            )
            target = torch.as_tensor(
                labels[selected], dtype=torch.long, device=device
            )
            retention = torch.empty(
                (len(raw), 1), dtype=torch.float32, device=device
            ).uniform_(args.minimum_retention, args.maximum_retention)
            corrupt = torch.binomial(raw, retention.expand_as(raw))
            combined_raw = torch.cat([raw, corrupt], dim=0)
            combined_aux = torch.cat([aux, aux], dim=0)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                combined_logits = model(combined_raw, combined_aux)
                clean_logits, corrupt_logits = combined_logits.chunk(2, dim=0)
                clean_ce = F.cross_entropy(clean_logits, target)
                corrupt_ce = F.cross_entropy(corrupt_logits, target)
                teacher = torch.softmax(clean_logits.detach().float(), dim=1)
                consistency = F.kl_div(
                    F.log_softmax(corrupt_logits.float(), dim=1),
                    teacher,
                    reduction="batchmean",
                )
                loss = (
                    clean_ce
                    + args.corrupt_ce_weight * corrupt_ce
                    + args.consistency_weight * consistency
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            count = len(selected)
            seen += count
            total_loss += float(loss.item()) * count
            total_clean += float(clean_ce.item()) * count
            total_corrupt += float(corrupt_ce.item()) * count
            total_consistency += float(consistency.item()) * count
        row = {
            "fold": fold,
            "epoch": epoch + 1,
            "loss": total_loss / seen,
            "clean_ce": total_clean / seen,
            "corrupt_ce": total_corrupt / seen,
            "consistency": total_consistency / seen,
            "seconds": time.time() - started,
        }
        history.append(row)
        print(
            f"fold={fold} epoch={epoch + 1}/{args.epochs} "
            f"loss={row['loss']:.4f} clean_ce={row['clean_ce']:.4f} "
            f"seconds={row['seconds']:.1f}",
            flush=True,
        )
    return history


def probability_frame(
    ids: np.ndarray, class_names: np.ndarray, probabilities: np.ndarray
) -> pd.DataFrame:
    frame = pd.DataFrame(
        probabilities, columns=[f"p__{name}" for name in class_names]
    )
    frame.insert(0, "Cell_ID", ids.astype(str))
    return frame


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for external gene-token training")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    data_dir = args.data_dir.resolve()
    raw_counts = np.load(data_dir / "raw_counts.npy", mmap_mode="r")
    auxiliary = np.load(data_dir / "auxiliary.npy", mmap_mode="r")
    ids = np.load(data_dir / "ids.npy").astype(str)
    labels = np.load(data_dir / "labels.npy").astype(np.int64)
    class_names = np.load(data_dir / "class_names.npy").astype(str)
    gene_names = np.load(data_dir / "gene_names.npy").astype(str)
    reference_positions = np.load(data_dir / "reference_positions.npy").astype(np.int64)
    train_positions = np.load(data_dir / "train_positions.npy").astype(np.int64)
    test_positions = np.load(data_dir / "test_positions.npy").astype(np.int64)
    folds = np.load(data_dir / "folds.npy").astype(np.int64)
    auxiliary_reference = np.asarray(auxiliary[reference_positions], dtype=np.float32)
    auxiliary_mean = np.nanmean(auxiliary_reference, axis=0).astype(np.float32)
    auxiliary_scale = np.nanstd(auxiliary_reference, axis=0).astype(np.float32)
    auxiliary_scale[auxiliary_scale < 1e-6] = 1.0
    del auxiliary_reference

    n_classes = len(class_names)
    oof = np.zeros((len(train_positions), n_classes), dtype=np.float32)
    test_probabilities: list[np.ndarray] = []
    history: list[dict[str, float | int]] = []
    fold_rows: list[dict[str, float | int]] = []
    train_row_by_universe = pd.Series(
        np.arange(len(train_positions), dtype=np.int64), index=train_positions
    )
    started = time.time()
    print(
        f"device={torch.cuda.get_device_name(0)} reference={len(reference_positions)} "
        f"train={len(train_positions)} test={len(test_positions)} "
        f"genes={raw_counts.shape[1]} auxiliary={auxiliary.shape[1]}",
        flush=True,
    )
    for fold in sorted(np.unique(folds[train_positions])):
        fold = int(fold)
        visible_competition = train_positions[folds[train_positions] != fold]
        fit_indices = np.concatenate([reference_positions, visible_competition])
        valid_indices = train_positions[folds[train_positions] == fold]
        model = GeneTokenEncoder(
            n_genes=raw_counts.shape[1],
            auxiliary_dim=auxiliary.shape[1],
            n_classes=n_classes,
            token_dim=args.token_dim,
            token_heads=args.token_heads,
            token_layers=args.token_layers,
            dropout=args.dropout,
        ).to(device)
        history.extend(
            train_fold(
                fold=fold,
                model=model,
                raw_counts=raw_counts,
                auxiliary=auxiliary,
                labels=labels,
                train_indices=fit_indices,
                auxiliary_mean=auxiliary_mean,
                auxiliary_scale=auxiliary_scale,
                device=device,
                args=args,
            )
        )
        valid_probability = predict(
            model,
            raw_counts,
            auxiliary,
            valid_indices,
            auxiliary_mean,
            auxiliary_scale,
            device,
            args.prediction_batch_size,
        )
        test_probability = predict(
            model,
            raw_counts,
            auxiliary,
            test_positions,
            auxiliary_mean,
            auxiliary_scale,
            device,
            args.prediction_batch_size,
        )
        valid_rows = train_row_by_universe.loc[valid_indices].to_numpy(dtype=np.int64)
        oof[valid_rows] = valid_probability
        test_probabilities.append(test_probability)
        fold_metrics = classification_metrics(labels[valid_indices], valid_probability)
        fold_rows.append({"fold": fold, "n_valid": len(valid_indices), **fold_metrics})
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "fold": fold,
                "class_names": class_names.tolist(),
                "gene_names": gene_names.tolist(),
                "auxiliary_mean": auxiliary_mean,
                "auxiliary_scale": auxiliary_scale,
                "configuration": vars(args),
            },
            checkpoint_dir / f"gene_token_fold{fold}.pt",
        )
        print(
            f"fold={fold} accuracy={fold_metrics['accuracy']:.4f} "
            f"macro_f1={fold_metrics['macro_f1']:.4f} "
            f"log_loss={fold_metrics['log_loss']:.4f}",
            flush=True,
        )
        del model
        torch.cuda.empty_cache()

    test = np.mean(np.stack(test_probabilities), axis=0)
    train_ids = ids[train_positions]
    test_ids = ids[test_positions]
    overall = classification_metrics(labels[train_positions], oof)
    probability_frame(train_ids, class_names, oof).to_csv(
        output_dir / "oof_probabilities_gene_token_encoder.csv", index=False
    )
    probability_frame(test_ids, class_names, test).to_csv(
        output_dir / "test_probabilities_gene_token_encoder.csv", index=False
    )
    pd.DataFrame(
        {
            "Cell_ID": test_ids,
            TARGET_COLUMN: class_names[test.argmax(axis=1)],
        }
    ).to_csv(output_dir / "submission_gene_token_encoder.csv", index=False)
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output_dir / "fold_metrics.csv", index=False)
    report = {
        "configuration": {
            **{
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "device": str(device),
            "device_name": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "training_protocol": (
                "three-fold OOF; each fold trains on all disjoint reference cells plus "
                "competition train cells outside the held-out fold"
            ),
        },
        "metrics": overall,
        "fold_metrics": fold_rows,
        "runtime_seconds": time.time() - started,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    print(f"saved={output_dir}", flush=True)


if __name__ == "__main__":
    main()
