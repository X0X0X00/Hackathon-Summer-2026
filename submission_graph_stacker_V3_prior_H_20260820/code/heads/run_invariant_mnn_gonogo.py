from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from run_reliability_piecewise_soft_slot_gonogo import (
    fit_zero_reliability,
    fold_parameters,
    neighbor_jaccard,
    neighbor_metrics,
    piecewise_transform,
    save_rows,
    set_seed,
    zero_expression_values,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = Path(
    r"C:\Users\lizhi\Hackathon-Summer-2026\Hackathon-Summer-2026"
    r"\other_model\Hackathon-Summer-2026\work\cache_ext\gene_token"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unsupervised three-view invariant MNN Go/No-Go audit."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--activation-parameters",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "gene_threshold_activation_ablation"
        / "learned_activation_parameters.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "invariant_mnn_gonogo",
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--encoding-batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--n-neighbors", type=int, default=8)
    parser.add_argument("--alphas", type=str, default="0,0.25,0.5,1,2,4")
    parser.add_argument("--piecewise-reliable-overlap-scale", type=float, default=20.0)
    parser.add_argument("--negative-detection-floor", type=float, default=0.70)
    parser.add_argument("--detectability-smoothing", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def parse_alphas(value: str) -> list[float]:
    result = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    if not result or result[0] != 0.0:
        raise ValueError("alphas must include 0")
    return result


class ThreeViewInvariantEncoder(nn.Module):
    def __init__(self, n_genes: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_genes * 2 + 1, 384),
            nn.GELU(),
            nn.LayerNorm(384),
            nn.Dropout(0.10),
            nn.Linear(384, 192),
            nn.GELU(),
            nn.LayerNorm(192),
            nn.Linear(192, latent_dim),
        )
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, 192),
            nn.GELU(),
            nn.Linear(192, latent_dim),
        )
        self.expression_decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.GELU(),
            nn.Linear(256, n_genes),
        )
        self.detection_decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.GELU(),
            nn.Linear(256, n_genes),
        )

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def forward(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = self.encode(inputs)
        projected = self.projector(latent)
        return (
            latent,
            projected,
            self.expression_decoder(latent),
            self.detection_decoder(latent),
        )


def normalized_expression(
    counts: np.ndarray,
    reference_positions: np.ndarray,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts_float = counts.astype(np.float32, copy=False)
    library = counts_float.sum(axis=1, keepdims=True)
    logged = np.log1p(counts_float / np.maximum(library, 1.0) * 100.0).astype(np.float32)
    if mean is None:
        mean = logged[reference_positions].mean(axis=0, dtype=np.float64).astype(np.float32)
    if std is None:
        std = logged[reference_positions].std(axis=0, dtype=np.float64).astype(np.float32)
    logged -= mean[None, :]
    logged /= np.maximum(std[None, :], 1e-4)
    return logged, mean, std


def make_inputs_numpy(
    expression: np.ndarray,
    detected: np.ndarray,
    depth_mean: float,
    depth_std: float,
) -> np.ndarray:
    depth = detected.sum(axis=1, keepdims=True).astype(np.float32)
    depth = (depth - depth_mean) / max(depth_std, 1e-4)
    return np.concatenate(
        [expression, detected.astype(np.float32), depth], axis=1
    ).astype(np.float32)


def make_inputs_torch(
    counts: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    depth_mean: float,
    depth_std: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    detected = counts > 0
    library = counts.sum(dim=1, keepdim=True)
    expression = torch.log1p(counts / library.clamp_min(1.0) * 100.0)
    expression = (expression - mean) / std.clamp_min(1e-4)
    depth = detected.sum(dim=1, keepdim=True).float()
    depth = (depth - depth_mean) / max(depth_std, 1e-4)
    return torch.cat([expression, detected.float(), depth], dim=1), expression


def off_diagonal_covariance(projected: torch.Tensor) -> torch.Tensor:
    centered = projected - projected.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(len(projected) - 1, 1)
    diagonal = torch.diagonal(covariance)
    return (covariance.square().sum() - diagonal.square().sum()) / projected.shape[1]


def variance_loss(projected: torch.Tensor) -> torch.Tensor:
    std = torch.sqrt(projected.var(dim=0, unbiased=False) + 1e-4)
    return F.relu(1.0 - std).mean()


def train_invariant_encoder(
    raw_counts: np.ndarray,
    clean_expression: np.ndarray,
    clean_detected: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    depth_mean: float,
    depth_std: float,
    args: argparse.Namespace,
) -> tuple[ThreeViewInvariantEncoder, list[dict[str, float | int]]]:
    device = torch.device("cuda")
    model = ThreeViewInvariantEncoder(clean_expression.shape[1], args.latent_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    counts_tensor = torch.from_numpy(np.asarray(raw_counts, dtype=np.float32)).to(device)
    clean_expression_tensor = torch.from_numpy(clean_expression).to(device)
    clean_detected_tensor = torch.from_numpy(clean_detected).to(device)
    mean_tensor = torch.from_numpy(mean).to(device)
    std_tensor = torch.from_numpy(std).to(device)
    indices = torch.arange(len(raw_counts), device=device)
    history: list[dict[str, float | int]] = []
    for epoch in range(args.epochs):
        model.train()
        permutation = indices[torch.randperm(len(indices), device=device)]
        totals = {
            "loss": 0.0,
            "invariance": 0.0,
            "variance": 0.0,
            "covariance": 0.0,
            "expression_reconstruction": 0.0,
            "detection_reconstruction": 0.0,
        }
        seen = 0
        for start in range(0, len(permutation), args.batch_size):
            batch_indices = permutation[start : start + args.batch_size]
            counts = counts_tensor[batch_indices]
            clean_expression_batch = clean_expression_tensor[batch_indices]
            clean_detected_batch = clean_detected_tensor[batch_indices]
            clean_depth = clean_detected_batch.sum(dim=1, keepdim=True).float()
            clean_depth = (clean_depth - depth_mean) / max(depth_std, 1e-4)
            clean_inputs = torch.cat(
                [clean_expression_batch, clean_detected_batch.float(), clean_depth], dim=1
            )
            retention1 = torch.empty(
                (len(counts), 1), device=device, dtype=counts.dtype
            ).uniform_(0.55, 0.85)
            retention2 = torch.empty(
                (len(counts), 1), device=device, dtype=counts.dtype
            ).uniform_(0.35, 0.70)
            counts1 = torch.binomial(counts, retention1.expand_as(counts))
            counts2 = torch.binomial(counts, retention2.expand_as(counts))
            inputs1, _ = make_inputs_torch(
                counts1, mean_tensor, std_tensor, depth_mean, depth_std
            )
            inputs2, _ = make_inputs_torch(
                counts2, mean_tensor, std_tensor, depth_mean, depth_std
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _, projected0, expression0, detection0 = model(clean_inputs)
                _, projected1, expression1, detection1 = model(inputs1)
                _, projected2, expression2, detection2 = model(inputs2)
            projected0 = projected0.float()
            projected1 = projected1.float()
            projected2 = projected2.float()
            invariance = (
                F.mse_loss(projected0, projected1)
                + F.mse_loss(projected0, projected2)
                + F.mse_loss(projected1, projected2)
            ) / 3.0
            variance = (
                variance_loss(projected0)
                + variance_loss(projected1)
                + variance_loss(projected2)
            ) / 3.0
            covariance = (
                off_diagonal_covariance(projected0)
                + off_diagonal_covariance(projected1)
                + off_diagonal_covariance(projected2)
            ) / 3.0
            expression_weight = 0.20 + 0.80 * clean_detected_batch.float()
            expression_reconstruction = (
                (
                    F.smooth_l1_loss(
                        expression0.float(), clean_expression_batch, reduction="none"
                    )
                    + F.smooth_l1_loss(
                        expression1.float(), clean_expression_batch, reduction="none"
                    )
                    + F.smooth_l1_loss(
                        expression2.float(), clean_expression_batch, reduction="none"
                    )
                )
                / 3.0
                * expression_weight
            ).sum() / expression_weight.sum().clamp_min(1)
            positive_weight = torch.tensor(2.0, device=device)
            detection_reconstruction = (
                F.binary_cross_entropy_with_logits(
                    detection0.float(),
                    clean_detected_batch.float(),
                    pos_weight=positive_weight,
                )
                + F.binary_cross_entropy_with_logits(
                    detection1.float(),
                    clean_detected_batch.float(),
                    pos_weight=positive_weight,
                )
                + F.binary_cross_entropy_with_logits(
                    detection2.float(),
                    clean_detected_batch.float(),
                    pos_weight=positive_weight,
                )
            ) / 3.0
            loss = (
                25.0 * invariance
                + 25.0 * variance
                + covariance
                + expression_reconstruction
                + 0.25 * detection_reconstruction
            )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            count = len(batch_indices)
            for key, value in [
                ("loss", loss),
                ("invariance", invariance),
                ("variance", variance),
                ("covariance", covariance),
                ("expression_reconstruction", expression_reconstruction),
                ("detection_reconstruction", detection_reconstruction),
            ]:
                totals[key] += float(value.detach().item()) * count
            seen += count
        row = {"epoch": epoch + 1, **{key: value / seen for key, value in totals.items()}}
        history.append(row)
        print(
            f"epoch={epoch + 1}/{args.epochs} loss={row['loss']:.4f} "
            f"inv={row['invariance']:.4f} var={row['variance']:.4f} "
            f"recon={row['expression_reconstruction']:.4f}",
            flush=True,
        )
    del counts_tensor, clean_expression_tensor, clean_detected_tensor
    torch.cuda.empty_cache()
    return model, history


@torch.inference_mode()
def encode_inputs(
    model: ThreeViewInvariantEncoder,
    inputs: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    output: list[np.ndarray] = []
    for start in range(0, len(inputs), batch_size):
        batch = torch.from_numpy(inputs[start : start + batch_size]).to(device)
        latent = model.encode(batch)
        output.append(F.normalize(latent.float(), dim=1).cpu().numpy())
    return np.concatenate(output)


def binomial_thin_counts(
    raw_counts: np.ndarray, retention: float, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    output = np.empty_like(raw_counts)
    for start in range(0, len(raw_counts), 4096):
        end = min(start + 4096, len(raw_counts))
        output[start:end] = rng.binomial(
            np.asarray(raw_counts[start:end], dtype=np.int64), retention
        ).astype(raw_counts.dtype)
    return output


@torch.inference_mode()
def segment_score_search(
    base_feature: np.ndarray,
    residual_feature: np.ndarray | None,
    reliable: np.ndarray | None,
    segment_codes: np.ndarray,
    reference_positions: np.ndarray,
    train_positions: np.ndarray,
    alphas: list[float],
    n_neighbors: int,
    reliable_overlap_scale: float,
) -> dict[float, dict[str, np.ndarray]]:
    device = torch.device("cuda")
    outputs = {
        alpha: {
            "knn": np.full((len(train_positions), n_neighbors), -1, dtype=np.int32),
            "mnn": np.full((len(train_positions), n_neighbors), -1, dtype=np.int32),
        }
        for alpha in alphas
    }
    train_segments = segment_codes[train_positions]
    reference_segments = segment_codes[reference_positions]
    for segment in np.unique(train_segments):
        query_local = np.flatnonzero(train_segments == segment)
        reference_local = np.flatnonzero(reference_segments == segment)
        if not len(query_local) or not len(reference_local):
            continue
        query_global = train_positions[query_local]
        reference_global = reference_positions[reference_local]
        base_query = F.normalize(torch.from_numpy(base_feature[query_global]).to(device), dim=1)
        base_reference = F.normalize(torch.from_numpy(base_feature[reference_global]).to(device), dim=1)
        base_score = base_query @ base_reference.T
        residual_score: torch.Tensor | None = None
        reliability_weight: torch.Tensor | None = None
        if residual_feature is not None:
            residual_query = F.normalize(
                torch.from_numpy(residual_feature[query_global]).to(device), dim=1
            )
            residual_reference = F.normalize(
                torch.from_numpy(residual_feature[reference_global]).to(device), dim=1
            )
            residual_score = residual_query @ residual_reference.T
            if reliable is not None:
                query_reliable = torch.from_numpy(
                    reliable[query_global].astype(np.float32)
                ).to(device)
                reference_reliable = torch.from_numpy(
                    reliable[reference_global].astype(np.float32)
                ).to(device)
                overlap = query_reliable @ reference_reliable.T
                query_count = query_reliable.sum(dim=1, keepdim=True)
                reference_count = reference_reliable.sum(dim=1, keepdim=True).T
                overlap_fraction = overlap / torch.sqrt(
                    (query_count * reference_count).clamp_min(1.0)
                )
                amount = torch.clamp(overlap / reliable_overlap_scale, 0.0, 1.0)
                reliability_weight = overlap_fraction * amount
            else:
                reliability_weight = torch.ones_like(residual_score)
        for alpha in alphas:
            score = (
                base_score
                if alpha == 0 or residual_score is None
                else base_score + alpha * reliability_weight * residual_score
            )
            k_reference = min(n_neighbors, len(reference_global))
            forward = torch.topk(score, k=k_reference, dim=1, largest=True, sorted=True).indices
            forward_np = forward.cpu().numpy()
            for local_query, local_references in enumerate(forward_np):
                outputs[alpha]["knn"][query_local[local_query], :k_reference] = (
                    reference_global[local_references]
                )
            k_train = min(n_neighbors, len(query_global))
            reverse = torch.topk(
                score.T, k=k_train, dim=1, largest=True, sorted=False
            ).indices.cpu().numpy()
            for local_query, local_references in enumerate(forward_np):
                write = 0
                for local_reference in local_references:
                    if np.any(reverse[local_reference] == local_query):
                        outputs[alpha]["mnn"][query_local[local_query], write] = (
                            reference_global[local_reference]
                        )
                        write += 1
                        if write == n_neighbors:
                            break
        del base_score
    return outputs


def evaluate_variant(
    neighbors: dict[str, np.ndarray],
    labels: np.ndarray,
    train_positions: np.ndarray,
    depth: np.ndarray,
    class_names: np.ndarray,
) -> tuple[dict[str, float | int], list[dict[str, object]]]:
    return neighbor_metrics(
        neighbors["mnn"], labels, train_positions, depth, class_names
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is disabled")
    set_seed(args.seed)
    started = time.time()
    alphas = parse_alphas(args.alphas)
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={torch.cuda.get_device_name(0)} alphas={alphas}", flush=True)

    cache_expression = np.array(
        np.load(
            data_dir / "mnn_residual" / "expression.npy",
            mmap_mode="r",
            allow_pickle=False,
        ),
        dtype=np.float32,
        copy=True,
    )
    raw_counts_mmap = np.load(data_dir / "raw_counts.npy", mmap_mode="r", allow_pickle=False)
    raw_counts = np.array(raw_counts_mmap, copy=True)
    detected = raw_counts > 0
    depth = detected.sum(axis=1).astype(np.int16)
    labels = np.load(data_dir / "labels.npy", allow_pickle=False).astype(np.int64)
    folds = np.load(data_dir / "folds.npy", allow_pickle=False).astype(np.int16)
    gene_names = np.load(data_dir / "gene_names.npy", allow_pickle=False).astype(str)
    class_names = np.load(data_dir / "class_names.npy", allow_pickle=False).astype(str)
    reference_positions = np.load(data_dir / "reference_positions.npy", allow_pickle=False)
    train_positions = np.load(data_dir / "train_positions.npy", allow_pickle=False)
    segment_codes = np.load(
        data_dir / "mnn_residual" / "segment_codes.npy", allow_pickle=False
    )
    parameters = pd.read_csv(args.activation_parameters.resolve())
    clean_expression, expression_mean, expression_std = normalized_expression(
        raw_counts, reference_positions
    )
    reference_depth = depth[reference_positions].astype(np.float32)
    depth_mean = float(reference_depth.mean())
    depth_std = float(reference_depth.std())
    clean_inputs = make_inputs_numpy(
        clean_expression, detected, depth_mean, depth_std
    )
    print(
        f"cells={len(raw_counts)} genes={raw_counts.shape[1]} "
        f"mean_detected={depth.mean():.1f}",
        flush=True,
    )

    model, history = train_invariant_encoder(
        raw_counts,
        clean_expression,
        detected,
        expression_mean,
        expression_std,
        depth_mean,
        depth_std,
        args,
    )
    invariant_clean = encode_inputs(model, clean_inputs, args.encoding_batch_size)
    thinned_counts_30 = binomial_thin_counts(raw_counts, 0.70, args.seed + 3000)
    thinned_counts_50 = binomial_thin_counts(raw_counts, 0.50, args.seed + 5000)
    thin30_expression, _, _ = normalized_expression(
        thinned_counts_30,
        reference_positions,
        expression_mean,
        expression_std,
    )
    thin50_expression, _, _ = normalized_expression(
        thinned_counts_50,
        reference_positions,
        expression_mean,
        expression_std,
    )
    detected30 = thinned_counts_30 > 0
    detected50 = thinned_counts_50 > 0
    inputs30 = make_inputs_numpy(thin30_expression, detected30, depth_mean, depth_std)
    inputs50 = make_inputs_numpy(thin50_expression, detected50, depth_mean, depth_std)
    invariant30 = encode_inputs(model, inputs30, args.encoding_batch_size)
    invariant50 = encode_inputs(model, inputs50, args.encoding_batch_size)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "gene_names": gene_names.tolist(),
            "expression_mean": expression_mean,
            "expression_std": expression_std,
            "depth_mean": depth_mean,
            "depth_std": depth_std,
            "training_labels_used": False,
        },
        output_dir / "three_view_invariant_encoder.pt",
    )
    del model, clean_inputs, inputs30, inputs50
    torch.cuda.empty_cache()

    zero_values = zero_expression_values(
        cache_expression, detected, reference_positions
    )
    cache30 = cache_expression.copy()
    cache50 = cache_expression.copy()
    dropped30 = detected & ~detected30
    dropped50 = detected & ~detected50
    for gene in range(cache_expression.shape[1]):
        cache30[dropped30[:, gene], gene] = zero_values[gene]
        cache50[dropped50[:, gene], gene] = zero_values[gene]

    raw_clean = segment_score_search(
        cache_expression,
        None,
        None,
        segment_codes,
        reference_positions,
        train_positions,
        [0.0],
        args.n_neighbors,
        args.piecewise_reliable_overlap_scale,
    )[0.0]
    raw30 = segment_score_search(
        cache30,
        None,
        None,
        segment_codes,
        reference_positions,
        train_positions,
        [0.0],
        args.n_neighbors,
        args.piecewise_reliable_overlap_scale,
    )[0.0]
    raw50 = segment_score_search(
        cache50,
        None,
        None,
        segment_codes,
        reference_positions,
        train_positions,
        [0.0],
        args.n_neighbors,
        args.piecewise_reliable_overlap_scale,
    )[0.0]
    invariant_clean_neighbors = segment_score_search(
        invariant_clean,
        None,
        None,
        segment_codes,
        reference_positions,
        train_positions,
        [0.0],
        args.n_neighbors,
        args.piecewise_reliable_overlap_scale,
    )[0.0]
    invariant30_neighbors = segment_score_search(
        invariant30,
        None,
        None,
        segment_codes,
        reference_positions,
        train_positions,
        [0.0],
        args.n_neighbors,
        args.piecewise_reliable_overlap_scale,
    )[0.0]
    invariant50_neighbors = segment_score_search(
        invariant50,
        None,
        None,
        segment_codes,
        reference_positions,
        train_positions,
        [0.0],
        args.n_neighbors,
        args.piecewise_reliable_overlap_scale,
    )[0.0]

    zero_reliability, reliable, reliability_diagnostics = fit_zero_reliability(
        detected,
        depth,
        segment_codes,
        reference_positions,
        args.negative_detection_floor,
        args.detectability_smoothing,
    )
    reliable30 = reliable.copy()
    reliable30[dropped30] = False
    reliable50 = reliable.copy()
    reliable50[dropped50] = False
    candidate = {
        alpha: {
            key: np.full((len(train_positions), args.n_neighbors), -1, dtype=np.int32)
            for key in ["knn", "mnn", "thin30_knn", "thin30_mnn", "thin50_knn", "thin50_mnn"]
        }
        for alpha in alphas
    }
    piecewise_only = {
        key: np.full((len(train_positions), args.n_neighbors), -1, dtype=np.int32)
        for key in ["knn", "mnn", "thin30_knn", "thin30_mnn", "thin50_knn", "thin50_mnn"]
    }
    for fold in [0, 1, 2]:
        print(f"piecewise fold={fold}", flush=True)
        fold_params = fold_parameters(parameters, gene_names, fold)
        piece = piecewise_transform(
            cache_expression, detected, fold_params, reference_positions
        )
        piece30 = piecewise_transform(
            cache30, detected30, fold_params, reference_positions
        )
        piece50 = piecewise_transform(
            cache50, detected50, fold_params, reference_positions
        )
        clean_search = segment_score_search(
            invariant_clean,
            piece,
            reliable,
            segment_codes,
            reference_positions,
            train_positions,
            alphas,
            args.n_neighbors,
            args.piecewise_reliable_overlap_scale,
        )
        search30 = segment_score_search(
            invariant30,
            piece30,
            reliable30,
            segment_codes,
            reference_positions,
            train_positions,
            alphas,
            args.n_neighbors,
            args.piecewise_reliable_overlap_scale,
        )
        search50 = segment_score_search(
            invariant50,
            piece50,
            reliable50,
            segment_codes,
            reference_positions,
            train_positions,
            alphas,
            args.n_neighbors,
            args.piecewise_reliable_overlap_scale,
        )
        piece_clean_search = segment_score_search(
            piece,
            None,
            None,
            segment_codes,
            reference_positions,
            train_positions,
            [0.0],
            args.n_neighbors,
            args.piecewise_reliable_overlap_scale,
        )[0.0]
        piece30_search = segment_score_search(
            piece30,
            None,
            None,
            segment_codes,
            reference_positions,
            train_positions,
            [0.0],
            args.n_neighbors,
            args.piecewise_reliable_overlap_scale,
        )[0.0]
        piece50_search = segment_score_search(
            piece50,
            None,
            None,
            segment_codes,
            reference_positions,
            train_positions,
            [0.0],
            args.n_neighbors,
            args.piecewise_reliable_overlap_scale,
        )[0.0]
        held = folds[train_positions] == fold
        for alpha in alphas:
            for key in ["knn", "mnn"]:
                candidate[alpha][key][held] = clean_search[alpha][key][held]
                candidate[alpha][f"thin30_{key}"][held] = search30[alpha][key][held]
                candidate[alpha][f"thin50_{key}"][held] = search50[alpha][key][held]
        for key in ["knn", "mnn"]:
            piecewise_only[key][held] = piece_clean_search[key][held]
            piecewise_only[f"thin30_{key}"][held] = piece30_search[key][held]
            piecewise_only[f"thin50_{key}"][held] = piece50_search[key][held]
        del piece, piece30, piece50, clean_search, search30, search50

    alpha_rows: list[dict[str, object]] = []
    for alpha in alphas:
        metrics, _ = evaluate_variant(
            candidate[alpha], labels, train_positions, depth, class_names
        )
        alpha_rows.append(
            {
                "alpha": alpha,
                **metrics,
                "mnn_jaccard_30pct_thinning": neighbor_jaccard(
                    candidate[alpha]["mnn"], candidate[alpha]["thin30_mnn"]
                ),
                "mnn_jaccard_50pct_thinning": neighbor_jaccard(
                    candidate[alpha]["mnn"], candidate[alpha]["thin50_mnn"]
                ),
            }
        )

    crossfit = {
        key: np.full((len(train_positions), args.n_neighbors), -1, dtype=np.int32)
        for key in ["knn", "mnn", "thin30_knn", "thin30_mnn", "thin50_knn", "thin50_mnn"]
    }
    fold_selection: list[dict[str, object]] = []
    for held_fold in [0, 1, 2]:
        tuning = folds[train_positions] != held_fold
        held = folds[train_positions] == held_fold
        tuning_rows = []
        for alpha in alphas:
            metrics, _ = neighbor_metrics(
                candidate[alpha]["mnn"][tuning],
                labels,
                train_positions[tuning],
                depth,
                class_names,
            )
            stability = neighbor_jaccard(
                candidate[alpha]["mnn"][tuning],
                candidate[alpha]["thin30_mnn"][tuning],
            )
            tuning_rows.append((alpha, metrics, stability))
        selected_alpha, tuning_metrics, tuning_stability = max(
            tuning_rows,
            key=lambda row: (
                row[1]["class_balanced_edge_purity"],
                row[2],
                row[1]["same_label_edge_rate"],
                -row[0],
            ),
        )
        for key in crossfit:
            crossfit[key][held] = candidate[selected_alpha][key][held]
        held_metrics, _ = neighbor_metrics(
            candidate[selected_alpha]["mnn"][held],
            labels,
            train_positions[held],
            depth,
            class_names,
        )
        fold_selection.append(
            {
                "held_fold": held_fold,
                "selected_alpha": selected_alpha,
                "tuning_class_balanced_purity": tuning_metrics[
                    "class_balanced_edge_purity"
                ],
                "tuning_jaccard_30pct": tuning_stability,
                "held_class_balanced_purity": held_metrics[
                    "class_balanced_edge_purity"
                ],
                "held_same_label_rate": held_metrics["same_label_edge_rate"],
                "held_coverage": held_metrics["cell_coverage"],
            }
        )

    variants = {
        "raw_expression_mnn": {
            **raw_clean,
            "thin30_mnn": raw30["mnn"],
            "thin50_mnn": raw50["mnn"],
        },
        "piecewise_mnn": piecewise_only,
        "invariant_mnn": {
            **invariant_clean_neighbors,
            "thin30_mnn": invariant30_neighbors["mnn"],
            "thin50_mnn": invariant50_neighbors["mnn"],
        },
        "invariant_piecewise_crossfit_mnn": crossfit,
    }
    comparison_rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    report_variants: dict[str, object] = {}
    for name, arrays in variants.items():
        metrics, per_class = neighbor_metrics(
            arrays["mnn"], labels, train_positions, depth, class_names
        )
        stability30 = neighbor_jaccard(arrays["mnn"], arrays["thin30_mnn"])
        stability50 = neighbor_jaccard(arrays["mnn"], arrays["thin50_mnn"])
        row = {
            "variant": name,
            **metrics,
            "mnn_jaccard_30pct_thinning": stability30,
            "mnn_jaccard_50pct_thinning": stability50,
        }
        comparison_rows.append(row)
        report_variants[name] = row
        class_rows.extend({"variant": name, **item} for item in per_class)

    candidate_metrics = report_variants["invariant_piecewise_crossfit_mnn"]
    criteria = [
        {
            "criterion": "30pct_thinning_jaccard",
            "value": candidate_metrics["mnn_jaccard_30pct_thinning"],
            "threshold": 0.25,
            "passed": candidate_metrics["mnn_jaccard_30pct_thinning"] >= 0.25,
        },
        {
            "criterion": "same_label_edge_rate",
            "value": candidate_metrics["same_label_edge_rate"],
            "threshold": 0.32,
            "passed": candidate_metrics["same_label_edge_rate"] >= 0.32,
        },
        {
            "criterion": "class_balanced_purity",
            "value": candidate_metrics["class_balanced_edge_purity"],
            "threshold": 0.254612,
            "passed": candidate_metrics["class_balanced_edge_purity"] >= 0.254612,
        },
        {
            "criterion": "absolute_depth_purity_gap",
            "value": abs(candidate_metrics["low_minus_high_depth_purity"]),
            "threshold": 0.05,
            "passed": abs(candidate_metrics["low_minus_high_depth_purity"]) <= 0.05,
        },
        {
            "criterion": "oligo2_progenitor_cross_rate",
            "value": candidate_metrics[
                "oligodendrocyte_2_to_progenitor_edge_rate"
            ],
            "threshold": 0.16238038277511962,
            "passed": candidate_metrics[
                "oligodendrocyte_2_to_progenitor_edge_rate"
            ]
            <= 0.16238038277511962,
        },
        {
            "criterion": "coverage",
            "value": candidate_metrics["cell_coverage"],
            "threshold": 0.95,
            "passed": candidate_metrics["cell_coverage"] >= 0.95,
        },
    ]
    passed_count = sum(bool(item["passed"]) for item in criteria)
    mandatory = criteria[0]["passed"] and criteria[2]["passed"] and criteria[4]["passed"]
    decision = "GO" if mandatory and passed_count >= 5 else "NO-GO"

    for name, arrays in variants.items():
        for key, values in arrays.items():
            if isinstance(values, np.ndarray):
                np.save(output_dir / f"{name}_{key}.npy", values, allow_pickle=False)
    save_rows(output_dir / "training_history.csv", history)
    save_rows(output_dir / "variant_comparison.csv", comparison_rows)
    save_rows(output_dir / "class_metrics.csv", class_rows)
    save_rows(output_dir / "alpha_audit.csv", alpha_rows)
    save_rows(output_dir / "fold_selection.csv", fold_selection)
    save_rows(output_dir / "go_no_go_criteria.csv", criteria)
    report = {
        "configuration": {
            **{
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "training_labels_used_for_invariant_encoder": False,
            "old_checkpoint_reused": False,
            "old_checkpoint_reason": "old model is a supervised fused classifier coupled to current-cell auxiliary and existing neighbors, not a portable self-only embedding",
            "invariant_training": "clean plus two binomial-thinned views; VICReg-style invariance/variance/covariance and clean-expression/detection reconstruction",
            "neighbor_constraint": "same Segment exact reciprocal top-8",
            "piecewise_alpha_selection": "cross-fit on the other two OOF folds",
        },
        "reliability_diagnostics": reliability_diagnostics,
        "variants": report_variants,
        "alpha_audit": alpha_rows,
        "fold_selection": fold_selection,
        "go_no_go": {
            "decision": decision,
            "passed_count": passed_count,
            "required": "at least 5/6 and mandatory thinning stability, balanced purity, oligodendrocyte maturity protection",
            "criteria": criteria,
        },
        "runtime_seconds": float(time.time() - started),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nVARIANT COMPARISON", flush=True)
    print(pd.DataFrame(comparison_rows).to_string(index=False), flush=True)
    print("\nFOLD SELECTION", flush=True)
    print(pd.DataFrame(fold_selection).to_string(index=False), flush=True)
    print("\nGO/NO-GO", flush=True)
    print(json.dumps(report["go_no_go"], ensure_ascii=False, indent=2), flush=True)
    print(f"runtime_seconds={report['runtime_seconds']:.1f}", flush=True)


if __name__ == "__main__":
    main()
