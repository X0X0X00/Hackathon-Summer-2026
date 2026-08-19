from __future__ import annotations

import argparse
import inspect
import json
import sys
from functools import wraps
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as torch_functional

import train_confidence_gated_logit_stacker as base


LAST_GRAPH_PENALTY: torch.Tensor | None = None
PENALTY_HISTORY: list[float] = []


def extract_output_dir(arguments: list[str]) -> Path:
    for index, argument in enumerate(arguments):
        if argument == "--output-dir":
            return Path(arguments[index + 1]).resolve()
        if argument.startswith("--output-dir="):
            return Path(argument.split("=", 1)[1]).resolve()
    return Path("outputs/graph_regularized_logit_stacker").resolve()


def patch_stacker_loss(affinity: np.ndarray, graph_lambda: float) -> list[str]:
    global LAST_GRAPH_PENALTY
    graph_cpu = torch.as_tensor(affinity, dtype=torch.float32)
    graph_mass = float(np.sum(affinity))
    if graph_mass <= 0:
        raise ValueError("Class graph contains no positive edges")
    laplacian_cpu = torch.diag(graph_cpu.sum(dim=1)) - graph_cpu

    patched_classes = []
    for name, candidate in vars(base).items():
        if not inspect.isclass(candidate) or candidate.__module__ != base.__name__:
            continue
        if not issubclass(candidate, torch.nn.Module) or "Stacker" not in name:
            continue
        original_forward = candidate.forward

        @wraps(original_forward)
        def graph_forward(self, anchor_log, *args, __forward=original_forward, **kwargs):
            global LAST_GRAPH_PENALTY
            result = __forward(self, anchor_log, *args, **kwargs)
            logits = result[0] if isinstance(result, tuple) else result
            if logits.shape[1] != laplacian_cpu.shape[0]:
                raise ValueError(
                    f"Graph has {laplacian_cpu.shape[0]} classes but logits have {logits.shape[1]}"
                )
            correction = logits - anchor_log
            laplacian = laplacian_cpu.to(
                device=correction.device, dtype=correction.dtype
            )
            energy = torch.sum((correction @ laplacian) * correction)
            LAST_GRAPH_PENALTY = 2.0 * energy / (
                correction.shape[0] * graph_mass
            )
            return result

        candidate.forward = graph_forward
        patched_classes.append(name)

    if not patched_classes:
        raise RuntimeError("No stacker torch module was found in the base trainer")

    original_cross_entropy = torch_functional.cross_entropy

    @wraps(original_cross_entropy)
    def graph_cross_entropy(*args, **kwargs):
        value = original_cross_entropy(*args, **kwargs)
        if LAST_GRAPH_PENALTY is None:
            return value
        if torch.is_grad_enabled():
            PENALTY_HISTORY.append(float(LAST_GRAPH_PENALTY.detach().cpu()))
        return value + graph_lambda * LAST_GRAPH_PENALTY

    torch_functional.cross_entropy = graph_cross_entropy
    if hasattr(base, "F"):
        base.F.cross_entropy = graph_cross_entropy
    return patched_classes


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--class-graph", type=Path, required=True)
    parser.add_argument("--graph-lambda", type=float, default=0.05)
    wrapper_args, base_args = parser.parse_known_args()
    if wrapper_args.graph_lambda < 0:
        raise ValueError("--graph-lambda must be non-negative")

    graph_data = np.load(wrapper_args.class_graph)
    affinity = np.asarray(graph_data["affinity"], dtype=np.float32)
    graph_class_names = graph_data["class_names"].astype(str).tolist()
    patched_classes = patch_stacker_loss(affinity, wrapper_args.graph_lambda)
    output_dir = extract_output_dir(base_args)

    sys.argv = [sys.argv[0], *base_args]
    base.main()

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    metrics["configuration"].update(
        {
            "class_graph": str(wrapper_args.class_graph.resolve()),
            "class_graph_source": "external reference cells only",
            "class_graph_classes": graph_class_names,
            "graph_regularization": (
                "mean weighted squared difference between neighboring-class "
                "logit corrections"
            ),
            "graph_lambda": wrapper_args.graph_lambda,
            "patched_stacker_classes": patched_classes,
        }
    )
    metrics["graph_penalty_audit"] = {
        "n_training_evaluations": len(PENALTY_HISTORY),
        "mean_unweighted_penalty": float(np.mean(PENALTY_HISTORY)),
        "final_unweighted_penalty": float(PENALTY_HISTORY[-1]),
        "mean_weighted_penalty": float(
            wrapper_args.graph_lambda * np.mean(PENALTY_HISTORY)
        ),
    }
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=True, indent=2)
    print(
        json.dumps(
            {
                "graph_lambda": wrapper_args.graph_lambda,
                "patched_stacker_classes": patched_classes,
                "mean_unweighted_graph_penalty": float(np.mean(PENALTY_HISTORY)),
                "metrics_path": str(metrics_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
