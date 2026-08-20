from __future__ import annotations

import os
import sys
from pathlib import Path


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(
    os.environ.get("HACKATHON_PROJECT_ROOT", r"C:\Users\lizhi\Documents\ChatGPT\hackathon")
).resolve()
SOURCE_DIR = PROJECT_ROOT / "src"
INPUT_DIR = BUNDLE_ROOT / "model" / "inputs"

if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

import train_confidence_gated_logit_stacker as base


base.MODEL_SOURCES = {
    "anchor_external_primary": INPUT_DIR,
    "graph_stacker_v2": INPUT_DIR,
    "current_anchor_prior": INPUT_DIR,
    "gene_token": INPUT_DIR,
    "strict_h": INPUT_DIR,
}
base.PROBABILITY_STEMS = {
    "anchor_external_primary": "prior_h_anchor",
    "graph_stacker_v2": "graph_stacker_v2",
    "current_anchor_prior": "current_anchor_prior",
    "gene_token": "gene_token",
    "strict_h": "strict_h",
}
base.EXPERT_NAMES = [
    "graph_stacker_v2",
    "current_anchor_prior",
    "gene_token",
    "strict_h",
]

import train_graph_regularized_logit_stacker as graph_stacker


if __name__ == "__main__":
    graph_stacker.main()
