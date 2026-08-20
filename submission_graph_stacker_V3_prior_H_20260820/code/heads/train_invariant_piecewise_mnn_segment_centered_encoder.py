from pathlib import Path

import train_piecewise_mnn_segment_centered_encoder as base

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "invariant_piecewise_mnn_segment_centered"

base.OUTPUT_DIR = OUTPUT_DIR
base.GRAPH_ROOT = OUTPUT_DIR / "graphs"
base.SHARED_DIR = OUTPUT_DIR / "graph_shared"

if __name__ == "__main__":
    base.main()
