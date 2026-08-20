"""Validate the official MERFISH-60 data contract. Exit nonzero on failure."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from merfish60.io import DataContractError, load_dataset, validate_contract  # noqa: E402


def main() -> int:
    try:
        data = load_dataset(ROOT)
        messages = validate_contract(data)
    except DataContractError as exc:
        print("DATA CONTRACT FAILED: {}".format(exc), file=sys.stderr)
        return 1
    print("DATA CONTRACT OK")
    for line in messages:
        print(" - {}".format(line))
    return 0


if __name__ == "__main__":
    sys.exit(main())
