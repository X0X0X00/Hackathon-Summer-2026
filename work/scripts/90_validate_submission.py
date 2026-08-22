"""Validate a candidate submission CSV against the official contract."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_WORK / "src"))
from merfish60.io import repo_root  # noqa: E402
ROOT = repo_root()

from merfish60.official_contract import official_example_prediction_path  # noqa: E402
from merfish60.validate_submission import (  # noqa: E402
    SubmissionContractError,
    validate_submission,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        default=str(official_example_prediction_path(ROOT)),
        help="candidate CSV (default: organizer example prediction/prediction.csv)",
    )
    args = parser.parse_args()
    path = Path(args.path)
    try:
        messages = validate_submission(path, ROOT)
    except SubmissionContractError as exc:
        print("SUBMISSION CONTRACT FAILED: {}".format(path), file=sys.stderr)
        for line in exc.violations:
            print(" - {}".format(line), file=sys.stderr)
        return 1
    print("SUBMISSION CONTRACT OK: {}".format(path))
    for line in messages:
        print(" - {}".format(line))
    return 0


if __name__ == "__main__":
    sys.exit(main())
