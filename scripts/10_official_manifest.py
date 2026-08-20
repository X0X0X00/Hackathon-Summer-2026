"""Write or verify experiments/official_data_manifest.json."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from merfish60.official_contract import (  # noqa: E402
    OfficialContractError,
    verify_official_manifest,
    write_official_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="verify official CSVs against the saved manifest and exit nonzero on change",
    )
    args = parser.parse_args()
    try:
        if args.verify:
            messages = verify_official_manifest(ROOT)
            print("OFFICIAL MANIFEST OK")
            for line in messages:
                print(" - {}".format(line))
            return 0
        path = write_official_manifest(ROOT)
        print("Wrote {}".format(path))
        return 0
    except OfficialContractError as exc:
        print("OFFICIAL MANIFEST FAILED: {}".format(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
