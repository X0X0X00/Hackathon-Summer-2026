import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
src = ROOT / "work" / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))
