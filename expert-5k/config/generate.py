from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv: list[str] | None = None) -> int:
    return import_module("scripts.build_k_gamma_targets").main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
