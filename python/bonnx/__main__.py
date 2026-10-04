import sys
from pathlib import Path

if __package__ in (None, ""):  # `python python/bonnx/__main__.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bonnx.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
