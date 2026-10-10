"""Builder agent: scan -> decide -> render."""

import os
from pathlib import Path

# where the attempt and call logs go; tests set BUILDER_LOG_DIR to a temp folder (tests/conftest.py)
LOG_DIR = Path(os.environ.get("BUILDER_LOG_DIR") or Path(__file__).resolve().parents[1] / "logs")
