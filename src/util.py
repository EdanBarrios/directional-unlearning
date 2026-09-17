"""Small helpers shared by every script: smoke flag, provenance hashes."""

import hashlib
import os
import subprocess
from pathlib import Path

SMOKE = os.environ.get("SMOKE") == "1"

# Smoke settings. Every script that honors SMOKE reads them from here.
SMOKE_MODEL = "EleutherAI/pythia-70m"
SMOKE_FACTS = 5
SMOKE_TRAIN_TEMPLATES = 2
SMOKE_HELDOUT_TEMPLATES = 1
SMOKE_STEPS = 3


def git_commit() -> str:
    """Current commit hash, with '-dirty' if the tree has uncommitted changes."""
    try:
        h = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet"]) != 0
        return h + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def file_sha(path: str | Path) -> str:
    """Short sha256 of a file, for recording which inputs produced an output."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]
