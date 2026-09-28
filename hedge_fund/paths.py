"""Where user data lives: ~/.hedge-fund/.

Everything the user owns — mandates, run/backtest receipts, API caches, and
the .env key file — lives under one home directory, outside the package. The
package directory stays read-only code, so a pipx install behaves exactly
like a checkout.

Textual-free and import-light on purpose: every layer (CLI, TUI, caches)
anchors its paths here, and nothing here may import them back.
"""

from __future__ import annotations

import shutil
from pathlib import Path

USER_DIR = Path.home() / ".hedge-fund"
MANDATES_DIR = USER_DIR / "mandates"
CACHE_DIR = USER_DIR / "cache"
RESEARCH_DIR = USER_DIR / "research"   # research reports saved by the web app
ENV_PATH = USER_DIR / ".env"

UNIVERSES_DIR = USER_DIR / "universes"

# The example mandate and universe ship inside the package; they are copied
# out (never read in place) so users edit their copy, not the install.
EXAMPLE_MANDATE = Path(__file__).resolve().parent / "fund" / "example.yaml"
EXAMPLE_UNIVERSE = Path(__file__).resolve().parent / "fund" / "example-universe.yaml"


def ensure_mandates_dir() -> Path:
    """Create the mandates dir on first use, seeded with the example."""
    if not MANDATES_DIR.exists():
        MANDATES_DIR.mkdir(parents=True)
        shutil.copy(EXAMPLE_MANDATE, MANDATES_DIR / "example.yaml")
    return MANDATES_DIR


def ensure_universes_dir() -> Path:
    """Create the universes dir on first use, seeded with the example."""
    if not UNIVERSES_DIR.exists():
        UNIVERSES_DIR.mkdir(parents=True)
        shutil.copy(EXAMPLE_UNIVERSE, UNIVERSES_DIR / "example.yaml")
    return UNIVERSES_DIR
