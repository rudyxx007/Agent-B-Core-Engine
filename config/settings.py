"""
config/settings.py — Centralized Configuration Loader for Agent-B

PURPOSE:
    This is the SINGLE SOURCE OF TRUTH for all configuration values in the
    entire project. Every other Python file imports from here. We NEVER
    scatter os.getenv() calls throughout the codebase.

HOW IT WORKS:
    1. python-dotenv reads your .env file and injects its KEY=VALUE pairs
       into the process environment (os.environ).
    2. This module reads those values once at import time and exposes them
       as clean Python constants.
    3. If a required variable is missing, it raises an error immediately
       rather than failing silently deep inside some pipeline step.

USAGE IN OTHER FILES:
    from config.settings import SUPABASE_URL, SUPABASE_SECRET_KEY
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# 1. Locate and load the .env file
# ---------------------------------------------------------------------------
# Path(__file__)          = this file: Agent-B-Core-Engine/config/settings.py
# .resolve().parent       = Agent-B-Core-Engine/config/
# .parent                 = Agent-B-Core-Engine/          (project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

# load_dotenv() reads the .env file and adds each KEY=VALUE to os.environ.
# override=False means if a variable is already set in the real OS environment
# (e.g., in Modal's secret storage), the OS value takes priority over .env.
load_dotenv(dotenv_path=ENV_PATH, override=False)

# ---------------------------------------------------------------------------
# 2. Read Supabase credentials from environment
# ---------------------------------------------------------------------------
SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
SUPABASE_PUBLISHABLE_KEY: str = os.getenv("SUPABASE_PUBLISHABLE_KEY", "")
SUPABASE_SECRET_KEY: str = os.getenv("SUPABASE_SECRET_KEY", "")

# ---------------------------------------------------------------------------
# 3. Validate: fail fast if credentials are missing
# ---------------------------------------------------------------------------
_REQUIRED = {
    "SUPABASE_URL": SUPABASE_URL,
    "SUPABASE_PUBLISHABLE_KEY": SUPABASE_PUBLISHABLE_KEY,
    "SUPABASE_SECRET_KEY": SUPABASE_SECRET_KEY,
}

_missing = [name for name, value in _REQUIRED.items() if not value]
if _missing:
    print(
        f"\n❌ FATAL: Missing required environment variables: {', '.join(_missing)}\n"
        f"   Make sure your .env file exists at: {ENV_PATH}\n"
        f"   And contains all required keys. See the implementation plan for details.\n",
        file=sys.stderr,
    )
    sys.exit(1)
