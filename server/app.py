"""
server/app.py — OpenEnv multi-mode deployment entry point.

Required by OpenEnv spec: server/app.py must exist and expose `app`.
This file imports and re-exports the FastAPI app from api/server.py.
"""

import os
import sys

# Make project root importable
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)

from api.server import app  # noqa: F401 — re-exported for OpenEnv

__all__ = ["app"]