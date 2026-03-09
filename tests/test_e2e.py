"""Backward-compatibility shim — tests are now split across:

- test_scenic.py  — Tier 1: Scenic-only tests (no LIBERO)
- test_libero.py  — Tier 2: LIBERO simulation integration
- test_gym.py     — Tier 3: Gym wrapper tests

Run all: pytest tests/ -v

This file is excluded from default collection (see conftest.py collect_ignore)
to avoid running every test twice. To use the shim explicitly::

    pytest tests/test_e2e.py -v
"""

# Re-export all test classes so `pytest tests/test_e2e.py` still works.
from test_gym import *  # noqa: F401, F403
from test_libero import *  # noqa: F401, F403
from test_scenic import *  # noqa: F401, F403
