#!/usr/bin/env python
"""Compatibility shim: ``python scripts/run_orbit.py ...``.

The CLI now lives in the installed package at ``orbit/cli.py`` so that
``python -m orbit`` and the ``orbit`` console script work after a plain
``pip install``. This shim keeps the in-checkout invocation working.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from orbit.cli import main

if __name__ == "__main__":
    main()
