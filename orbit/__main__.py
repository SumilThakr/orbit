"""Console entry point: ``python -m orbit ...``.

Delegates to the CLI in :mod:`orbit.cli`, which is also exposed as the
``orbit`` console script (see ``pyproject.toml``).
"""

from orbit.cli import main

if __name__ == "__main__":
    main()
