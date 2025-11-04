"""Package initializer for the `src` package.

This file is intentionally present so that Python treats the `src` directory as a
regular package (not a namespace package). Having an `__init__.py` helps with:

- Explicit package imports (e.g., `from src import app`).
- Tooling and packaging systems that expect a concrete package file.

If you intentionally move to PEP 420 implicit namespace packages, this file can be
removed, but you'll need to update imports and packaging/tooling configurations.
"""

# Expose package-level names here if desired. Keep this file minimal to avoid
# side-effects during import.
