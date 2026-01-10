# Release flow (PyPI)

1) Ensure you are on `main` with a clean working tree.
2) Run `scripts/bump-version.sh` and follow prompts:
   - installs deps
   - runs unit tests and py_compile
   - (optional) runs `scripts/run-examples.sh` if a config is present
   - builds + `twine check`
   - bumps `pyproject.toml` version, commits, pushes, tags
3) CI should be configured to publish on tag push (`vX.Y.Z`).

CI coverage:
- `.github/workflows/ci.yml` runs install + unit tests + compile.

PyPI notes:
- Package name: `onyx-database`
- Build: `python -m build`
- Validate: `twine check dist/*`

## Manual publish (local first)
1) Build locally and upload once to PyPI with your token:
   ```bash
   python -m build
   TWINE_USERNAME=__token__ TWINE_PASSWORD=pypi-*** twine upload dist/*
   ```
   Verify `pip install onyx-database` works after the first publish.

2) After the first publish and PyPI project creation, connect CI to the repo for automated releases on tags (optional):
   - Use the trusted publisher flow with the tag-triggered workflow in `.github/workflows/publish.yml`.
   - It triggers on tags like `v*.*.*`, builds the package, and publishes to PyPI via OIDC (no PyPI secrets needed once trusted publisher is set).
   - Tagging `vX.Y.Z` will build and publish automatically.
