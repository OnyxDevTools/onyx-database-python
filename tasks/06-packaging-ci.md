# Packaging, CI, and quality tooling

**Gap**: Not published to PyPI; lacks CI with lint/format/mypy/tests/coverage seen in TS project.

**Plan**
- Add PyPI-ready packaging metadata (version, classifiers, long_description) and release steps.
- Introduce lint/format/mypy configs and a minimal unit test suite (mocked HTTP/codegen).
- Set up CI workflow to run lint+tests+coverage badge upload.
- Optional: add smoke job running `scripts/run-examples.sh` with provided config.

**Reference (TypeScript)**: `.github/workflows/ci.yml`, `package.json` scripts, `tests/` in `onyx-database-typescript`
