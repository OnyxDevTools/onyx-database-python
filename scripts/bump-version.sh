# filename: scripts/bump-version.sh
#!/usr/bin/env bash
# Interactive version bump + publish trigger for PyPI.
# Flow:
#   - ensure main branch
#   - install deps, run tests, compile
#   - optional smoke examples (if config present)
#   - build + twine check
#   - bump version in pyproject.toml
#   - commit, push, tag (CI should publish on tag)

set -euo pipefail

DRYRUN=0
if [[ "${1:-}" == "--dryrun" ]]; then
  DRYRUN=1
  shift
fi

abort() { echo "ERROR: $*" >&2; exit 1; }
info()  { echo "==> $*"; }
cmd()   { echo "+ $*"; "$@"; }

[[ -f "pyproject.toml" ]] || abort "Run from the repo root (pyproject.toml not found)."

CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
MAIN_BRANCH="main"

if [[ "${CURRENT_BRANCH}" != "${MAIN_BRANCH}" ]]; then
  info "Switching to ${MAIN_BRANCH}..."
  cmd git checkout "${MAIN_BRANCH}"
fi

# Ensure working tree is clean (or auto-commit if user agrees)
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Working tree is not clean."
  read -rp "Enter a commit message to save current changes (or leave empty to abort): " AUTO_COMMIT_MSG
  if [[ -z "${AUTO_COMMIT_MSG}" ]]; then
    abort "Working tree not clean. Commit or stash changes first."
  fi
  info "Committing pending changes..."
  cmd git add -A
  cmd git commit -m "${AUTO_COMMIT_MSG}"
  cmd git push origin "${MAIN_BRANCH}"
else
  AUTO_COMMIT_MSG=""
fi

info "Installing deps..."
cmd python -m pip install --upgrade pip
cmd python -m pip install -e .

# Ensure required tooling is present
MISSING=()
for pkg in build twine tomli_w ruff flake8; do
  import_name="${pkg//-/_}"
  if ! python -c "import ${import_name}" >/dev/null 2>&1; then
    MISSING+=("${pkg}")
  fi
done
if (( ${#MISSING[@]} )); then
  info "Installing missing tools: ${MISSING[*]}"
  cmd python -m pip install "${MISSING[@]}"
fi

info "Running unit tests..."
cmd python -m unittest discover tests

info "Compiling sources..."
cmd python -m py_compile $(find onyx_database tests -name '*.py')

# Optional lint phase gate if ruff or flake8 is available
if command -v ruff >/dev/null 2>&1; then
  info "Linting with ruff..."
  cmd ruff check onyx_database examples tests
elif command -v flake8 >/dev/null 2>&1; then
  info "Linting with flake8..."
  cmd flake8 onyx_database examples tests
else
  info "Lint step skipped (ruff/flake8 not installed)."
fi

# Optional smoke examples if config is present
if [[ -f "./examples/onyx-database.json" || -f "./config/onyx-database.json" || -f "./onyx-database.json" ]]; then
  info "Running example smoke (scripts/run-examples.sh)..."
  ./scripts/run-examples.sh || abort "Examples failed; fix before bump."
else
  info "Skipping example smoke (no config found)."
fi

info "Building distributions..."
cmd python -m build

info "Twine check..."
cmd twine check dist/* 

if [[ "${DRYRUN}" -eq 1 ]]; then
  info "Dry run complete. Skipping version bump/tagging."
  exit 0
fi

echo "Select version bump type:"
select BUMP_TYPE in patch minor major; do
  [[ -n "${BUMP_TYPE}" ]] && break
done

read -rp "Enter release message: " MESSAGE
if [[ -z "${MESSAGE}" && -n "${AUTO_COMMIT_MSG}" ]]; then
  MESSAGE="${AUTO_COMMIT_MSG}"
fi
MESSAGE="${MESSAGE:-"${BUMP_TYPE} release"}"

CURRENT_VERSION="$(python - <<'PY'
import tomllib
with open("pyproject.toml","rb") as f:
    data=tomllib.load(f)
print(data.get("project",{}).get("version",""))
PY
)"
[[ -n "${CURRENT_VERSION}" ]] || abort "Could not read current version."

next_version() {
  IFS='.' read -r MAJ MIN PATCH <<<"$1"
  case "$2" in
    patch) PATCH=$((PATCH+1));;
    minor) MIN=$((MIN+1)); PATCH=0;;
    major) MAJ=$((MAJ+1)); MIN=0; PATCH=0;;
    *) abort "Unknown bump type $2";;
  esac
  echo "${MAJ}.${MIN}.${PATCH}"
}

NEW_VERSION="$(next_version "${CURRENT_VERSION}" "${BUMP_TYPE}")"
info "Bumping version ${CURRENT_VERSION} -> ${NEW_VERSION}"

python - <<PY
from pathlib import Path
import tomllib
import tomli_w
p = Path("pyproject.toml")
data = tomllib.loads(p.read_text())
data.setdefault("project", {})["version"] = "${NEW_VERSION}"
p.write_text(tomli_w.dumps(data))
PY

if git diff --quiet; then
  abort "No changes after version bump."
fi

info "Committing version bump..."
cmd git add pyproject.toml
cmd git commit -m "chore: release v${NEW_VERSION}"
cmd git push origin "${MAIN_BRANCH}"

TAG="v${NEW_VERSION}"
info "Tagging ${TAG}..."
if git rev-parse "${TAG}" >/dev/null 2>&1; then
  info "Tag ${TAG} exists; moving to current HEAD."
  git tag -d "${TAG}" >/dev/null 2>&1 || true
fi
cmd git tag -a "${TAG}" -m "${TAG}"
cmd git push origin "${TAG}"

cat <<NOTE

Done.
- Bump type: ${BUMP_TYPE}
- Message:   ${MESSAGE}
- Version:   ${NEW_VERSION}
- Tag:       ${TAG}

CI publish will run on tag ${TAG}.

Links:
- PyPI project: https://pypi.org/project/onyx-database/
- GitHub Actions: https://github.com/OnyxDevTools/onyx-database-python/actions

NOTE
