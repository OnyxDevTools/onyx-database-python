#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
examples_root="${repo_root}/examples"

# Resolve config path with fallbacks
config_candidates=(
  "${examples_root}/onyx-database.json"
  "${repo_root}/config/onyx-database.json"
  "${repo_root}/onyx-database.json"
)
export ONYX_CONFIG_PATH=""
for cfg in "${config_candidates[@]}"; do
  if [[ -f "$cfg" ]]; then
    ONYX_CONFIG_PATH="$cfg"
    break
  fi
done

if [[ -z "${ONYX_CONFIG_PATH}" ]]; then
  echo "Missing onyx-database.json. Place one in examples/ or config/."
  exit 1
fi

export ONYX_SCHEMA_PATH="${examples_root}/onyx.schema.json"

examples=(
  "delete/by-id:delete/by_id.py"
  "delete/query:delete/query.py"
  "document/save-get-delete:document/save_get_delete_document.py"
  "query/aggregate-avg:query/aggregate_avg.py"
  "query/aggregates-with-grouping:query/aggregates_with_grouping.py"
  "query/basic:query/basic.py"
  "query/basic-inference:query/basic_inference.py"
  "query/find-by-id:query/find_by_id.py"
  "query/first-or-none:query/first_or_none.py"
  "query/in-partition:query/in_partition.py"
  "query/inner-query:query/inner_query.py"
  "query/list:query/list.py"
  "query/order-by:query/order_by.py"
  "query/compound:query/compound.py"
  "query/not-inner-query:query/not_inner_query.py"
  "query/resolver:query/resolver.py"
  "query/search-by-resolver-fields:query/search_by_resolver_fields.py"
  "query/select:query/select_example.py"
  "query/sorting-and-paging:query/sorting_and_paging.py"
  "query/update:query/update.py"
  "save/basic:save/basic.py"
  "save/batch-save:save/batch_save.py"
  "save/cascade:save/cascade.py"
  "save/cascade-builder:save/cascade_builder.py"
  "schema/basic:schema/basic.py"
  "secrets/basic:secrets/basic.py"
  "stream/close:stream/close.py"
  "stream/create-events:stream/create_events.py"
  "stream/delete-events:stream/delete_events.py"
  "stream/query-stream:stream/query_stream.py"
  "stream/update-events:stream/update_events.py"
  "seed:seed.py"
)

line_width=50
green=$'\033[32m'
red=$'\033[31m'
reset=$'\033[0m'
passed=0
failed=0
declare -a failed_names=()
declare -a failed_logs=()

if [[ ! -d "${examples_root}/.venv" ]]; then
  python3 -m venv "${examples_root}/.venv"
fi
source "${examples_root}/.venv/bin/activate"
python -m pip install -e "${repo_root}"

# seed data for examples (fails fast on errors)
echo "seeding examples..."
if ! seed_output=$(cd "${examples_root}" && python seed.py 2>&1); then
  echo "seed failed:"
  echo "$seed_output"
  exit 1
fi
echo "seed output:"
echo "$seed_output"
echo

for entry in "${examples[@]}"; do
  name=${entry%%:*}
  path=${entry#*:}
  status="FAIL"
  output=""

  if output=$(cd "${examples_root}" && python "$path" 2>&1); then
    status="PASS"
    ((passed++))
  else
    status="FAIL"
    ((failed++))
    failed_names+=("$name")
    failed_logs+=("$output")
  fi

  dots_count=$((line_width - ${#name} - ${#status}))
  if ((dots_count < 1)); then
    dots_count=1
  fi
  dots=$(printf '%*s' "$dots_count" '' | tr ' ' '.')
  color="$red"
  if [[ "$status" == "PASS" ]]; then
    color="$green"
  fi
  printf '%s%s%s%s%s\n' "$name" "$dots" "$color" "$status" "$reset"
done

if ((failed > 0)); then
  echo
  echo "failed logs"
  echo "-----------"
  for i in "${!failed_names[@]}"; do
    echo "[$((i+1))] ${failed_names[$i]}"
    echo "${failed_logs[$i]}"
    echo "-----------"
  done
fi

echo
echo "totals"
echo "------"
printf 'PASSED: %s%d%s\n' "$green" "$passed" "$reset"
printf 'FAILED: %s%d%s\n' "$red" "$failed" "$reset"

if ((failed > 0)); then
  exit_code=$failed
  if ((exit_code > 255)); then
    exit_code=255
  fi
  exit "$exit_code"
fi
