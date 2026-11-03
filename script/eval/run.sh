#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL="${MODEL:-$ROOT/model/LoCo-7B}"
OUT="${OUT:-$ROOT/out/eval}"; mkdir -p "$OUT"
GPU="${GPU:-0}"
BENCHES="math500 amc23 olympiad aime24 aime25"

declare -A EVAL=(
    [math500]=eval_math500   [amc23]=eval_amc23    [olympiad]=eval_olympiad
    [aime24]=eval_aime24     [aime25]=eval_aime25
)

benches=("$@")
if [[ ${#benches[@]} -eq 0 ]]; then
    echo "usage: [GPU=<id>] bash script/eval/run.sh <bench> [bench...]"
    echo "benchmarks: ${BENCHES}"
    exit 1
fi
for b in "${benches[@]}"; do
    if [[ -z "${EVAL[$b]:-}" ]]; then
        echo "unknown benchmark: $b"
        echo "benchmarks: ${BENCHES}"
        exit 1
    fi
done

for b in "${benches[@]}"; do
    echo "[gpu ${GPU}] ${b}"
    CUDA_VISIBLE_DEVICES="$GPU" python -m "eval.${EVAL[$b]}" \
        --base-model "$MODEL" \
        --output "$OUT/${b}.json" \
        --batch-size 1000 \
        > "$OUT/${b}.log" 2>&1
done

echo "=== results:"
for b in "${benches[@]}"; do
    if [[ -f "$OUT/${b}.json" ]]; then
        python3 -c "
import json
d=json.load(open('$OUT/${b}.json'))
print('  ${b}: acc=%.2f%%  tokens=%.0f' % (d.get('accuracy',0)*100, d.get('avg_tokens',0)))
" 2>/dev/null || echo "  ${b}: see $OUT/${b}.json"
    else
        echo "  ${b}: FAILED -- see $OUT/${b}.log"
    fi
done
