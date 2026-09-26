#!/usr/bin/env bash
# Autor: Adam Skrodzki
# Benchmark wszystkich ekspertów GPT (data/models/*_ckpt.pt) sędzią judge_v2.
# Log każdej sesji do data/benchmarks/bench_<timestamp>.log — wyniki interpretujemy później.
# Użycie: ./run_benchmarks.sh [judge] [samples]
#   ./run_benchmarks.sh                                  # judge_v2, 100 próbek
#   ./run_benchmarks.sh data/models/judge_v2.pt 50       # inny sędzia / mniej próbek
set -u
cd "$(dirname "$0")"

JUDGE="${1:-data/models/judge_v2.pt}"
SAMPLES="${2:-100}"
OUT="data/benchmarks/bench_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$OUT")"

echo "sędzia: $JUDGE | próbek/model: $SAMPLES | log: $OUT" | tee "$OUT"
for ckpt in data/models/*_ckpt.pt; do
  [ -e "$ckpt" ] || continue   # glob bez trafień
  echo "================================================================" | tee -a "$OUT"
  echo ">>> $ckpt" | tee -a "$OUT"
  python src/tools/benchmark_judge.py --model "$ckpt" --judge "$JUDGE" \
      --samples "$SAMPLES" 2>&1 | tee -a "$OUT"
done

echo "================================================================" | tee -a "$OUT"
echo "gotowe: $(ls data/models/*_ckpt.pt | wc -l) checkpointów -> $OUT" | tee -a "$OUT"
