#!/bin/bash
# FastWave.AXERA one-click demo: any sample rate -> 48 kHz.
#   bash run_ax650.sh [input.wav] [output.wav] [--cpp|--python] [--steps N] [--overlap R] [--denoise]
set -e
cd "$(dirname "$0")"
export LD_LIBRARY_PATH=/soc/lib:${LD_LIBRARY_PATH:-}
export PYTHONDONTWRITEBYTECODE=1

INPUT="${1:-samples/listen/8k/OSR_us_000_0010_input_8k.wav}"
OUTPUT="${2:-output.wav}"
if [ $# -ge 2 ]; then shift 2; else shift $#; fi

MODE="cpp"
EXTRA=()
for a in "$@"; do
  case "$a" in
    --python) MODE="python" ;;
    --cpp) MODE="cpp" ;;
    *) EXTRA+=("$a") ;;
  esac
done
if [ ! -x ./bin/fastwave_ax650 ]; then MODE="python"; fi

if [ "$MODE" = "cpp" ]; then
  exec ./bin/fastwave_ax650 models/fastwave_nfe.axmodel "$INPUT" "$OUTPUT" \
      --assets models/assets "${EXTRA[@]}"
else
  exec python3 python/demo.py "$INPUT" "$OUTPUT" \
      --model models/fastwave_nfe.axmodel "${EXTRA[@]}"
fi
