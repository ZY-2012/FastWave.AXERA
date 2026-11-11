#!/usr/bin/env bash
# Compile the FastWave NFE ONNX into an AX650 AXMODEL with Pulsar2 8.0.
# usage: bash compile.sh [pulsar2 build config]
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PULSAR2="${PULSAR2:-pulsar2}"
CFG="${1:-$ROOT/compile/pulsar2_config.json}"
"$PULSAR2" build --config "$CFG" 2>&1 | tee "$ROOT/compile/compile.log"
