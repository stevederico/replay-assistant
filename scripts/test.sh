#!/usr/bin/env bash
# Run the whole test suite. Uses the local onnxruntime when it is installed
# (scripts/install-onnxruntime.sh); otherwise runs only the pure-logic tests and
# says so, because the model tests cannot link without the library.
set -euo pipefail
cd "$(dirname "$0")/.."

# shellcheck source=scripts/ort-env.sh
. scripts/ort-env.sh
if [ -e "${ORT_HOME}/lib/libonnxruntime.so" ]; then
  cargo test --locked "$@"
else
  echo "SKIP: libonnxruntime not found at ${ORT_HOME}; run scripts/install-onnxruntime.sh." >&2
  echo "      Running the pure-logic tests only (--no-default-features)." >&2
  cargo test --locked --no-default-features "$@"
fi
