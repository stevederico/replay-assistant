#!/usr/bin/env bash
# Download the official onnxruntime release tarball into ~/.local/opt (no sudo)
# and verify a pinned sha256 before unpacking. Idempotent.
#
# Keep ORT_VERSION and ORT_SHA256 in step with the Dockerfile.
set -euo pipefail

ORT_VERSION="1.30.0"
ORT_SHA256="a5ed5a3cac51fbb2e90da632ae43d19212faaa20e76484e62bcb7c23ddb3b3fd"
DEST="${ORT_PREFIX:-$HOME/.local/opt}"
NAME="onnxruntime-linux-x64-${ORT_VERSION}"
URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${NAME}.tgz"

if [ -e "${DEST}/${NAME}/lib/libonnxruntime.so" ]; then
  echo "onnxruntime ${ORT_VERSION} already at ${DEST}/${NAME}"
  exit 0
fi

mkdir -p "${DEST}"
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT
curl -fsSL -o "${tmp}/${NAME}.tgz" "${URL}"
echo "${ORT_SHA256}  ${tmp}/${NAME}.tgz" | sha256sum -c -
tar -xzf "${tmp}/${NAME}.tgz" -C "${DEST}"
echo "installed ${DEST}/${NAME}"
