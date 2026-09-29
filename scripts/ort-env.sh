# Source this file: `. scripts/ort-env.sh`
# Points the linker and the loader at the locally installed onnxruntime.
ORT_VERSION="1.30.0"
ORT_HOME="${ORT_HOME:-$HOME/.local/opt/onnxruntime-linux-x64-${ORT_VERSION}}"
export LIBRARY_PATH="${ORT_HOME}/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export LD_LIBRARY_PATH="${ORT_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
