# yolo-server for Railway. The build context is this directory (the repo root): point the
# Railway service's root directory here. Not deployed from this repo's CI.
#
# onnxruntime is the official release tarball, pinned and sha256-verified at
# build time. Keep ORT_VERSION and ORT_SHA256 in step with
# scripts/install-onnxruntime.sh.

FROM rust:bookworm AS build

ARG ORT_VERSION=1.30.0
ARG ORT_SHA256=a5ed5a3cac51fbb2e90da632ae43d19212faaa20e76484e62bcb7c23ddb3b3fd

RUN curl -fsSL -o /tmp/ort.tgz \
      "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz" \
    && echo "${ORT_SHA256}  /tmp/ort.tgz" | sha256sum -c - \
    && mkdir -p /opt/onnxruntime \
    && tar -xzf /tmp/ort.tgz -C /opt/onnxruntime --strip-components=1 \
    && rm /tmp/ort.tgz

WORKDIR /build
COPY rust-toolchain.toml Cargo.toml Cargo.lock ./
COPY src ./src
# LIBRARY_PATH is for the linker; the runtime image gets the library below.
RUN LIBRARY_PATH=/opt/onnxruntime/lib cargo build --release --locked

FROM debian:bookworm-slim

ARG ORT_VERSION=1.30.0

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --home /app --shell /usr/sbin/nologin yolo

COPY --from=build /opt/onnxruntime/lib/libonnxruntime.so.${ORT_VERSION} /usr/local/lib/
RUN ln -s libonnxruntime.so.${ORT_VERSION} /usr/local/lib/libonnxruntime.so.1 && ldconfig

WORKDIR /app
COPY --from=build /build/target/release/yolo-server /usr/local/bin/yolo-server
COPY models ./models
RUN chown -R yolo:yolo /app

USER yolo
ENV PORT=5001
ENV YOLO_MODEL=/app/models/yolo26n-seg.onnx
EXPOSE 5001

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD ["yolo-server", "healthcheck"]

CMD ["yolo-server"]
