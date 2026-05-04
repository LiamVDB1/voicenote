# syntax=docker/dockerfile:1.7
# voicenote — single image. Models baked in so the image is portable and there
# are no bind-mount surprises across local Docker / remote daemons / CI.

# ---------- Build stage 1: compile whisper.cpp ----------
FROM debian:bookworm-slim AS whisper-builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git ca-certificates pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
ARG WHISPER_REF=master
RUN git clone --depth 1 --branch ${WHISPER_REF} https://github.com/ggml-org/whisper.cpp.git \
    && cd whisper.cpp \
    && cmake -B build -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build -j"$(nproc)" --config Release \
    && cp build/bin/whisper-cli /usr/local/bin/whisper-cli

RUN mkdir -p /out/lib && \
    (cp /src/whisper.cpp/build/src/libwhisper.so*    /out/lib/ 2>/dev/null || true) && \
    (cp /src/whisper.cpp/build/ggml/src/libggml*.so* /out/lib/ 2>/dev/null || true)


# ---------- Build stage 2: download model files ----------
# Baking models means the image is self-contained — no bind mounts to fight.
# Whisper is opt-in via build arg (saves ~1.1 GB if you don't need it).
FROM debian:bookworm-slim AS models

RUN apt-get update && apt-get install -y --no-install-recommends \
      curl ca-certificates bzip2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /models

# Parakeet TDT 0.6B v3 INT8 (the default engine) — ~640 MB
RUN curl -L --fail --retry 3 -o parakeet.tar.bz2 \
      "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2" \
    && tar xjf parakeet.tar.bz2 \
    && rm parakeet.tar.bz2

# Silero VAD for chunking long audio (~2 MB)
RUN curl -L --fail --retry 3 -o silero_vad.onnx \
      "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"

# Optional: include Whisper large-v3 too. Build with --build-arg INCLUDE_WHISPER=true
ARG INCLUDE_WHISPER=false
ARG WHISPER_FILE=ggml-large-v3-q5_0.bin
RUN if [ "$INCLUDE_WHISPER" = "true" ]; then \
      curl -L --fail --retry 3 -o "$WHISPER_FILE" \
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/$WHISPER_FILE" ; \
    fi


# ---------- Runtime stage ----------
FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    LD_LIBRARY_PATH=/usr/local/lib

RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg ca-certificates curl tini \
      libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# whisper.cpp binary + its shared libs (small, always included)
COPY --from=whisper-builder /usr/local/bin/whisper-cli /usr/local/bin/whisper-cli
COPY --from=whisper-builder /out/lib/                  /usr/local/lib/
RUN ldconfig

# Pre-baked models — image is now self-contained
COPY --from=models /models /models

WORKDIR /app

# Python deps
COPY api/pyproject.toml /app/pyproject.toml
RUN pip install --upgrade pip && \
    pip install \
      "fastapi>=0.115" "uvicorn[standard]>=0.32" "python-multipart>=0.0.12" \
      "pydantic>=2.9" "pydantic-settings>=2.6" \
      "sqlalchemy>=2.0" "aiosqlite>=0.20" \
      "bcrypt>=4.2" "itsdangerous>=2.2" "httpx>=0.27" \
      "numpy>=1.26" "sherpa-onnx>=1.10"

# App code + web files (always baked in — no bind-mount surprises)
COPY api/voicenote /app/voicenote
COPY web           /app/web

RUN mkdir -p /data

ENV VN_DATA_DIR=/data \
    VN_MODELS_DIR=/models \
    VN_WEB_DIR=/app/web \
    VN_HOST=0.0.0.0 \
    VN_PORT=8732

EXPOSE 8732

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:8732/v1/health >/dev/null || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "uvicorn", "voicenote.main:app", "--host", "0.0.0.0", "--port", "8732"]
