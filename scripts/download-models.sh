#!/usr/bin/env bash
# Download model files into ./models/.
#
#   ./scripts/download-models.sh                   # Parakeet + Silero VAD only (~640 MB)
#   ./scripts/download-models.sh whisper           # adds Whisper large-v3 (~1.1 GB)
#   ./scripts/download-models.sh voxtral           # adds Voxtral Mini 3B (~3 GB)
#   ./scripts/download-models.sh all               # all three engines
#   ./scripts/download-models.sh whisper voxtral   # multiple
#
# Idempotent — skips files that already exist.
set -euo pipefail

cd "$(dirname "$0")/.."
MODELS_DIR="${MODELS_DIR:-$(pwd)/models}"
mkdir -p "$MODELS_DIR"

WANT_PARAKEET=1
WANT_WHISPER=0
WANT_VOXTRAL=0

if [[ $# -eq 0 ]]; then
  : # default = parakeet only
else
  WANT_PARAKEET=0
  for arg in "$@"; do
    case "$arg" in
      parakeet) WANT_PARAKEET=1 ;;
      whisper)  WANT_WHISPER=1 ;;
      voxtral)  WANT_VOXTRAL=1 ;;
      all)      WANT_PARAKEET=1; WANT_WHISPER=1; WANT_VOXTRAL=1 ;;
      -h|--help)
        sed -n '2,11p' "$0" | sed 's/^# //;s/^#$//'
        exit 0 ;;
      *) echo "unknown engine: $arg" >&2; exit 1 ;;
    esac
  done
fi

fetch() {
  local dest="$1" url="$2"
  if [[ -s "$dest" ]]; then
    printf '✓ %s\n' "$(basename "$dest")"
    return
  fi
  printf '↓ %s\n' "$(basename "$dest")"
  curl -L --fail --progress-bar -o "$dest.part" "$url"
  mv "$dest.part" "$dest"
}

# ----- Parakeet TDT 0.6B V3 INT8 (sherpa-onnx) -----
if [[ "$WANT_PARAKEET" -eq 1 ]]; then
  PK_DIR="$MODELS_DIR/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
  PK_TAR="$MODELS_DIR/parakeet.tar.bz2"
  if [[ ! -d "$PK_DIR" || ! -f "$PK_DIR/encoder.int8.onnx" ]]; then
    fetch "$PK_TAR" \
      "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2"
    echo "extracting parakeet…"
    tar -xjf "$PK_TAR" -C "$MODELS_DIR"
    rm -f "$PK_TAR"
  else
    printf '✓ parakeet/ already extracted\n'
  fi
  fetch "$MODELS_DIR/silero_vad.onnx" \
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"
fi

# ----- Whisper large-v3-turbo (whisper.cpp, multilingual, ~5× faster than v3) -----
if [[ "$WANT_WHISPER" -eq 1 ]]; then
  WHISPER_FILE="${WHISPER_FILE:-ggml-large-v3-turbo-q5_0.bin}"
  fetch "$MODELS_DIR/$WHISPER_FILE" \
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/$WHISPER_FILE"
fi

# ----- Voxtral Mini 3B (llama.cpp / mtmd-cli) -----
if [[ "$WANT_VOXTRAL" -eq 1 ]]; then
  fetch "$MODELS_DIR/Voxtral-Mini-3B-2507-Q4_K_M.gguf" \
    "https://huggingface.co/bartowski/mistralai_Voxtral-Mini-3B-2507-GGUF/resolve/main/mistralai_Voxtral-Mini-3B-2507-Q4_K_M.gguf"
  fetch "$MODELS_DIR/mmproj-Voxtral-Mini-3B-2507-f16.gguf" \
    "https://huggingface.co/bartowski/mistralai_Voxtral-Mini-3B-2507-GGUF/resolve/main/mmproj-mistralai_Voxtral-Mini-3B-2507-f16.gguf"
fi

printf '\nContents of %s:\n' "$MODELS_DIR"
ls -lh "$MODELS_DIR"
