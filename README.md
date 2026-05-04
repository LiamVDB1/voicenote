# voicenote

Self-hosted multilingual transcription with a beautiful web UI. **NVIDIA Parakeet
TDT 0.6B v3** (default — fast, ASR-native, EN/NL/FR/DE + 21 more) with optional
**Whisper large-v3** as a fallback. SQLite history, simple login + API keys,
REST API, CLI. The web UI keeps things simple: upload → transcript. Engine
choice is a power-user concern, available via the CLI / API only.

```
[ web UI · CLI · API clients ]
              │
       FastAPI  (port 8732)  ─── SQLite (./data/voicenote.db)
              │
   ┌──────────┴────────┬──────────────────┐
   ▼                   ▼                  ▼
 Parakeet           Whisper            Voxtral
 (sherpa-onnx)      (whisper.cpp)      (llama.cpp)
   default            optional           optional
   ~640 MB            ~1.1 GB            ~3 GB
              │
            ./models/   (only what you download is active)
```

## Why this stack

Parakeet is a **600M-param ASR model**; Voxtral is an audio-language model.
For raw "audio → text", Parakeet is the right class of tool: smaller, faster,
ASR-native. Whisper sits in the middle as a mature, well-trodden fallback —
particularly strong on Dutch.

| Engine | FLEURS WER · en | nl | fr | de | RAM | Disk |
|---|---|---|---|---|---|---|
| **Parakeet TDT 0.6B v3 (INT8)** | 5–7 | 12.78 | **4.97** | — | ~1 GB | ~640 MB |
| **Whisper large-v3** (Q5_0) | 4.00 | **5.87** | 5.55 | 5.46 | ~4 GB | ~1.1 GB |
| Voxtral Mini 3B (Q4_K_M) | 3.61 | 4.89 | 4.22 | 3.54 | ~3 GB | ~3 GB |

Parakeet wins decisively on speed (≈30× Whisper on CPU) and on
French. Whisper has the cleanest Dutch numbers; that's why it's the default
fallback. Voxtral's quality is excellent but llama.cpp's audio path is
upstream-flagged "experimental" — it's there if you want it.

(WER from the Parakeet & Voxtral papers; lower is better.)

## Layout

```
voicenote/
├── api/                       FastAPI backend (Python 3.11+)
│   ├── voicenote/
│   │   ├── main.py            app factory + static mount
│   │   ├── config.py          env-driven settings
│   │   ├── db.py              SQLAlchemy async + SQLite
│   │   ├── auth.py            bcrypt + signed cookie + API keys
│   │   ├── audio.py           ffmpeg → 16kHz mono wav
│   │   ├── engines/           parakeet · whisper · voxtral
│   │   ├── routes/            /v1/{auth,transcribe,transcripts,health}
│   │   └── scripts/seed_user.py
│   └── Dockerfile             builds whisper.cpp; pip-installs sherpa-onnx
├── web/                       static frontend (no build step)
│   ├── index.html  login.html  style.css  app.js
├── cli/voicenote              pure-stdlib Python CLI
├── deploy/nginx.conf.example  drop-in proxy_pass for your existing nginx
├── docker-compose.yml
├── scripts/download-models.sh
└── README.md
```

## 1. Quickstart — local Mac

Prereq: Docker Desktop. Default model footprint: **~640 MB** (Parakeet only).

```bash
cd ~/Workspace/Active/Personal/voicenote

cp .env.example .env
echo "VN_SECRET_KEY=$(openssl rand -hex 32)" >> .env

# Default download = Parakeet + Silero VAD only
./scripts/download-models.sh
# Want the Whisper fallback too? ~1.1 GB more:
# ./scripts/download-models.sh whisper
# Want the Voxtral fallback? ~3 GB more:
# ./scripts/download-models.sh voxtral
# Or all of it: ./scripts/download-models.sh all

docker compose up -d --build
docker compose logs -f voicenote   # ctrl-c to detach

# Create your first user (prompts for password)
docker compose exec voicenote \
  python -m voicenote.scripts.seed_user --username liam --display-name "Liam"

# Visit http://localhost:8732
```

The web UI auto-detects which engines have models present and disables the
others. Mom only ever sees the engines you've actually downloaded.

## 2. Quickstart — Oracle Free Tier (Ampere A1, 4 OCPU / 24 GB)

Same flow. The container binds to `127.0.0.1:8732` (see `docker-compose.yml`),
so nothing is exposed unless your existing nginx proxies to it.

```bash
git clone <your-fork> voicenote && cd voicenote
cp .env.example .env
echo "VN_SECRET_KEY=$(openssl rand -hex 32)" >> .env
echo "VN_COOKIE_SECURE=true" >> .env       # once you have HTTPS

./scripts/download-models.sh               # Parakeet only
docker compose up -d --build

docker compose exec voicenote \
  python -m voicenote.scripts.seed_user --username mama --display-name "Mama"
```

Then wire it into nginx — copy `deploy/nginx.conf.example` to
`/etc/nginx/sites-available/voicenote`, edit the hostname:

```bash
sudo ln -sf /etc/nginx/sites-available/voicenote /etc/nginx/sites-enabled/voicenote
sudo nginx -t && sudo nginx -s reload
sudo certbot --nginx -d voicenote.example.com   # HTTPS
```

After certbot succeeds, set `VN_COOKIE_SECURE=true` in `.env` and
`docker compose up -d`.

### Performance expectations on Ampere A1

| Engine | RAM peak | 5-min audio | 30-min audio |
|---|---|---|---|
| **Parakeet INT8 (with VAD)** | ~1 GB | ~30–60 s | ~3–6 min |
| Whisper large-v3 Q5 | ~4 GB | ~6–10 min | ~40 min – 1 h |
| Voxtral Mini 3B Q4 | ~3 GB | ~10–18 min | ~70–110 min |

Parakeet's 30-50× speed advantage on CPU is exactly why it's now the default.
For an hour-long meeting, Parakeet finishes in minutes and leaves the box quiet.

## 3. Using it

### Web (mom-friendly)

`http://localhost:8732` (or your domain) → log in → drop a file → done.
The engine toggle shows "Snel" (Parakeet) and "Zorgvuldig" (Whisper).
Whichever you pick, the server falls back automatically if the choice fails.

### CLI (your daily driver, uses an API key)

```bash
# Once: create a key in the UI (your-pill → Geschiedenis page is coming;
# for now via API or curl). Or via the database directly with a quick SQL.
# Then:

~/Workspace/Active/Personal/voicenote/cli/voicenote login \
  --server https://voicenote.example.com \
  --key vn_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Then anywhere
voicenote transcribe ~/note.m4a
voicenote transcribe ~/note.m4a -o note.txt
voicenote transcribe ~/note.m4a --json
voicenote transcribe ~/note.m4a --engine whisper --lang nl
voicenote list
voicenote get 42
voicenote health
```

Pure stdlib. Symlink it anywhere on `$PATH`:

```bash
ln -s ~/Workspace/Active/Personal/voicenote/cli/voicenote ~/.local/bin/voicenote
```

### API (for SecondBrain or any other project)

```bash
SERVER=https://voicenote.example.com
KEY=vn_xxxxxxxxxxxxxxxxxxxxxxxxxx

curl -sS -X POST "$SERVER/v1/transcribe" \
  -H "Authorization: Bearer $KEY" \
  -F "audio=@./note.m4a" \
  -F "engine=parakeet" \
  -F "language=nl" | jq .
```

OpenAPI docs live at `/api/docs`.

### SecondBrain integration sketch

In your vault's voice-note ingest:

```bash
voicenote transcribe "raw/voice_notes/$1.m4a" \
  -o "raw/voice_notes/$1.transcript.txt"
```

Each transcript also lives in voicenote's DB; refetch later with
`voicenote get <id>`.

## 4. Configuration

All env, prefixed `VN_`. The ones you'll actually touch:

- `VN_SECRET_KEY` — set once, treat like a password
- `VN_DEFAULT_ENGINE` — `parakeet` (default) | `whisper` | `voxtral`
- `VN_FALLBACK_CHAIN` — `whisper,voxtral` (only kicks in if the model files are present)
- `VN_INFERENCE_THREADS` — 3 on A1, 4–6 on a desktop
- `VN_USE_VAD` — `true` (default) — Silero VAD chunks long audio so we don't OOM
- `VN_COOKIE_SECURE` — flip to `true` once you're behind HTTPS

## 5. Engine cascade behavior

When you POST to `/v1/transcribe` with `engine=parakeet`:

1. Try Parakeet. If model files exist + sherpa-onnx is installed → run.
2. On failure, walk `VN_FALLBACK_CHAIN`. Skip engines whose model files aren't
   downloaded (no need to download them just to be skipped).
3. If everything fails, return 500 with the per-engine errors.

Result includes `attempts: [...]` so you can see which engines were tried.

## 6. Troubleshooting

**"Geen beschikbare engine" / 500 on every request.**
Run `curl -sS http://localhost:8732/v1/health | jq` — it shows which engines
have their models present. The default download script gets Parakeet only;
add Whisper or Voxtral if you want fallbacks active.

**Slow Parakeet first request.**
First call loads the ONNX model (~1.5 s). Subsequent calls reuse the loaded
recognizer and are instant.

**`whisper-cli` / `llama-mtmd-cli` not found.**
The Dockerfile builds both from `master`. If upstream broke something, pin
tags: `docker compose build --build-arg WHISPER_REF=v1.8.4 --build-arg LLAMA_REF=b6300`.

**Voxtral fallback consistently fails.**
llama.cpp's mtmd audio is upstream-flagged experimental. Drop it from the
chain: `VN_FALLBACK_CHAIN=whisper`. Quality won't suffer noticeably.

**Login cookie not sticking on HTTPS.**
Set `VN_COOKIE_SECURE=true` and ensure nginx forwards `X-Forwarded-Proto`
(the example config does).

## 7. Endpoints reference

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET`    | `/v1/health`              | —      | per-engine readiness |
| `POST`   | `/v1/auth/login`          | —      | username + password → cookie |
| `POST`   | `/v1/auth/logout`         | —      | clears cookie |
| `GET`    | `/v1/me`                  | yes    | current user |
| `GET`    | `/v1/keys`                | yes    | list API keys |
| `POST`   | `/v1/keys`                | yes    | create API key (returned once) |
| `DELETE` | `/v1/keys/{id}`           | yes    | revoke key |
| `POST`   | `/v1/transcribe`          | yes    | upload (multipart `audio`); returns `attempts[]` |
| `GET`    | `/v1/transcripts`         | yes    | list (limit/offset) |
| `GET`    | `/v1/transcripts/{id}`    | yes    | full transcript + segments |
| `DELETE` | `/v1/transcripts/{id}`    | yes    | delete |

Auth = either `Cookie: vn_session=…` (web) or `Authorization: Bearer vn_…` (programmatic).

## 8. License

MIT for this scaffolding. Models are governed by their respective licenses
(Parakeet: NVIDIA Open Model License; Whisper: MIT; Voxtral: Apache 2.0).
