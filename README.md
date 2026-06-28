# ChatGB10 — A Private Multi-Model AI Router

Turn a desk-side AI box (built and tuned on a **Dell Pro Max with GB10**, but works on any Ollama host) into a private, four-tier AI assistant — **chat, code, reasoning, vision, and document reading** — behind one OpenAI-compatible endpoint, with a built-in web UI. Nothing leaves your network.

> One door, and the right brain answers every time.

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## What's new in v2.2 (See, Listen & Talk)

ChatGB10 now has three senses, all running on the GB10:

- **👁 See** — capture a webcam photo with the 📷 button and ask about it; the frame is read by the local **vision** model. Snapshot-based, fully on-box.
- **👂 Listen** — record with the 🎤 button and your speech is transcribed by **Whisper running locally on the GB10** (no cloud), then dropped into the composer so you can review it before sending.
- **🗣 Talk** — turn on **Speak replies aloud** in Settings (or press the 🔊 button on any reply) to hear answers via your browser's offline voice.

**Enable the microphone** (optional install on the GB10 — pick one):

```
pip install faster-whisper --break-system-packages          # recommended (GPU)
# or:
pip install openai-whisper --break-system-packages
sudo apt-get install -y ffmpeg                              # needed by openai-whisper
```

Without a backend, the mic returns a clear message instead of failing silently; **See** and **Talk** are unaffected.

**Heads-up — camera & mic need a secure context.** Browsers only allow camera/microphone access over HTTPS or `localhost`. If you reach ChatGB10 over plain HTTP at an IP address, add that origin to `chrome://flags/#unsafely-treat-insecure-origin-as-secure` (then relaunch), or serve it over HTTPS. Voice output works either way.

Optional speech-to-text tuning via environment variables: `CHATGB10_STT_BACKEND` (auto / faster / openai), `CHATGB10_STT_MODEL` (tiny / base / small / medium / large-v3), `CHATGB10_STT_DEVICE` (cuda / cpu / auto), `CHATGB10_STT_LANGUAGE` (blank auto-detects).

---

## What's new in v2.1 (RAG Update)

- **Knowledge base (RAG)** — build a private, on-box document knowledge base and query it in chat with `@kb your question`. Documents are chunked and embedded with a **local** Ollama embedding model, stored in an on-device vector index, and the most relevant passages are retrieved to answer — **with source citations**. Per-user and fully on-prem; nothing leaves the box.
- Manage it under **Settings -> Knowledge base**: add documents, see doc/chunk counts, or clear.
- Requires a local embedding model: `ollama pull nomic-embed-text` (configurable in `config.yaml`).

---

## What's new in v2.0

- Message **timestamps**, **Copy** on any message, **👍 / 👎**, and **Try again** to regenerate an answer.
- **Retry / Edit** buttons when the connection drops — no retyping.
- **Export** any answer to **.md / .docx / .pdf**; the filename defaults to the chat name (or set your own).
- **Rename** chats (the name sticks), **sort** by Recent / Oldest, and **search** across titles *and* message contents.
- **Backup & Restore** — export your whole history to a file and import it on another browser/machine.
- **Markdown tables** render cleanly.

---

## What it does

You send every prompt to `model: "auto"`. A tiny classifier model reads it and routes to the best local model for the job:

| Tier   | Example model      | Handles |
|--------|--------------------|---------|
| fast   | `llama3.2`         | greetings, quick facts, light edits — and the routing itself |
| coder  | `qwen3.6:35b-a3b`  | code, debugging, refactoring, shell, SQL, config |
| brain  | `gpt-oss:120b`     | reasoning, analysis, planning, math, long-form writing |
| vision | `qwen2.5vl:7b`     | images, screenshots, charts, diagrams, document photos |

*(Use whatever you have — match the tags to your own `ollama list`.)*

## Features

- **Smart routing** — `auto` picks the tier; a keyword pre-pass catches code, and images always go to vision. Force a tier with an `@brain` / `@coder` prefix.
- **OpenAI-compatible API** — drop-in `/v1/chat/completions`; point any OpenAI client at it.
- **Built-in web UI** — a single self-contained HTML page (streaming, per-tier badges, paste & drag-and-drop).
- **Vision** — attach or paste an image and it's read by the vision model.
- **Document reading** — drop in a Word, Excel, PowerPoint, or PDF; text is extracted server-side and summarized.
- **Conversation history** — saved server-side, so it follows you across devices; reopen or delete from the sidebar.
- **Full-text search** — search across the *contents* of all your saved chats, with snippets.
- **Per-user workspaces** — each browser gets its own private history (set your name in settings).
- **Concurrent chats** — each conversation streams independently; start one, open another, both keep running.
- **Message tools** — timestamp, Copy, 👍/👎, and **Try again** (regenerate) on every answer; Retry/Edit on connection errors.
- **Export answers** — download any answer as **.md / .docx / .pdf**; filename defaults to the chat name or your own.
- **Chat management** — rename chats, sort by Recent/Oldest, search titles + contents.
- **Backup & Restore** — export all chats to JSON and import on another browser/machine.
- **Knowledge base (RAG)** — private, on-box document Q&A with citations via `@kb` (local embeddings + on-device vector store).

## Requirements

- A host running [Ollama](https://ollama.com) (built/tuned on a Dell Pro Max with GB10 — NVIDIA Grace Blackwell, 128 GB unified memory).
- Python 3.10+ and `pip`.
- Disk for your models (~150 GB if you use the 120B brain tier).
- (Optional, for export) `pandoc` for .docx/.pdf and `wkhtmltopdf` for .pdf — `sudo apt install pandoc wkhtmltopdf`.
- (Optional, for the knowledge base) a local embedding model — `ollama pull nomic-embed-text`.

## Quick start

```bash
# 1. clone
git clone https://github.com/<your-username>/local-llm-router.git
cd local-llm-router

# 2. install deps
pip install -r requirements.txt --break-system-packages

# 3. pull models (swap for whatever you run)
ollama pull llama3.2
ollama pull qwen3.6:35b-a3b
ollama pull gpt-oss:120b
ollama pull qwen2.5vl:7b

# 4. point config.yaml at your model tags, then run
chmod +x start.sh
./start.sh
```

Open `http://<your-host-ip>:8000`, pick **auto**, and start chatting.

## Configuration

Everything lives in `config.yaml` — one block per tier (`base_url`, `model`, `description`) plus a `router` section (`classifier_route`, `default_route`, `auto_model_names`, `category_map`). Make each `model:` match a tag from `ollama list`.

## Tune Ollama for concurrency (optional, recommended)

So models stay warm and several users can be served at once:

```bash
sudo systemctl edit ollama
```
```ini
[Service]
Environment="OLLAMA_MAX_LOADED_MODELS=3"
Environment="OLLAMA_KEEP_ALIVE=-1"
Environment="OLLAMA_NUM_PARALLEL=2"
```
```bash
sudo systemctl restart ollama
```

## Run as a service

```bash
# edit User= and WorkingDirectory= in chatgb10.service first
sudo cp chatgb10.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now chatgb10
journalctl -u chatgb10 -f      # live logs
```

## API usage

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"auto","messages":[{"role":"user","content":"Refactor this Python loop"}]}'
```

Every response includes `X-Router-Tier` and `X-Router-Model` headers so you can see where it went. Smoke-test all tiers with:

```bash
python3 test_client.py
```

## Security note

ChatGB10 has **no built-in authentication**, and the per-user split is a workspace convenience (a client-supplied id), **not** security isolation. Run it on a trusted LAN or a private mesh like Tailscale — keep it **off the public internet**, and put a real auth proxy in front if you need true multi-tenant isolation.

## License

MIT © 2026 Jay-R Barrios. See [LICENSE](LICENSE).

---

Built on a Dell Pro Max with GB10. Full write-up on [LABDEMO](https://jayrbarrios.com).
