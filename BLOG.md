# Build Your Own Local ChatGPT — with a Smart Model Router

*How I turned a desktop AI box into a private, four-tier assistant that picks the right model for every prompt — no cloud, no API bills, no data leaving the building.*

---

## The problem with "just run a local model"

Running a local LLM is easy now. `ollama run llama3.2` and you're chatting in thirty seconds. The trouble starts the moment you get serious about it, because **no single local model is good at everything within a reasonable speed and memory budget**:

- The *big* models (100B+) reason and write beautifully but are slow and eat memory for breakfast.
- The *small* models are fast and cheap but fumble anything that needs real thinking.
- *Coder* models run circles around generalists on code — and waste their talent on "good morning."
- *Vision* models can read a screenshot or a chart, but they're a separate model entirely.

The frontier assistants you already use solve this quietly. When you talk to a hosted assistant, you're usually talking to a *family* of models at different sizes, and something decides which one should answer. That "something" is a **router**.

So I built one for my own hardware. One endpoint, one chat window, and a lightweight router that reads each message and sends it to the right local model. This post walks through the whole thing. The full code is on GitHub (link at the bottom) under an MIT license — clone it and point it at your own Ollama.

## What we're building

```
                          ┌──────────────────────────────────┐
   any OpenAI client ───▶ │  ChatGB10 Router  (:8000)          │
   browser UI at  /       │  intent classifier + keyword pass │
   (model: "auto")        └───────────────┬───────────────────┘
                                          │  OpenAI-compatible HTTP
                                          ▼
                          ┌──────────────────────────────────┐
                          │  Ollama  (:11434)                 │
                          │   fast · coder · brain · vision   │
                          └──────────────────────────────────┘
```

Four tiers, all served locally by [Ollama](https://ollama.com):

| Tier   | Job                                                | Model I used        |
|--------|----------------------------------------------------|---------------------|
| fast   | greetings, quick facts, light edits (+ classifier) | `llama3.2`          |
| coder  | code, debugging, refactoring, shell, SQL           | `qwen3.6:35b-a3b`   |
| brain  | reasoning, analysis, planning, long-form           | `gpt-oss:120b`      |
| vision | images, screenshots, charts, diagrams              | `qwen2.5vl:7b`      |

On top of that: an OpenAI-compatible API (so any existing client works), a built-in web chat UI with streaming and a conversation-history sidebar, and file attachments — Word, Excel, PowerPoint, PDF, and images.

## The hardware (and why it shapes the design)

I ran this on a unified-memory AI desktop with 128 GB of shared CPU/GPU memory. That's a lot of capacity — enough to hold a 120B model — but here's the catch that drives every design decision: **on these unified-memory machines, memory *bandwidth*, not capacity, is the bottleneck.** You can *load* a huge dense model, but token generation crawls because every token has to stream the whole weight set through limited bandwidth.

The fix is **Mixture-of-Experts (MoE)** models. An MoE model has lots of total parameters but only activates a few billion per token, so it generates fast while staying smart. That's why my coder tier is a 35B MoE that only activates ~3B params per token rather than a dense 32B — same intelligence ballpark, multiples faster on this hardware.

You don't need a 128 GB box to follow along. Swap `gpt-oss:120b` for `gpt-oss:20b` and `qwen3.6:35b-a3b` for `qwen2.5-coder:7b` and the whole thing runs on a far more modest machine. Only `config.yaml` changes.

## Step 1 — Models

Install Ollama, then pull a model for each tier:

```bash
ollama pull llama3.2            # fast tier + classifier
ollama pull qwen3.6:35b-a3b     # coder (MoE — fast on unified memory)
ollama pull gpt-oss:120b        # brain (or gpt-oss:20b on smaller machines)
ollama pull qwen2.5vl:7b        # vision
```

A note on vision: as of this writing, some of the newest vision GGUFs don't load cleanly in Ollama yet (multimodal projector issues), so I stuck with the proven `qwen2.5vl:7b`, which is excellent at screenshots, charts, and document images.

## Step 2 — The router

The router is a small [FastAPI](https://fastapi.tiangolo.com) app that speaks the OpenAI API. Because everything talks plain OpenAI HTTP, the router doesn't care what's behind it — Ollama, llama.cpp, anything. The interesting part is `decide_route`, a cheap three-stage decision:

```python
async def decide_route(text: str) -> str:
    # 1. explicit "@brain ..." override
    m = OVERRIDE_RE.match(text)
    if m and m.group(1).lower() in ROUTES:
        return m.group(1).lower()

    # 2. high-precision keyword catch for obvious code
    if keyword_route(text):
        return CATEGORY_MAP["code"]          # -> coder

    # 3. let the small model classify the rest
    cat = await classify(text)               # "simple" | "code" | "reasoning"
    return CATEGORY_MAP.get(cat, DEFAULT_ROUTE)
```

Stage 2 is a regex that catches code fences, file extensions, SQL keywords, and words like `traceback` or `refactor` — these are unambiguous, so we skip the classifier entirely. Stage 3 hands anything else to the **fast** model with a one-word-label system prompt:

```python
system = (
    "You are a routing classifier. Reply with EXACTLY ONE word:\n"
    "- simple: greetings, quick facts, casual chat, simple rewrites\n"
    "- code: writing/debugging code, shell, SQL, config, stack traces\n"
    "- reasoning: analysis, planning, math, long-form, anything complex\n"
    "Answer with only one of: simple, code, reasoning"
)
```

Because the classifier *is* the small fast model — and it only generates one token — the routing overhead is negligible.

Images are handled structurally: if the current message contains an image, it goes straight to **vision**, no classifier needed.

```python
def has_image(messages):
    m = last_user_msg(messages)         # only the CURRENT turn matters
    c = m.get("content") if m else None
    return isinstance(c, list) and any(
        p.get("type") == "image_url" for p in c if isinstance(p, dict))
```

That "only the current turn" detail is a lesson I learned the hard way — see the gotchas below.

The handler then forwards the request to the chosen tier and streams the response straight back, tagging it with `X-Router-Tier` and `X-Router-Model` headers so the UI can show which model answered.

## Step 3 — Reading documents

LLMs can't parse a binary `.docx`. So the router exposes an `/extract` endpoint that pulls plain text out of uploads server-side, using the obvious libraries — `python-docx`, `openpyxl`, `python-pptx`, `pypdf`:

```python
EXTRACTORS = {
    ".docx": extract_docx, ".xlsx": extract_xlsx, ".pptx": extract_pptx,
    ".pdf": extract_pdf, ".txt": extract_text, ".csv": extract_text,
}
```

The browser uploads a file, gets back the extracted text, and prepends it to your message as context. Any *text* tier can then "read" the document — no special model needed. (Images are different: those carry their pixels to the vision tier as base64.)

## Step 4 — The web UI

The router serves a single self-contained HTML page at `/` — a dark, ChatGPT-style chat with streaming, a conversation-history sidebar, file/image attachment chips, and a colored badge on each answer showing which tier handled it. Because the page is served by the router itself, it's same-origin, so there are no CORS headaches.

Conversation history is stored **on the router** (a simple JSON file), not in the browser. That was a deliberate choice: I open this thing from my laptop, my desktop, and the AI box itself, and I want the same history everywhere — exactly like a hosted assistant.

## Step 5 — Make it an appliance

Running it by hand is fine for testing, but I want it *always there*. A tiny systemd unit does that:

```ini
[Service]
User=YOUR_USERNAME
WorkingDirectory=/path/to/local-llm-router
ExecStart=/usr/bin/python3 -m uvicorn router:app --host 0.0.0.0 --port 8000
Restart=on-failure
```

```bash
sudo systemctl enable --now chatgb10
```

Now the whole stack comes up on boot. Power on the box, and a private four-tier assistant is just *there* on the network.

## Hard-won lessons

A few things that cost me time so they don't cost you yours:

**1. On unified-memory hardware, choose MoE models.** A dense 32B model felt sluggish; a 35B MoE that activates ~3B params per token felt snappy. Capacity wasn't my limit — bandwidth was.

**2. Route on the user's *intent*, not the document's *content*.** My first version classified the full message, including the injected document text. Attach a battery-spec doc full of BIOS and firmware jargon, ask "summarize this," and the router decided it was a *coding* task. The fix: strip the attached-document text before classifying, so "summarize this" routes to **brain** regardless of what the file contains. The model still receives the full document — only the routing decision ignores it.

**3. Only check the *current* turn for images.** Once an image was anywhere in the conversation history, every subsequent text message kept getting pinned to the vision tier. Checking only the latest user message fixes it.

**4. Keep-alive is a memory trade.** `OLLAMA_KEEP_ALIVE=-1` pins models in memory so responses are instant — wonderful, until four warm tiers crowd a full machine. If you need headroom for other work, drop it to a timeout like `30m` and let idle tiers release their memory.

**5. Ollama is the fast path to "working."** You can squeeze more raw throughput later with llama.cpp and quantized formats tuned to your accelerator, but Ollama got the whole thing running in an evening, and the router doesn't change when you migrate — you just repoint a URL.

## The payoff

The result is genuinely useful. "Good morning" comes back instantly from the small model. "Refactor my Dockerfile" gets the coder. "Walk me through a go-to-market plan" gets the 120B brain. Drop in a screenshot and the vision model reads it. Attach a Word doc and ask for a summary and the big model gives you a structured breakdown — all on my own hardware, nothing leaving the building, and zero per-token cost.

A model router isn't magic. It's a hundred lines of routing logic in front of `ollama`. But that small layer is the difference between "I have some models installed" and "I have a private assistant that picks the right brain for the job."

## Get the code

The full project — router, extraction, vision, web UI, and systemd service — is on GitHub:

> **`github.com/<your-username>/local-llm-router`**

Clone it, edit `config.yaml` to match your models, `./start.sh`, and you're running. PRs and ideas welcome.

*One last note: keep it on your LAN or a private mesh like Tailscale — there's no auth on the router, so don't expose it to the open internet.*
