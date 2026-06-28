"""
ChatGB10 Router — an OpenAI-compatible local model router for the GB10.

One endpoint (/v1/chat/completions). With model="auto" the router
picks the tier:

    fast    -> llama3        (greetings, quick facts, simple edits) + classifier
    coder   -> qwen3.6 MoE   (code, debugging, refactoring, shell, SQL)
    brain   -> GPT-OSS-120B  (hard reasoning, analysis, long-form)

Decision = keyword pre-pass (catches obvious code) -> small-LLM classifier.
Bypass routing: set model to "fast"|"coder"|"brain", or prefix "@brain ...".

Also serves a built-in web chat UI at  /  (so you can just open the URL).
"""

import os
import re
import json
import time
import uuid
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import yaml
import httpx
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
import uvicorn

from extractors import EXTRACTORS

try:
    import rag
    RAG_OK = True
    _RAG_ERR = ""
except Exception as _e:  # numpy not installed
    rag = None
    RAG_OK = False
    _RAG_ERR = str(_e)

try:
    import stt
    STT_OK = True
    _STT_ERR = ""
except Exception as _e:  # module present but a dependency is missing, etc.
    stt = None
    STT_OK = False
    _STT_ERR = str(_e)


# --------------------------------------------------------------------------- #
# Config + logging
# --------------------------------------------------------------------------- #
HERE = Path(__file__).parent
CONFIG_PATH = os.environ.get("CHATGB10_CONFIG", str(HERE / "config.yaml"))
with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

ROUTES = CFG["routes"]
RCFG = CFG["router"]
DEFAULT_ROUTE = RCFG["default_route"]
CLASSIFIER_ROUTE = RCFG["classifier_route"]
CATEGORY_MAP = RCFG["category_map"]
AUTO_NAMES = set(RCFG.get("auto_model_names", ["auto", "chatgb10"]))
PHRASE_KEYWORDS = [k.lower() for k in RCFG.get("code_keywords", [])]
MAX_EXTRACT_CHARS = int(os.environ.get("CHATGB10_MAX_EXTRACT_CHARS", "40000"))

# RAG (knowledge base) settings
_RAGCFG = CFG.get("rag", {})
EMBED_MODEL = os.environ.get("CHATGB10_EMBED_MODEL", _RAGCFG.get("embed_model", "nomic-embed-text"))
RAG_CHUNK = int(_RAGCFG.get("chunk_chars", 1200))
RAG_OVERLAP = int(_RAGCFG.get("chunk_overlap", 200))
RAG_TOPK = int(_RAGCFG.get("top_k", 5))
RAG_ANSWER_ROUTE = _RAGCFG.get("answer_route", DEFAULT_ROUTE)
EMBED_URL = ROUTES[CLASSIFIER_ROUTE]["base_url"].rstrip("/") + "/embeddings"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-5s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("chatgb10")

client = httpx.AsyncClient(timeout=httpx.Timeout(None, connect=10.0))
app = FastAPI(title="ChatGB10 Router")

# Allow a browser page (even opened from file://) to call the API directly,
# and let JS read the routing-decision headers.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Router-Tier", "X-Router-Model"],
)

# --------------------------------------------------------------------------- #
# Routing helpers
# --------------------------------------------------------------------------- #
CODE_RE = re.compile(
    r"```"
    r"|\b(def|class|import|function|const|let|var|public|private|#include"
    r"|SELECT|INSERT|UPDATE|DELETE|CREATE TABLE)\b"
    r"|</?[a-zA-Z][\w-]*>"
    r"|\.(py|js|ts|tsx|jsx|java|cpp|cc|c|h|go|rs|rb|php|sql|sh|bash|ya?ml|json|html?|css|toml)\b",
    re.IGNORECASE,
)
OVERRIDE_RE = re.compile(r"^\s*@(\w+)\s+", re.IGNORECASE)
# Matches the "[Attached file: name]\n\"\"\"\n...body...\n\"\"\"\n\n" block the UI
# prepends for document attachments, so routing ignores the file's contents.
DOC_WRAPPER_RE = re.compile(r'\[Attached file: [^\]]*\]\n"""\n[\s\S]*?\n"""\n\n')


def last_user_text(messages: list) -> str:
    for msg in reversed(messages or []):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [p.get("text", "") for p in content
                     if isinstance(p, dict) and p.get("type") == "text"]
            return " ".join(parts)
    return ""


def _last_user_msg(messages: list):
    for m in reversed(messages or []):
        if m.get("role") == "user":
            return m
    return None


def has_image(messages: list) -> bool:
    # Only the CURRENT user turn matters — an old image earlier in the
    # conversation should not pin every later text message to the vision tier.
    m = _last_user_msg(messages)
    if not m:
        return False
    c = m.get("content")
    return isinstance(c, list) and any(
        isinstance(p, dict) and p.get("type") == "image_url" for p in c
    )


def routing_text(messages: list) -> str:
    """Text used ONLY for routing. Strips injected document bodies so the route
    is chosen from the user's instruction, not the attached file's contents.
    (The model still receives the full document text.)"""
    return DOC_WRAPPER_RE.sub("", last_user_text(messages)).strip()


def keyword_route(text: str):
    if CODE_RE.search(text):
        return "code"
    lowered = text.lower()
    if any(k in lowered for k in PHRASE_KEYWORDS):
        return "code"
    return None


async def classify(text: str):
    route_cfg = ROUTES[CLASSIFIER_ROUTE]
    system = (
        "You are a routing classifier. Read the user's message and reply with "
        "EXACTLY ONE word, the category, and nothing else.\n"
        "Categories:\n"
        "- simple: greetings, quick facts, short lookups, casual chat, simple rewrites\n"
        "- code: writing, debugging, reviewing or explaining code; shell, SQL, config, stack traces\n"
        "- reasoning: analysis, planning, math, multi-step reasoning, long-form writing, anything complex\n"
        "Answer with only one of: simple, code, reasoning"
    )
    body = {
        "model": route_cfg["model"],
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": text[:4000]},
        ],
        "max_tokens": 4,
        "temperature": 0.0,
        "stream": False,
    }
    try:
        url = route_cfg["base_url"].rstrip("/") + "/chat/completions"
        r = await client.post(url, json=body)
        r.raise_for_status()
        out = r.json()["choices"][0]["message"]["content"].lower()
        for cat in ("code", "reasoning", "simple"):
            if cat in out:
                return cat
    except Exception as e:  # noqa: BLE001
        log.warning("classifier failed (%s); using default route", e)
    return None


async def decide_route(text: str) -> str:
    m = OVERRIDE_RE.match(text)
    if m and m.group(1).lower() in ROUTES:
        return m.group(1).lower()
    kw = keyword_route(text)
    if kw:
        return CATEGORY_MAP.get(kw, DEFAULT_ROUTE)
    cat = await classify(text)
    if cat:
        return CATEGORY_MAP.get(cat, DEFAULT_ROUTE)
    return DEFAULT_ROUTE


def strip_override(messages: list):
    for msg in reversed(messages):
        if msg.get("role") == "user" and isinstance(msg.get("content"), str):
            msg["content"] = OVERRIDE_RE.sub("", msg["content"], count=1)
            break
    return messages


# --------------------------------------------------------------------------- #
# OpenAI-compatible endpoints
# --------------------------------------------------------------------------- #
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    requested = (body.get("model") or "auto").lower()
    messages = body.get("messages", [])
    text = routing_text(messages)

    if requested in ROUTES:
        route = requested
    elif requested in AUTO_NAMES:
        route = await decide_route(text)
    else:
        route = DEFAULT_ROUTE

    # An image can only be read by the vision tier, regardless of routing.
    if has_image(messages) and "vision" in ROUTES:
        route = "vision"

    body["messages"] = strip_override(messages)
    target = ROUTES[route]
    body["model"] = target["model"]
    url = target["base_url"].rstrip("/") + "/chat/completions"
    stream = bool(body.get("stream", False))

    preview = text.replace("\n", " ")[:60]
    log.info("route=%-6s model=%-22s stream=%s  | %s", route, target["model"], stream, preview)

    headers = {"X-Router-Tier": route, "X-Router-Model": target["model"]}

    if stream:
        async def gen():
            async with client.stream("POST", url, json=body) as resp:
                async for chunk in resp.aiter_raw():
                    yield chunk
        return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)

    r = await client.post(url, json=body)
    return JSONResponse(r.json(), status_code=r.status_code, headers=headers)


@app.get("/v1/models")
async def list_models():
    data = [{"id": "auto", "object": "model", "owned_by": "chatgb10"}]
    for name, cfg in ROUTES.items():
        data.append({"id": name, "object": "model", "owned_by": "chatgb10",
                     "description": cfg.get("description", "")})
    return {"object": "list", "data": data}


@app.get("/health")
async def health():
    return {"status": "ok", "routes": list(ROUTES.keys()), "default": DEFAULT_ROUTE}


@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    """Extract plain text from an uploaded document (docx/xlsx/pptx/pdf/text)."""
    name = file.filename or "file"
    ext = ("." + name.rsplit(".", 1)[-1].lower()) if "." in name else ""
    fn = EXTRACTORS.get(ext)
    if not fn:
        return JSONResponse(
            {"error": f"Unsupported file type: {ext or 'unknown'}", "filename": name},
            status_code=415,
        )
    data = await file.read()
    try:
        text = fn(data)
    except Exception as e:  # noqa: BLE001
        log.warning("extract failed for %s: %s", name, e)
        return JSONResponse(
            {"error": f"Could not read {name}: {e}", "filename": name},
            status_code=500,
        )
    truncated = len(text) > MAX_EXTRACT_CHARS
    if truncated:
        text = text[:MAX_EXTRACT_CHARS]
    log.info("extract  %-22s ext=%-5s chars=%d%s", name[:22], ext, len(text),
             " (truncated)" if truncated else "")
    return {"filename": name, "ext": ext, "chars": len(text),
            "truncated": truncated, "text": text}


# --------------------------------------------------------------------------- #
# Speech-to-text (local Whisper on the GB10) — used by the 🎤 mic button
# --------------------------------------------------------------------------- #
@app.post("/stt")
async def speech_to_text(file: UploadFile = File(...)):
    """Transcribe an uploaded audio clip locally. Returns {"text": ...}."""
    if not STT_OK or stt is None:
        return JSONResponse(
            {"error": f"STT module not loaded: {_STT_ERR}"}, status_code=501)
    if not stt.available():
        s = stt.status()
        return JSONResponse(
            {"error": s.get("error") or "No Whisper backend installed on the GB10."},
            status_code=501)
    data = await file.read()
    if not data:
        return JSONResponse({"error": "empty audio"}, status_code=400)
    try:
        # Whisper is blocking (and the first call also loads the model) — keep
        # it off the event loop so streaming chats stay responsive.
        text = await run_in_threadpool(stt.transcribe, data, file.filename or "audio.webm")
    except Exception as e:  # noqa: BLE001
        log.warning("stt failed: %s", e)
        return JSONResponse({"error": f"transcription failed: {e}"}, status_code=500)
    log.info("stt      %7d bytes -> %d chars", len(data), len(text))
    return {"text": text}


@app.get("/stt/status")
async def stt_status():
    if not STT_OK or stt is None:
        return {"available": False, "error": f"module not loaded: {_STT_ERR}"}
    return stt.status()


# --------------------------------------------------------------------------- #
# Chat history store (JSON file on the router — shared across all your devices)
# --------------------------------------------------------------------------- #
CHATS_FILE = HERE / "chats.json"


def _user_id(request: Request) -> str:
    """Identity for per-user private histories. Behind Cloudflare Access the
    verified SSO email (Cf-Access-Authenticated-User-Email) is trusted as the
    identity — provided this service is reachable ONLY via the tunnel (bind to
    127.0.0.1). Falls back to an X-User-Id header (LAN/workspace use), then a
    shared bucket."""
    return (request.headers.get("Cf-Access-Authenticated-User-Email")
            or request.headers.get("X-User-Id")
            or "shared")


def _load_chats() -> dict:
    """Returns {user_id: {chat_id: chat}}. Migrates the old flat format."""
    if not CHATS_FILE.exists():
        return {}
    try:
        d = json.loads(CHATS_FILE.read_text())
    except Exception:  # noqa: BLE001
        return {}
    # old format was flat {chat_id: {..."messages"...}} -> bucket under "shared"
    if d and all(isinstance(v, dict) and "messages" in v for v in d.values()):
        d = {"shared": d}
    return d


def _save_chats(d: dict):
    CHATS_FILE.write_text(json.dumps(d))


@app.get("/chats")
async def list_chats(request: Request):
    d = _load_chats().get(_user_id(request), {})
    items = [{"id": k, "title": v.get("title", "New chat"),
              "updated": v.get("updated", 0)} for k, v in d.items()]
    items.sort(key=lambda x: x["updated"], reverse=True)
    return {"chats": items}


@app.get("/chats/export")
async def export_chats(request: Request):
    """Download all of this user's chats (with messages) as a portable backup."""
    d = _load_chats().get(_user_id(request), {})
    return {"chats": list(d.values()), "exported": time.time()}


@app.post("/chats/import")
async def import_chats(request: Request):
    """Merge a backup file's chats into this user's workspace (e.g. on another browser)."""
    body = await request.json()
    incoming = body.get("chats", []) if isinstance(body, dict) else body
    if not isinstance(incoming, list):
        return JSONResponse({"error": "expected a list of chats"}, status_code=400)
    uid = _user_id(request)
    alld = _load_chats()
    user = alld.setdefault(uid, {})
    n = 0
    for chat in incoming:
        if not isinstance(chat, dict):
            continue
        cid = chat.get("id")
        if not cid:
            continue
        user[cid] = {
            "id": cid,
            "title": (chat.get("title") or "New chat")[:80],
            "messages": chat.get("messages", []),
            "updated": chat.get("updated", time.time()),
            "custom_title": bool(chat.get("custom_title", False)),
        }
        n += 1
    _save_chats(alld)
    return {"ok": True, "imported": n}


@app.get("/chats/{cid}")
async def get_chat(cid: str, request: Request):
    d = _load_chats().get(_user_id(request), {})
    if cid not in d:
        return JSONResponse({"error": "not found"}, status_code=404)
    return d[cid]


@app.post("/chats")
async def save_chat(request: Request):
    body = await request.json()
    uid = _user_id(request)
    cid = body.get("id") or uuid.uuid4().hex[:12]
    alld = _load_chats()
    user = alld.setdefault(uid, {})
    prev = user.get(cid) or {}
    # a manually renamed chat keeps its custom title even as new messages arrive
    title = prev["title"] if prev.get("custom_title") else (body.get("title") or "New chat")[:80]
    user[cid] = {
        "id": cid,
        "title": title,
        "messages": body.get("messages", []),
        "updated": time.time(),
        "custom_title": prev.get("custom_title", False),
    }
    _save_chats(alld)
    return {"id": cid}


@app.delete("/chats/{cid}")
async def delete_chat(cid: str, request: Request):
    uid = _user_id(request)
    alld = _load_chats()
    if uid in alld:
        alld[uid].pop(cid, None)
        _save_chats(alld)
    return {"ok": True}


@app.post("/chats/{cid}/title")
async def rename_chat(cid: str, request: Request):
    body = await request.json()
    title = (body.get("title") or "").strip()[:80]
    if not title:
        return JSONResponse({"error": "empty title"}, status_code=400)
    uid = _user_id(request)
    alld = _load_chats()
    user = alld.get(uid, {})
    if cid not in user:
        return JSONResponse({"error": "not found"}, status_code=404)
    user[cid]["title"] = title
    user[cid]["custom_title"] = True
    _save_chats(alld)
    return {"ok": True, "title": title}


@app.get("/search")
async def search_chats(q: str, request: Request):
    """Full-text search across the current user's saved chat contents."""
    needle = (q or "").strip().lower()
    if not needle:
        return {"results": []}
    d = _load_chats().get(_user_id(request), {})
    results = []
    for cid, chat in d.items():
        parts = []
        for m in chat.get("messages", []):
            c = m.get("content")
            if isinstance(c, str):
                parts.append(c)
            elif isinstance(c, list):
                parts += [p.get("text", "") for p in c
                          if isinstance(p, dict) and p.get("type") == "text"]
        hay = "\n".join([chat.get("title", "")] + parts)
        i = hay.lower().find(needle)
        if i != -1:
            s = max(0, i - 40)
            e = min(len(hay), i + len(needle) + 60)
            snippet = ("\u2026" if s > 0 else "") + hay[s:e].replace("\n", " ") + ("\u2026" if e < len(hay) else "")
            results.append({"id": cid, "title": chat.get("title", "New chat"),
                            "updated": chat.get("updated", 0), "snippet": snippet})
    results.sort(key=lambda x: x["updated"], reverse=True)
    return {"results": results}


@app.post("/export")
async def export_doc(request: Request):
    """Convert an answer (Markdown) to .md / .docx / .pdf via pandoc and return it
    as a download. .md needs nothing; .docx needs pandoc; .pdf needs pandoc + an engine."""
    body = await request.json()
    fmt = (body.get("format") or "md").lower()
    md = body.get("markdown") or ""
    title = re.sub(r"[^\w\- ]", "", body.get("title") or "chatgb10-export")[:60] or "chatgb10-export"
    if fmt not in ("md", "docx", "pdf"):
        return JSONResponse({"error": "unsupported format"}, status_code=400)
    if fmt == "md":
        return Response(content=md.encode("utf-8"), media_type="text/markdown",
                        headers={"Content-Disposition": f'attachment; filename="{title}.md"'})
    if not shutil.which("pandoc"):
        return JSONResponse({"error": "pandoc is not installed on the server. Run: sudo apt install pandoc"},
                            status_code=501)
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "in.md")
        out = os.path.join(d, "out." + fmt)
        with open(src, "w", encoding="utf-8") as f:
            f.write(md)
        cmd = ["pandoc", src, "-o", out]
        if fmt == "pdf":
            for eng in ("wkhtmltopdf", "weasyprint"):
                if shutil.which(eng):
                    cmd += ["--pdf-engine", eng]
                    break
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=60)
        except Exception as e:  # noqa: BLE001
            return JSONResponse({"error": f"pandoc failed: {e}"}, status_code=500)
        if r.returncode != 0 or not os.path.exists(out):
            err = r.stderr.decode("utf-8", "replace")[:400] or "pandoc error"
            if fmt == "pdf":
                err += " (PDF needs an engine: sudo apt install wkhtmltopdf)"
            return JSONResponse({"error": err}, status_code=500)
        data = open(out, "rb").read()
    media = {"docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
             "pdf": "application/pdf"}[fmt]
    return Response(content=data, media_type=media,
                    headers={"Content-Disposition": f'attachment; filename="{title}.{fmt}"'})


@app.post("/rag/ingest")
async def rag_ingest(request: Request, file: UploadFile = File(...)):
    """Add a document to the user's private, on-box knowledge base."""
    if not RAG_OK:
        return JSONResponse({"error": f"RAG unavailable: {_RAG_ERR}. "
                             "Install: pip install numpy --break-system-packages"}, status_code=501)
    uid = _user_id(request)
    name = file.filename or "document"
    ext = ("." + name.rsplit(".", 1)[-1].lower()) if "." in name else ""
    fn = EXTRACTORS.get(ext)
    if not fn:
        return JSONResponse({"error": f"Unsupported file type: {ext or 'unknown'}"}, status_code=415)
    data = await file.read()
    try:
        text = fn(data)
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": f"Could not read {name}: {e}"}, status_code=500)
    try:
        res = await rag.ingest(client, EMBED_URL, EMBED_MODEL, uid, name, text, RAG_CHUNK, RAG_OVERLAP)
    except Exception as e:  # noqa: BLE001
        log.warning("rag ingest failed for %s: %s", name, e)
        return JSONResponse({"error": f"Embedding failed: {e}. Is '{EMBED_MODEL}' pulled? "
                             f"(ollama pull {EMBED_MODEL})"}, status_code=502)
    log.info("rag      ingest %-20s chunks=%d", name[:20], res["chunks"])
    return res


@app.get("/rag/status")
async def rag_status(request: Request):
    if not RAG_OK:
        return {"documents": [], "chunks": 0, "embed_model": EMBED_MODEL, "available": False}
    docs = rag.store_for(_user_id(request)).docs()
    return {"documents": docs, "chunks": sum(d["chunks"] for d in docs),
            "embed_model": EMBED_MODEL, "available": True}


@app.post("/rag/clear")
async def rag_clear(request: Request):
    if RAG_OK:
        rag.store_for(_user_id(request)).clear()
    return {"ok": True}


@app.post("/rag/ask")
async def rag_ask(request: Request):
    """Answer a question from the knowledge base: retrieve -> augment -> generate."""
    if not RAG_OK:
        return JSONResponse({"error": f"RAG unavailable: {_RAG_ERR}."}, status_code=501)
    uid = _user_id(request)
    body = await request.json()
    question = (body.get("question") or "").strip()
    if not question:
        return JSONResponse({"error": "empty question"}, status_code=400)
    try:
        hits = await rag.retrieve(client, EMBED_URL, EMBED_MODEL, uid, question, RAG_TOPK)
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": f"Retrieval failed: {e}. Is '{EMBED_MODEL}' pulled? "
                             f"(ollama pull {EMBED_MODEL})"}, status_code=502)
    if not hits:
        return JSONResponse({"error": "Knowledge base is empty. Add documents in "
                             "Settings \u2192 Knowledge Base."}, status_code=400)
    context, sources = rag.build_context(hits)
    system = ("You are a helpful assistant. Answer the question using ONLY the context below. "
              "If the answer is not in the context, say you don't have that information. "
              "Cite source files in brackets like [filename].\n\nContext:\n" + context)
    target = ROUTES.get(RAG_ANSWER_ROUTE) or ROUTES[DEFAULT_ROUTE]
    url = target["base_url"].rstrip("/") + "/chat/completions"
    payload = {"model": target["model"], "stream": False,
               "messages": [{"role": "system", "content": system},
                            {"role": "user", "content": question}]}
    try:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        answer = r.json()["choices"][0]["message"]["content"]
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": f"Answer generation failed: {e}"}, status_code=502)
    log.info("rag      ask    %-20s sources=%d", question[:20], len(sources))
    return {"answer": answer, "sources": sources,
            "chunks": [{"source": c["source"], "score": round(s, 3)} for c, s in hits]}


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the built-in chat UI."""
    page = HERE / "chat.html"
    if page.exists():
        return HTMLResponse(page.read_text())
    return HTMLResponse("<h1>ChatGB10 Router is running.</h1>"
                        "<p>Drop chat.html next to router.py to get the chat UI.</p>")


if __name__ == "__main__":
    host = os.environ.get("CHATGB10_HOST", "0.0.0.0")
    port = int(os.environ.get("CHATGB10_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
