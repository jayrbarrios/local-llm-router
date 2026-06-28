"""Smoke test for the ChatGB10 router. Start the router first (./start.sh),
then run:  python3 test_client.py"""
import httpx

ROUTER = "http://127.0.0.1:8000/v1/chat/completions"
PROMPTS = [
    "good morning!",                                  # -> fast
    "What is the capital of France?",                 # -> fast
    "Fix this: def add(a,b) return a+b",              # -> coder
    "Refactor my Dockerfile to use multi-stage builds",# -> coder
    "Explain the trade-offs of microservices vs a monolith",  # -> brain
    "@brain just say hi",                             # -> brain (forced)
]

def check_stt(c):
    try:
        s = c.get("http://127.0.0.1:8000/stt/status").json()
        print(f"STT: available={s.get('available')} backend={s.get('backend')} "
              f"model={s.get('model')} device={s.get('device')}")
        if not s.get("available"):
            print("     (mic returns a clear 501 until a Whisper backend is installed — expected)")
    except Exception as e:
        print("STT status check failed:", e)


def main():
    with httpx.Client(timeout=None) as c:
        check_stt(c)
        for p in PROMPTS:
            r = c.post(ROUTER, json={"model": "auto", "stream": False,
                                     "messages": [{"role": "user", "content": p}]})
            tier = r.headers.get("X-Router-Tier", "?")
            model = r.headers.get("X-Router-Model", "?")
            try:
                ans = r.json()["choices"][0]["message"]["content"].strip().replace("\n", " ")
            except Exception:
                ans = f"[error {r.status_code}]"
            print(f"\n>>> {p}\n    tier={tier}  model={model}\n    {ans[:160]}")

if __name__ == "__main__":
    main()
