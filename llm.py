"""LLM wrapper supporting two backends: Ollama (local) and Gemini Flash (online).

Set `CONFIG["backend"]` to "ollama" or "gemini". Everything else in the app calls
`llm.chat(system, user, json_mode=...)` — the backend choice is transparent.
"""
import os
import re

try:
    import ollama
except ImportError:
    ollama = None

try:
    from google import genai
    from google.genai import types
    _HAS_GENAI = True
except ImportError:
    genai = None
    types = None
    _HAS_GENAI = False


CONFIG = {
    "use_llm": True,
    "backend": "ollama",                       # "ollama" | "gemini"
    "model": "llama3.1:8b",                    # Ollama model
    "gemini_model": "gemini-2.6-flash",        # or "gemini-2.6-flash" if available in your region
    "gemini_api_key": None,                    # or env GOOGLE_API_KEY / GEMINI_API_KEY
}


def _strip_fences(raw):
    if not raw:
        return raw
    s = raw.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[-1]
        s = s.rsplit("```", 1)[0]
    return s.strip()


def _gemini_key():
    return (CONFIG.get("gemini_api_key")
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GEMINI_API_KEY"))


_client = None


def _gemini_client():
    global _client
    key = _gemini_key()
    if _client is None and _HAS_GENAI and key:
        _client = genai.Client(api_key=key)
    return _client


def chat(system, user, json_mode=False):
    """Return the model's reply as a string, or None on any failure (caller must fall back)."""
    if not CONFIG["use_llm"]:
        return None
    if CONFIG["backend"] == "gemini":
        return _gemini_chat(system, user, json_mode)
    return _ollama_chat(system, user, json_mode)


# ------------------------------------------------------------------ backends
def _ollama_chat(system, user, json_mode=False):
    if ollama is None:
        return None
    try:
        kwargs = {"format": "json"} if json_mode else {}
        r = ollama.chat(
            model=CONFIG["model"],
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            options={"temperature": 0},
            **kwargs,
        )
        return _strip_fences(r["message"]["content"])
    except Exception:
        return None


def _gemini_chat(system, user, json_mode=False):
    client = _gemini_client()
    if client is None:
        return None
    try:
        cfg = types.GenerateContentConfig(
            system_instruction=system,
            temperature=0,
            response_mime_type="application/json" if json_mode else "text/plain",
        )
        r = client.models.generate_content(
            model=CONFIG["gemini_model"],
            contents=user,
            config=cfg,
        )
        return _strip_fences(r.text)
    except Exception:
        return None