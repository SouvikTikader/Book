"""Thin Ollama wrapper. Returns None on any failure so the agent falls back to rule-based parsing (demo never breaks)."""
try:
    import ollama
except ImportError:  # ollama python package not installed
    ollama = None

CONFIG = {"use_llm": True, "model": "llama3.1:8b"}


def chat(system, user, json_mode=False):
    if not CONFIG["use_llm"] or ollama is None:
        return None
    try:
        kwargs = {"format": "json"} if json_mode else {}
        r = ollama.chat(model=CONFIG["model"],
                        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                        options={"temperature": 0}, **kwargs)
        return r["message"]["content"]
    except Exception:
        return None
