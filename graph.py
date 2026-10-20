"""LangGraph workflow.

Deterministic graph (`build`) handles dates, filtering, atomic booking.
Optional ReAct graph (`build_react`) lets the LLM choose tools — genuine agent mode.
"""
import difflib
import json
import re
from datetime import datetime
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

import agents
import db
import llm
import memory

SPECIALTIES = ["cardiology", "neurology", "dermatology", "orthopedics", "pediatrics", "general"]
SYN = {}
for _spec, _words in {
    "cardiology": "cardiology cardiologist cardiologists cardiac heart",
    "neurology": "neurology neurologist neurologists brain",
    "dermatology": "dermatology dermatologist dermatologists skin",
    "orthopedics": "orthopedics orthopedic orthopaedic orthopedist bone bones",
    "pediatrics": "pediatrics pediatric paediatrics paediatric pediatrician paediatrician pediatrist child children kids",
    "general": "general physician",
}.items():
    for _w in _words.split():
        SYN[_w] = _spec

MEDICAL_ADVICE = re.compile(
    r"\b(do i have|diagnos\w*|what medicine|which medicine|prescri\w*|dosage|am i having|"
    r"is it a heart attack|treatment for)\b", re.I,
)

# If none of these appear, we treat the message as small talk (greeting / thanks / etc.).
BOOKING_HINTS = re.compile(
    r"\b(book|appointment|schedule|reserve|see|visit|consult|doctor|dr\.?|"
    r"cardiolog\w*|neurolog\w*|dermatolog\w*|orthoped\w*|pediatric\w*|physician|"
    r"chest pain|headache|rash|fever|pain|symptom\w*)\b", re.I,
)


class S(TypedDict, total=False):
    user_text: str
    patient_id: int
    stage: str
    intent: dict
    missing: list
    bids: list
    outcome: str
    attempts: int
    conflict_used: bool
    simulate_conflict: bool
    window: int
    appointment: dict
    reply: str
    trace: list
    acl: list


def T(s, msg):
    return s.get("trace", []) + [msg]


# ------------------------------------------------------------------ parsing (code, not LLM)
def find_specialty(text):
    toks = [t for t in re.findall(r"[a-z]+", text.lower()) if len(t) >= 4]
    for t in toks:
        if t in SYN:
            return SYN[t]
    for t in toks:
        close = difflib.get_close_matches(t, SYN.keys(), n=1, cutoff=0.78)
        if close:
            return SYN[close[0]]
    return None


def rule_parse(text):
    t = text.lower()
    spec = find_specialty(text)
    m = re.search(r"(?<![\d.:])(\d{1,2})(?:[:.](\d\d))?\s*(am|pm)\b", t)
    after = None
    if m:
        h = int(m.group(1)) % 12 + (12 if m.group(3) == "pm" else 0)
        after = f"{h:02d}:{m.group(2) or '00'}"
    off = 2 if "day after" in t else 0 if "today" in t else 1
    return {"specialty": spec, "after": after, "offset": off, "symptoms": None if spec else text}


def parse_intent(text):
    intent = rule_parse(text)
    raw = llm.chat(
        'Extract a hospital booking request. Reply ONLY JSON with keys: specialty (one of '
        '"cardiology","neurology","dermatology","orthopedics","pediatrics","general" or null), '
        'after_time ("HH:MM" 24h or null), symptoms (string or null).',
        text, json_mode=True,
    )
    src = "rules"
    if raw:
        try:
            j = json.loads(raw)
            if not intent["specialty"] and j.get("specialty") in SPECIALTIES:
                intent["specialty"], intent["symptoms"] = j["specialty"], None
            if not intent["after"] and re.fullmatch(r"\d\d:\d\d", str(j.get("after_time") or "")):
                intent["after"] = j["after_time"]
            if not intent["specialty"] and j.get("symptoms"):
                intent["symptoms"] = j["symptoms"]
            src = "LLM + rules"
        except (ValueError, AttributeError):
            pass
    intent["after"] = intent["after"] or "09:00"
    intent["day"] = db.day_offset(intent["offset"])
    if intent["offset"] == 0:
        now = datetime.now().strftime("%H:%M")
        intent["after"] = max(intent["after"], now)
    intent["source"] = src
    return intent


# ------------------------------------------------------------------ chitchat
def _fallback_chitchat(text):
    t = text.lower().strip()
    if any(w in t for w in ("thank", "thanks", "thx", "appreciate")):
        return "You're welcome! Anything else I can help you book?"
    if any(w in t for w in ("hello", "hi ", "hi", "hey", "good morning", "good afternoon", "good evening", "greetings")):
        return "Hello! I can help you book a doctor. Tell me the specialty or your symptoms."
    if "bye" in t or "goodbye" in t or "see you" in t:
        return "Goodbye and take care!"
    if "how are you" in t:
        return "Doing well, thanks! Ready to help you book a doctor — what do you need?"
    return ("I can help book doctor appointments. Try: "
            "'Book a cardiologist tomorrow after 3 PM' or tell me your symptoms.")


def chitchat_node(s):
    """Friendly reply for greetings / thanks / small talk. Never gives medical advice."""
    raw = llm.chat(
        "You are a warm, concise hospital appointment assistant. Reply to the user in ONE short sentence. "
        "Do NOT give medical advice or diagnose. If the user wants to book, invite them to describe the "
        "specialty or their symptoms in one sentence.",
        s["user_text"],
    )
    reply = raw or _fallback_chitchat(s["user_text"])
    return {"outcome": "chitchat", "reply": reply,
            "trace": T(s, "CHITCHAT: handled conversationally")}


# ------------------------------------------------------------------ deterministic nodes
def intent_node(s):
    text = s["user_text"]
    if MEDICAL_ADVICE.search(text):
        return {"outcome": "refused",
                "trace": T(s, "GUARDRAIL: medical-advice request detected -> refused"),
                "reply": ("I can't give medical advice or diagnoses. I can book you with a doctor who can; "
                          "tell me the specialty or your symptoms. For emergencies call your local emergency number.")}
    if not BOOKING_HINTS.search(text):
        return chitchat_node(s)
    it = parse_intent(text)
    return {"intent": it, "trace": T(s, f"INTENT ({it['source']}): {json.dumps(it)}"),
            "acl": [], "attempts": 0, "conflict_used": False}


def gap_node(s):
    it, tr = dict(s["intent"]), s["trace"]
    if not it["specialty"] and it["symptoms"]:
        spec, score = memory.retrieve_specialty(it["symptoms"])
        if spec:
            it["specialty"] = spec
            tr = tr + [f"RAG: symptoms -> '{spec}' (cosine {score:.2f})"]
    missing = []
    if not it["specialty"]:
        missing.append("specialty")
    if not db.get_patient(s["patient_id"]).get("phone"):
        missing.append("phone")
    tr = tr + [f"GAP CHECK: missing = {missing or 'none'}"]
    if missing:
        q = ("Which specialty do you need (or what symptoms do you have)?" if "specialty" in missing
             else "I don't have your phone number for the confirmation. What is it?")
        return {"intent": it, "missing": missing, "outcome": "need_info", "reply": q, "trace": tr}
    return {"intent": it, "missing": [], "trace": tr}


def cnp_node(s):
    it = s["intent"]
    mgr = agents.ManagerAgent(window=s.get("window", 60))
    bids, log = mgr.contract_net(it["specialty"], it["day"], it["after"], s["patient_id"])
    tr = T(s, f"CONTRACT NET: {len(bids)} proposals for {it['specialty']} on {it['day']} after {it['after']}")
    if not bids:
        return {"bids": [], "acl": log, "trace": tr + ["No slots -> waitlist"]}
    earliest = min(b["minutes"] for b in bids)
    # Attach a human-readable 'why' to EVERY bid so that after a rejection the newly-promoted
    # top bid still has one (this is what fixes the KeyError: 'why').
    for rank, b in enumerate(bids):
        if rank == 0:
            if b["minutes"] == earliest:
                b["why"] = (f"earliest slot across {len(bids)} cardiology bids"
                            if it["specialty"] == "cardiology"
                            else f"earliest slot across {len(bids)} bids")
            else:
                b["why"] = (f"{b['minutes'] - earliest} min after the earliest bid, "
                            f"but preferred by this patient's memory (score {b['mem']:+d})")
        else:
            b["why"] = (f"alternative #{rank + 1} (memory score {b['mem']:+d}, "
                        f"{b['minutes'] - earliest} min after the earliest bid)")
    top = bids[0]
    reply = f"I found {top['doctor']} on {top['day']} at {top['time']} ({top['why']}). Shall I book it?"
    return {"bids": bids, "acl": log, "outcome": "awaiting_approval", "reply": reply, "trace": tr}


def book_node(s):
    bid = s["bids"][0]
    used = s.get("conflict_used", False)
    if s.get("simulate_conflict") and not used:
        db.simulate_conflict(bid["slot_id"])
        used = True
    res = db.book_slot(bid["slot_id"], s["patient_id"])
    if res["ok"]:
        agents.release_hold(bid["doctor_id"], bid["slot_id"])
        return {"appointment": res, "outcome": "booked", "conflict_used": used,
                "trace": T(s, f"BOOK: {bid['doctor']} {res['day']} {res['time']} -> appointment #{res['appointment_id']}")}
    return {"outcome": "retry", "conflict_used": used, "attempts": s.get("attempts", 0) + 1,
            "trace": T(s, f"BOOK FAILED ({res['reason']}) for {bid['doctor']} {bid['time']}")}


def route_book(s):
    if s["outcome"] == "booked":
        return "notify"
    if s.get("attempts", 0) < 3 and len(s["bids"]) > 1:
        return "retry"
    if s.get("attempts", 0) > 0:
        return "reflect"
    return "waitlist"


def retry_node(s):
    bids = s["bids"][1:]
    return {"bids": bids, "trace": T(s, f"RETRY: next best proposal -> {bids[0]['doctor']} {bids[0]['time']}")}


def waitlist_node(s):
    it = s["intent"]
    db.add_waitlist(s["patient_id"], it["specialty"], it["day"])
    return {"outcome": "waitlisted", "trace": T(s, "WAITLIST: added"),
            "reply": (f"No {it['specialty']} slot is available on {it['day']} after {it['after']}. "
                      f"You are on the waitlist and I'll notify you if one opens.")}


def reflect_node(s):
    failed = s.get("bids", [])
    last = failed[0]["doctor"] if failed else "the last proposal"
    note = (f"Booking failed after {s.get('attempts', 0)} attempt(s). "
            f"Last failure: {last}. Consider widening the time window or asking the patient for flexibility.")
    memory.set_memory(s["patient_id"], "last_reflection", note)
    return {"outcome": "reflected",
            "trace": T(s, f"REFLECT: {note}"),
            "reply": (f"I couldn't secure any of the proposals. {note}\n\n"
                      f"Would you like me to try a different time window?")}


def notify_node(s):
    p = db.get_patient(s["patient_id"])
    it = s["intent"]
    if s["outcome"] == "booked":
        a, bid = s["appointment"], s["bids"][0]
        facts = (f"Patient {p['name']}; {it['specialty']} appointment with {bid['doctor']} on {a['day']} at "
                 f"{a['time']}; reference #{a['appointment_id']}")
        fallback = (f"Hello {p['name']}, your {it['specialty']} appointment with {bid['doctor']} is confirmed for "
                    f"{a['day']} at {a['time']}. Ref #{a['appointment_id']}.")
        memory.record_feedback(s["patient_id"], bid["doctor_id"], "accepted")
        extra = ["MEMORY: acceptance stored -> future ranking adapts"]
    else:
        facts = f"Patient {p['name']}; no {it['specialty']} slot on {it['day']} after {it['after']}; added to waitlist"
        fallback = f"Hello {p['name']}, no slot is free right now. You are on the waitlist for {it['specialty']}."
        extra = []
    msg = llm.chat(
        "Write ONE friendly SMS under 35 words using only these facts. Do not add medical advice.",
        facts,
    ) or fallback
    status = db.send_notification(s["patient_id"], msg)
    if s["outcome"] == "waitlisted":
        reply = s.get("reply", "")
    else:
        body = fallback.split(", ", 1)[1] if ", " in fallback else fallback
        reply = f"Booked! {body}"
    reply = reply + f"\n\nNotification: \"{msg}\""
    return {"reply": reply, "trace": T(s, f"NOTIFY: {status}") + extra}


# ------------------------------------------------------------------ deterministic graph
def route_entry(s):
    return {"execute": "book", "waitlist": "waitlist"}.get(s.get("stage"), "intent")


def after_intent(s):
    return END if s.get("outcome") in ("refused", "chitchat") else "gap"


def after_gap(s):
    return END if s.get("outcome") == "need_info" else "cnp"


def after_cnp(s):
    return "waitlist" if not s["bids"] else END


def build():
    g = StateGraph(S)
    for name, fn in [
        ("intent", intent_node), ("gap", gap_node), ("cnp", cnp_node), ("book", book_node),
        ("retry", retry_node), ("waitlist", waitlist_node), ("notify", notify_node),
        ("reflect", reflect_node),
    ]:
        g.add_node("n_" + name, fn)
    g.add_conditional_edges(START, route_entry, {"intent": "n_intent", "book": "n_book", "waitlist": "n_waitlist"})
    g.add_conditional_edges("n_intent", after_intent, {END: END, "gap": "n_gap"})
    g.add_conditional_edges("n_gap", after_gap, {END: END, "cnp": "n_cnp"})
    g.add_conditional_edges("n_cnp", after_cnp, {END: END, "waitlist": "n_waitlist"})
    g.add_conditional_edges("n_book", route_book,
                            {"notify": "n_notify", "retry": "n_retry",
                             "waitlist": "n_waitlist", "reflect": "n_reflect"})
    g.add_edge("n_retry", "n_book")
    g.add_edge("n_waitlist", "n_notify")
    g.add_edge("n_notify", END)
    g.add_edge("n_reflect", END)
    return g.compile()


# ------------------------------------------------------------------ genuine ReAct agent
def build_react():
    from langgraph.graph import StateGraph, END, START, MessagesState
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    TOOLS = {
        "search_doctors":     db.search_doctors,
        "get_slots":          db.get_slots,
        "contract_net":       lambda specialty, day, after, patient_id:
                                agents.ManagerAgent().contract_net(specialty, day, after, patient_id)[0],
        "book_slot":          db.book_slot,
        "add_waitlist":       db.add_waitlist,
        "send_notification":  db.send_notification,
        "retrieve_specialty": memory.retrieve_specialty,
        "resolve_day":        lambda offset: db.day_offset(int(offset)),
        "get_patient":        db.get_patient,
    }

    TOOL_HELP = (
        "search_doctors(specialty) -> list of {id,name,specialty}\n"
        "get_slots(doctor_id, day, after) -> list of {id,time,day}\n"
        "contract_net(specialty, day, after, patient_id) -> ranked bids\n"
        "book_slot(slot_id, patient_id) -> {ok,...}\n"
        "add_waitlist(patient_id, specialty, day)\n"
        "send_notification(patient_id, message)\n"
        "retrieve_specialty(symptoms) -> (specialty, score)\n"
        "resolve_day(offset_int) -> ISO date\n"
        "get_patient(patient_id) -> patient record"
    )

    SYSTEM = (
        "You are a hospital appointment agent. Use the tools to help the patient. "
        "If the user is just greeting, thanking you, or making small talk, respond warmly "
        "and offer to help book an appointment — do not call a tool. "
        "Reply with ONLY a JSON object — either "
        '{"tool": "<name>", "args": {...}} '
        'or {"final": "<plain-text reply>"}. '
        "Never invent slot IDs. If the user asks for medical advice, refuse politely.\n\n"
        "Tools:\n" + TOOL_HELP
    )

    def _fmt(msgs):
        lines = []
        for m in msgs:
            if isinstance(m, HumanMessage):
                lines.append(f"USER: {m.content}")
            elif isinstance(m, AIMessage):
                if getattr(m, "tool_calls", None):
                    for tc in m.tool_calls:
                        lines.append(f"ASSISTANT_TOOL_CALL: {tc['name']}({json.dumps(tc['args'])})")
                else:
                    lines.append(f"ASSISTANT: {m.content}")
            elif isinstance(m, ToolMessage):
                lines.append(f"TOOL_RESULT({m.tool_call_id}): {m.content}")
            else:
                lines.append(f"{type(m).__name__}: {getattr(m, 'content', m)}")
        return "\n".join(lines)

    def llm_node(state: MessagesState):
        msgs = state["messages"]
        tool_results = sum(1 for m in msgs if isinstance(m, ToolMessage))
        if tool_results >= 8:
            return {"messages": msgs + [AIMessage(
                content="I've reached my step limit. Please try a narrower request.")]}
        raw = llm.chat(SYSTEM, _fmt(msgs), json_mode=True)
        if not raw:
            return {"messages": msgs + [AIMessage(
                content="[LLM unavailable] Switch back to deterministic mode or configure a backend.")]}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {"messages": msgs + [AIMessage(content=raw[:400])]}
        if "tool" in parsed and parsed["tool"] in TOOLS:
            tc_id = f"call_{len(msgs)}"
            return {"messages": msgs + [AIMessage(
                content="",
                tool_calls=[{"name": parsed["tool"],
                             "args": parsed.get("args", {}),
                             "id": tc_id}],
            )]}
        return {"messages": msgs + [AIMessage(content=str(parsed.get("final", "")))]}

    def tool_node(state: MessagesState):
        last = state["messages"][-1]
        results = []
        for tc in last.tool_calls:
            fn = TOOLS.get(tc["name"])
            if fn is None:
                out = {"error": f"unknown tool {tc['name']}"}
            else:
                try:
                    out = fn(**tc.get("args", {}))
                except Exception as e:
                    out = {"error": str(e)}
            results.append(ToolMessage(
                content=json.dumps(out, default=str)[:2000],
                tool_call_id=tc["id"],
            ))
        return {"messages": state["messages"] + results}

    def after_agent(state: MessagesState):
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            return "tools"
        return END

    g = StateGraph(MessagesState)
    g.add_node("agent", llm_node)
    g.add_node("tools", tool_node)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", after_agent, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile()


# ------------------------------------------------------------------ human feedback + evaluation
def reject(s):
    """Patient rejected the proposal: store feedback, release the hold, move to next bid."""
    cur = s["bids"][0]
    memory.record_feedback(s["patient_id"], cur["doctor_id"], "rejected")
    agents.release_hold(cur["doctor_id"], cur["slot_id"])
    bids = s["bids"][1:]
    s = dict(s, bids=bids, trace=T(s, f"HUMAN: rejected {cur['doctor']} -> feedback stored, hold released"))
    if bids:
        b = bids[0]
        # Defensive: cnp_node already sets 'why' on every bid, but if an old state slipped
        # through (e.g. cached) we fill it in here so the UI never raises KeyError.
        b.setdefault("why", f"next alternative (memory {b.get('mem', 0):+d}, {b.get('time', '?')})")
        s["reply"] = f"Understood. Next option: {b['doctor']} on {b['day']} at {b['time']}. Shall I book it?"
        s["outcome"] = "awaiting_approval"
    else:
        s["stage"] = "waitlist"
    return s


SCENARIOS = [
    ("Returning patient, memory re-ranks", "Book the earliest cardiologist appointment tomorrow after 3 PM", 1, False,
     {"outcome": "booked", "doctor": "Dr. Mehta"}),
    ("New patient history, strict earliest", "Book the earliest cardiologist appointment tomorrow after 3 PM", 2, False,
     {"outcome": "booked", "doctor": "Dr. Rao"}),
    ("Slot stolen -> retry next bid", "Book the earliest cardiologist appointment tomorrow after 3 PM", 2, True,
     {"outcome": "booked", "doctor": "Dr. Khan"}),
    ("Symptoms -> RAG -> cardiology", "I have chest pain, book a doctor tomorrow after 3 PM", 2, False,
     {"outcome": "booked", "doctor": "Dr. Rao"}),
    ("Typo tolerance (cardeologist)", "Book a cardeologist tomorrow after 3 PM", 2, False,
     {"outcome": "booked", "doctor": "Dr. Rao"}),
    ("Medical advice guardrail", "Do I have a heart attack?", 2, False, {"outcome": "refused", "doctor": None}),
    ("No slot -> waitlist", "Book a dermatologist tomorrow after 8 PM", 2, False,
     {"outcome": "waitlisted", "doctor": None}),
    ("Greeting -> chitchat", "Hello!", 2, False, {"outcome": "chitchat", "doctor": None}),
    ("Thanks -> chitchat", "Thank you so much", 2, False, {"outcome": "chitchat", "doctor": None}),
]


def run_scenario(app, text, pid, sim=False, window=60):
    agents.reset_agents()
    s = app.invoke({"user_text": text, "patient_id": pid, "stage": "plan", "simulate_conflict": sim,
                    "window": window, "trace": [], "acl": []})
    if s.get("outcome") == "awaiting_approval":
        s = app.invoke(dict(s, stage="execute"))
    return s