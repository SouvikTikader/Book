"""LangGraph workflow. Deterministic work (dates, times, booking) is done by Python tools; the LLM only
understands the request and writes the patient message."""
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
MEDICAL_ADVICE = re.compile(r"\b(do i have|diagnos\w*|what medicine|which medicine|prescri\w*|dosage|am i having|"
                            r"is it a heart attack|treatment for)\b", re.I)


class S(TypedDict, total=False):
    user_text: str
    patient_id: int
    stage: str            # plan | execute | waitlist
    intent: dict
    missing: list
    bids: list
    outcome: str          # refused | need_info | awaiting_approval | booked | waitlisted
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


# ------------------------------------------------------------------ parsing helpers (code, not LLM)
def find_specialty(text):
    """Exact keyword first, then fuzzy match so typos like 'cardeologist' still work."""
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
    raw = llm.chat('Extract a hospital booking request. Reply ONLY JSON with keys: specialty (one of '
                   '"cardiology","neurology","dermatology","orthopedics","pediatrics","general" or null), after_time ("HH:MM" '
                   '24h or null), symptoms (string or null).', text, json_mode=True)
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
    intent["day"] = db.day_offset(intent["offset"])  # "tomorrow" resolved in Python
    if intent["offset"] == 0:  # never offer a slot that is already in the past today
        now = datetime.now().strftime("%H:%M")
        intent["after"] = max(intent["after"], now)
    intent["source"] = src
    return intent


# ------------------------------------------------------------------ nodes
def intent_node(s):
    if MEDICAL_ADVICE.search(s["user_text"]):  # guardrail (Unit 4: human-agent collaboration / safety)
        return {"outcome": "refused", "trace": T(s, "GUARDRAIL: medical-advice request detected -> refused"),
                "reply": "I can't give medical advice or diagnoses. I can book you with a doctor who can; "
                         "tell me the specialty or your symptoms. For emergencies call your local emergency number."}
    it = parse_intent(s["user_text"])
    return {"intent": it, "trace": T(s, f"INTENT ({it['source']}): {json.dumps(it)}"), "acl": [], "attempts": 0,
            "conflict_used": False}


def gap_node(s):
    it, tr = dict(s["intent"]), s["trace"]
    if not it["specialty"] and it["symptoms"]:
        spec, score = memory.retrieve_specialty(it["symptoms"])  # Unit 3: vector retrieval
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
    b = bids[0]
    reply = (f"I found {b['doctor']} on {b['day']} at {b['time']} ({b['why']}). Shall I book it?")
    return {"bids": bids, "acl": log, "outcome": "awaiting_approval", "reply": reply, "trace": tr}


def book_node(s):
    bid = s["bids"][0]
    used = s.get("conflict_used", False)
    if s.get("simulate_conflict") and not used:
        db.simulate_conflict(bid["slot_id"])  # another patient grabs the slot first
        used = True
    res = db.book_slot(bid["slot_id"], s["patient_id"])
    if res["ok"]:
        return {"appointment": res, "outcome": "booked", "conflict_used": used,
                "trace": T(s, f"BOOK: {bid['doctor']} {res['day']} {res['time']} -> appointment #{res['appointment_id']}")}
    return {"outcome": "retry", "conflict_used": used, "attempts": s.get("attempts", 0) + 1,
            "trace": T(s, f"BOOK FAILED ({res['reason']}) for {bid['doctor']} {bid['time']}")}


def route_book(s):
    if s["outcome"] == "booked":
        return "notify"
    if s.get("attempts", 0) < 3 and len(s["bids"]) > 1:
        return "retry"
    return "waitlist"


def retry_node(s):
    bids = s["bids"][1:]
    return {"bids": bids, "trace": T(s, f"RETRY: next best proposal -> {bids[0]['doctor']} {bids[0]['time']}")}


def waitlist_node(s):
    it = s["intent"]
    db.add_waitlist(s["patient_id"], it["specialty"], it["day"])
    return {"outcome": "waitlisted", "trace": T(s, "WAITLIST: added"),
            "reply": f"No {it['specialty']} slot is available on {it['day']} after {it['after']}. "
                     f"You are on the waitlist and I'll notify you if one opens."}


def notify_node(s):
    p = db.get_patient(s["patient_id"])
    it = s["intent"]
    if s["outcome"] == "booked":
        a, bid = s["appointment"], s["bids"][0]
        facts = (f"Patient {p['name']}; {it['specialty']} appointment with {bid['doctor']} on {a['day']} at "
                 f"{a['time']}; reference #{a['appointment_id']}")
        fallback = (f"Hello {p['name']}, your {it['specialty']} appointment with {bid['doctor']} is confirmed for "
                    f"{a['day']} at {a['time']}. Ref #{a['appointment_id']}.")
        memory.record_feedback(s["patient_id"], bid["doctor_id"], "accepted")  # Unit 3: learn from acceptance
        extra = ["MEMORY: acceptance stored -> future ranking adapts"]
    else:
        facts = f"Patient {p['name']}; no {it['specialty']} slot on {it['day']} after {it['after']}; added to waitlist"
        fallback = f"Hello {p['name']}, no slot is free right now. You are on the waitlist for {it['specialty']}."
        extra = []
    msg = llm.chat("Write ONE friendly SMS under 35 words using only these facts. Do not add medical advice.",
                   facts) or fallback
    status = db.send_notification(s["patient_id"], msg)
    reply = (s.get("reply", "") if s["outcome"] == "waitlisted" else
             f"Booked! {fallback.split(', ', 1)[1]}") + f"\n\nNotification: \"{msg}\""
    return {"reply": reply, "trace": T(s, f"NOTIFY: {status}") + extra}


# ------------------------------------------------------------------ graph
def route_entry(s):
    return {"execute": "book", "waitlist": "waitlist"}.get(s.get("stage"), "intent")


def after_intent(s):
    return END if s.get("outcome") == "refused" else "gap"


def after_gap(s):
    return END if s.get("outcome") == "need_info" else "cnp"


def after_cnp(s):
    return "waitlist" if not s["bids"] else END


def build():
    # NOTE: LangGraph forbids a node name equal to a state key (state has "intent"), so nodes get an "n_" prefix.
    g = StateGraph(S)
    for name, fn in [("intent", intent_node), ("gap", gap_node), ("cnp", cnp_node), ("book", book_node),
                     ("retry", retry_node), ("waitlist", waitlist_node), ("notify", notify_node)]:
        g.add_node("n_" + name, fn)
    g.add_conditional_edges(START, route_entry, {"intent": "n_intent", "book": "n_book", "waitlist": "n_waitlist"})
    g.add_conditional_edges("n_intent", after_intent, {END: END, "gap": "n_gap"})
    g.add_conditional_edges("n_gap", after_gap, {END: END, "cnp": "n_cnp"})
    g.add_conditional_edges("n_cnp", after_cnp, {END: END, "waitlist": "n_waitlist"})
    g.add_conditional_edges("n_book", route_book, {"notify": "n_notify", "retry": "n_retry", "waitlist": "n_waitlist"})
    g.add_edge("n_retry", "n_book")
    g.add_edge("n_waitlist", "n_notify")
    g.add_edge("n_notify", END)
    return g.compile()


# ------------------------------------------------------------------ human feedback + evaluation
def reject(s):
    """Patient rejected the proposal: store feedback (Unit 3) and move to the next bid (Unit 4)."""
    cur = s["bids"][0]
    memory.record_feedback(s["patient_id"], cur["doctor_id"], "rejected")
    bids = s["bids"][1:]
    s = dict(s, bids=bids, trace=T(s, f"HUMAN: rejected {cur['doctor']} -> feedback stored"))
    if bids:
        b = bids[0]
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
]


def run_scenario(app, text, pid, sim=False, window=60):
    s = app.invoke({"user_text": text, "patient_id": pid, "stage": "plan", "simulate_conflict": sim,
                    "window": window, "trace": [], "acl": []})
    if s.get("outcome") == "awaiting_approval":
        s = app.invoke(dict(s, stage="execute"))
    return s