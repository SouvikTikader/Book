import asyncio
import json
import re
import threading

import pandas as pd
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import agents
import db
import graph
import llm
import memory

st.set_page_config(page_title="Hospital Appointment Agent", page_icon="🏥", layout="wide")


@st.cache_resource
def bootstrap_db():
    db.init_db()
    return True


bootstrap_db()


@st.cache_resource
def get_deterministic():
    return graph.build()


@st.cache_resource
def get_react():
    return graph.build_react()


@st.cache_resource
def start_watcher():
    def _run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(waitlist_watcher())
        except Exception:
            pass
    threading.Thread(target=_run, daemon=True).start()
    return True


async def waitlist_watcher(interval=60):
    """Notify waitlisted patients when a matching slot frees up.
    Reads are done on a short-lived connection that is closed before any writes."""
    while True:
        pairs = []
        try:
            c = db.conn()
            try:
                rows = c.execute(
                    "SELECT w.id AS wid, w.patient_id, w.specialty, w.day, "
                    "       s.id AS slot_id, s.time, d.name AS doctor "
                    "FROM waitlist w "
                    "JOIN slots s ON s.day = w.day AND s.booked = 0 "
                    "JOIN doctors d ON d.id = s.doctor_id AND d.specialty = w.specialty "
                    "LIMIT 3"
                ).fetchall()
                pairs = [dict(r) for r in rows]
            finally:
                c.close()
        except Exception:
            pairs = []

        for r in pairs:
            try:
                msg = (f"Good news: a {r['specialty']} slot opened with {r['doctor']} "
                       f"on {r['day']} at {r['time']}.")
                db.send_notification(r["patient_id"], msg)
                c = db.conn()
                try:
                    c.execute("DELETE FROM waitlist WHERE id=?", (r["wid"],))
                finally:
                    c.close()
            except Exception:
                pass

        await asyncio.sleep(interval)


agent = get_deterministic()
react_agent = get_react()
start_watcher()

# ------------------------------------------------------------------ per-patient state
ss = st.session_state
ss.setdefault("chats", {})     # pid -> list[(role, msg)]
ss.setdefault("states", {})    # pid -> last graph state
ss.setdefault("pendings", {})  # pid -> pending user request (for slot-filling)
ss.setdefault("askings", {})   # pid -> "phone" | "specialty" | None
ss.setdefault("mode", "deterministic")

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Settings")
    patients = db.list_patients()
    names = {p["id"]: f"{p['name']} (#{p['id']})" for p in patients}
    pid = st.selectbox("Patient", list(names), format_func=names.get, key="pid")
    with st.expander("➕ New patient"):
        nm = st.text_input("Name", key="newname")
        if st.button("Create") and nm.strip():
            db.create_patient(nm.strip())
            st.rerun()

    st.subheader("LLM backend")
    llm.CONFIG["use_llm"] = st.toggle("Use LLM", value=True,
                                     help="Off = rule-based fallback only")
    backend = st.radio("Backend", ["ollama", "gemini"], horizontal=True,
                       index=0 if llm.CONFIG["backend"] == "ollama" else 1)
    llm.CONFIG["backend"] = backend
    if backend == "gemini":
        llm.CONFIG["gemini_api_key"] = st.text_input(
            "Gemini API key", type="password",
            value=llm.CONFIG.get("gemini_api_key") or "",
            help="Or set GOOGLE_API_KEY / GEMINI_API_KEY in the environment.",
        )
        llm.CONFIG["gemini_model"] = st.text_input(
            "Gemini model", llm.CONFIG["gemini_model"],
            help="e.g. gemini-2.5-flash or gemini-2.6-flash (if available in your region)",
        )
    else:
        llm.CONFIG["model"] = st.text_input("Ollama model", llm.CONFIG["model"])

    st.subheader("Agent mode")
    st.radio("Mode", ["deterministic", "react"], key="mode", horizontal=True,
             help="'react' = LLM chooses tools (genuine agent); 'deterministic' = fixed graph with LLM fallback.")

    sim = st.checkbox("Simulate slot conflict (forces retry)")
    window = st.slider("Memory tolerance window (min)", 0, 120, 60, 15,
                       help="Within this many minutes of the earliest slot, patient memory breaks ties. "
                            "0 = strictly earliest.")
    if st.button(" Reset demo data"):
        db.reset()
        agents.reset_agents()
        ss.chats.clear()
        ss.states.clear()
        ss.pendings.clear()
        ss.askings.clear()
        st.rerun()

st.title(" Hospital Appointment Agent")
st.caption("LangGraph + multi-agent negotiation + long-term memory "
           "| try: *Book the earliest cardiologist appointment tomorrow after 3 PM*")

# ------------------------------------------------------------------ per-patient views
chat = ss.chats.setdefault(pid, [])
asking = ss.askings.get(pid)


def _run_react(text, pid):
    user_msg = f"[patient_id={pid}] {text}"
    out = react_agent.invoke({"messages": [HumanMessage(content=user_msg)]})
    msgs = out["messages"]
    final = msgs[-1].content or "(no reply)"
    trace, acl = [], []
    for m in msgs:
        if isinstance(m, ToolMessage):
            trace.append(f"TOOL RESULT: {m.content[:300]}")
        elif isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            for tc in m.tool_calls:
                trace.append(f"TOOL CALL: {tc['name']}({json.dumps(tc['args'])})")
    ss.states[pid] = {"trace": trace, "acl": acl, "outcome": "done", "reply": final}
    ss.askings[pid] = None
    ss.chats[pid].append(("assistant", final))


def run(text, pid):
    ss.pendings[pid] = text
    if ss.mode == "react":
        _run_react(text, pid)
        return
    out = agent.invoke({
        "user_text": text, "patient_id": pid, "stage": "plan",
        "simulate_conflict": sim, "window": window, "trace": [], "acl": [],
    })
    ss.states[pid] = out
    ss.askings[pid] = out["missing"][0] if out.get("outcome") == "need_info" else None
    ss.chats[pid].append(("assistant", out["reply"]))



user_text = st.chat_input("Type your request…")
if user_text:
    chat.append(("user", user_text))
    digits = re.sub(r"\D", "", user_text)
    short = len(user_text.split()) <= 4
    if asking == "phone" and len(digits) >= 10:
        db.set_phone(pid, digits[-10:])
        chat.append(("assistant", "Thanks, phone saved to your profile."))
        run(ss.pendings.get(pid, ""), pid)
    elif asking == "phone" and short:
        chat.append(("assistant", "That doesn't look like a valid phone number. Please enter 10 digits."))
    elif asking == "specialty" and short:
        run((ss.pendings.get(pid) or "") + " " + user_text, pid)
    else:
        run(user_text, pid)

# ------------------------------------------------------------------ chat
for role, msg in chat:
    with st.chat_message(role):
        st.write(msg)

stt = ss.states.get(pid)
if stt and stt.get("outcome") == "awaiting_approval":
    b = stt["bids"][0]
    why = b.get("why") or f"memory score {b.get('mem', 0):+d}"
    st.info(f"Proposal: **{b['doctor']}** | {b['day']} {b['time']} | {why}")
    c1, c2 = st.columns(2)
    if c1.button("✅ Approve & book", use_container_width=True):
        out = agent.invoke(dict(stt, stage="execute"))
        ss.states[pid] = out
        ss.chats[pid].append(("assistant", out["reply"]))
        st.rerun()
    if c2.button("❌ Reject", use_container_width=True):
        new = graph.reject(stt)
        if new.get("stage") == "waitlist":
            new = agent.invoke(new)
        ss.states[pid] = new
        ss.chats[pid].append(("assistant", new["reply"]))
        st.rerun()

# ------------------------------------------------------------------ inspection tabs
t1, t2, t3, t4, t5 = st.tabs([
    " Agent trace", " Agent messages (FIPA-ACL)", " Memory",
    " Appointments", " Evaluation",
])
with t1:
    st.code("\n".join(stt["trace"]) if stt and stt.get("trace") else "No run yet", language="text")
    if stt and stt.get("bids"):
        st.dataframe(
            pd.DataFrame(stt["bids"])[["doctor", "day", "time", "mem", "in_window"]]
              .rename(columns={"mem": "memory score", "in_window": "in tolerance window"}),
            use_container_width=True,
        )
with t2:
    st.code("\n".join(stt["acl"]) if stt and stt.get("acl") else "No negotiation yet", language="text")
with t3:
    st.write("**Long-term memory**", memory.get_memory(pid) or "(empty)")
    c = db.conn()
    try:
        fb = pd.DataFrame([dict(r) for r in c.execute(
            "SELECT d.name AS doctor, f.event, COUNT(*) AS count FROM feedback f "
            "JOIN doctors d ON d.id=f.doctor_id WHERE f.patient_id=? GROUP BY d.name, f.event",
            (pid,))])
    finally:
        c.close()
    st.write("**Feedback history**")
    st.dataframe(fb if not fb.empty else pd.DataFrame({"info": ["no feedback yet"]}),
                 use_container_width=True)
    st.write("**Notifications**")
    notif = db.notifications_for(pid)
    st.dataframe(pd.DataFrame(notif) if notif else pd.DataFrame({"info": ["no notifications yet"]}),
                 use_container_width=True)
with t4:
    ap = db.appointments_for(pid)
    st.dataframe(pd.DataFrame(ap) if ap else pd.DataFrame({"info": ["no appointments yet"]}),
                 use_container_width=True)
with t5:
    st.caption("Runs 9 scenarios on fresh demo data (this resets the database).")
    if st.button("Run evaluation"):
        rows = []
        for name, text, p, conflict, expect in graph.SCENARIOS:
            db.reset()
            agents.reset_agents()
            out = graph.run_scenario(agent, text, p, conflict)
            doc = out["bids"][0]["doctor"] if out.get("outcome") == "booked" else None
            ok = out.get("outcome") == expect["outcome"] and doc == expect["doctor"]
            rows.append({
                "scenario": name,
                "expected": f"{expect['outcome']} / {expect['doctor']}",
                "actual": f"{out.get('outcome')} / {doc}",
                "result": "PASS " if ok else "FAIL ",
            })
        db.reset()
        agents.reset_agents()
        ss.chats.clear()
        ss.states.clear()
        st.dataframe(pd.DataFrame(rows), use_container_width=True)