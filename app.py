import re

import pandas as pd
import streamlit as st

import db
import graph
import llm
import memory

st.set_page_config(page_title="Hospital Appointment Agent", page_icon="🏥", layout="wide")
db.init_db()


@st.cache_resource
def get_app():
    return graph.build()


agent = get_app()
ss = st.session_state
ss.setdefault("chat", [])
ss.setdefault("state", None)
ss.setdefault("pending", None)
ss.setdefault("asking", None)

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Settings")
    patients = db.list_patients()
    names = {p["id"]: f"{p['name']} (#{p['id']})" for p in patients}
    pid = st.selectbox("Patient", list(names), format_func=names.get)
    with st.expander("➕ New patient"):
        nm = st.text_input("Name", key="newname")
        if st.button("Create") and nm.strip():
            db.create_patient(nm.strip())
            st.rerun()
    llm.CONFIG["use_llm"] = st.toggle("Use Ollama LLM", value=True, help="Off = rule-based fallback only")
    llm.CONFIG["model"] = st.text_input("Ollama model", llm.CONFIG["model"])
    sim = st.checkbox("Simulate slot conflict (forces retry)")
    window = st.slider("Memory tolerance window (min)", 0, 120, 60, 15,
                       help="Within this many minutes of the earliest slot, patient memory breaks ties. 0 = strictly earliest.")
    if st.button("♻️ Reset demo data"):
        db.reset()
        ss.chat, ss.state, ss.pending, ss.asking = [], None, None, None
        st.rerun()

st.title("🏥 Hospital Appointment Agent")
st.caption("LangGraph + Contract-Net multi-agent negotiation + long-term memory | try: "
           "*Book the earliest cardiologist appointment tomorrow after 3 PM*")


def run(text):
    ss.pending = text
    st.session_state.state = agent.invoke({"user_text": text, "patient_id": pid, "stage": "plan",
                                           "simulate_conflict": sim, "window": window, "trace": [], "acl": []})
    ss.asking = ss.state["missing"][0] if ss.state.get("outcome") == "need_info" else None
    ss.chat.append(("assistant", ss.state["reply"]))


# ------------------------------------------------------------------ input handling (before rendering)
user_text = st.chat_input("Type your request…")
if user_text:
    ss.chat.append(("user", user_text))
    digits = re.sub(r"\D", "", user_text)
    short = len(user_text.split()) <= 4          # a short reply is treated as an answer to the agent's question
    if ss.asking == "phone" and len(digits) >= 10:
        db.set_phone(pid, digits[-10:])
        ss.chat.append(("assistant", "Thanks, phone saved to your profile."))
        run(ss.pending)                          # continue the ORIGINAL request
    elif ss.asking == "phone" and short:
        ss.chat.append(("assistant", "That doesn't look like a valid phone number. Please enter 10 digits."))
    elif ss.asking == "specialty" and short:
        run(ss.pending + " " + user_text)        # merge the answer into the original request (slot filling)
    else:
        run(user_text)                           # a fresh request

# ------------------------------------------------------------------ chat + approval (human-in-the-loop)
for role, msg in ss.chat:
    with st.chat_message(role):
        st.write(msg)

stt = ss.state
if stt and stt.get("outcome") == "awaiting_approval":
    b = stt["bids"][0]
    st.info(f"Proposal: **{b['doctor']}** | {b['day']} {b['time']} | {b['why']}")
    c1, c2 = st.columns(2)
    if c1.button("✅ Approve & book", use_container_width=True):
        out = agent.invoke(dict(stt, stage="execute"))
        ss.state = out
        ss.chat.append(("assistant", out["reply"]))
        st.rerun()
    if c2.button("❌ Reject", use_container_width=True):
        new = graph.reject(stt)
        if new.get("stage") == "waitlist":
            new = agent.invoke(new)
        ss.state = new
        ss.chat.append(("assistant", new["reply"]))
        st.rerun()

# ------------------------------------------------------------------ inspection tabs
t1, t2, t3, t4, t5 = st.tabs(["🧠 Agent trace", "📨 Agent messages (FIPA-ACL)", "💾 Memory", "📅 Appointments", "✅ Evaluation"])
with t1:
    st.code("\n".join(stt["trace"]) if stt and stt.get("trace") else "No run yet", language="text")
    if stt and stt.get("bids"):
        st.dataframe(pd.DataFrame(stt["bids"])[["doctor", "day", "time", "mem", "in_window"]]
                     .rename(columns={"mem": "memory score", "in_window": "in tolerance window"}),
                     use_container_width=True)
with t2:
    st.code("\n".join(stt["acl"]) if stt and stt.get("acl") else "No negotiation yet", language="text")
with t3:
    st.write("**Long-term memory**", memory.get_memory(pid) or "(empty)")
    c = db.conn()
    fb = pd.DataFrame([dict(r) for r in c.execute(
        "SELECT d.name AS doctor, f.event, COUNT(*) AS count FROM feedback f JOIN doctors d ON d.id=f.doctor_id "
        "WHERE f.patient_id=? GROUP BY d.name, f.event", (pid,))])
    c.close()
    st.write("**Feedback history**")
    st.dataframe(fb if not fb.empty else pd.DataFrame({"info": ["no feedback yet"]}), use_container_width=True)
with t4:
    ap = db.appointments_for(pid)
    st.dataframe(pd.DataFrame(ap) if ap else pd.DataFrame({"info": ["no appointments yet"]}), use_container_width=True)
with t5:
    st.caption("Runs 6 scenarios on fresh demo data (this resets the database).")
    if st.button("Run evaluation"):
        rows = []
        for name, text, p, conflict, expect in graph.SCENARIOS:
            db.reset()
            out = graph.run_scenario(agent, text, p, conflict)
            doc = out["bids"][0]["doctor"] if out.get("outcome") == "booked" else None
            ok = out.get("outcome") == expect["outcome"] and doc == expect["doctor"]
            rows.append({"scenario": name, "expected": f"{expect['outcome']} / {expect['doctor']}",
                         "actual": f"{out.get('outcome')} / {doc}", "result": "PASS ✅" if ok else "FAIL ❌"})
        db.reset()
        ss.chat, ss.state = [], None
        st.dataframe(pd.DataFrame(rows), use_container_width=True)