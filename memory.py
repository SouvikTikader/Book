"""UNIT 3 MODULE: memory + vector retrieval + feedback-based adaptation.

- Short-term memory : the LangGraph state for the current request (graph.py)
- Long-term memory  : SQLite tables `memory` and `feedback`, persisted across sessions
- Vector retrieval  : symptom -> specialty lookup over an embedded knowledge base (RAG)
- Feedback adaptation: accepted/rejected proposals change future ranking
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import db

# ---------------------------------------------------------------- vector store (RAG)
KB = [
    ("cardiology", "chest pain palpitations racing heart shortness of breath high blood pressure angina irregular heartbeat cardiac"),
    ("neurology", "headache migraine dizziness seizures numbness memory loss tingling tremor stroke"),
    ("dermatology", "skin rash itching acne eczema hair loss allergy redness psoriasis"),
    ("orthopedics", "bone fracture joint pain knee back pain sprain shoulder arthritis spine"),
    ("pediatrics", "child baby infant newborn toddler kids vaccination growth fever in children"),
    ("general", "fever cold cough flu weakness body ache stomach infection checkup"),
]


class VectorStore:
    """TF-IDF vectors + cosine similarity. Swap for ChromaDB + nomic-embed-text with the same search() API."""

    def __init__(self, docs):
        self.docs = docs
        self.vec = TfidfVectorizer(ngram_range=(1, 2)).fit([d[1] for d in docs])
        self.matrix = self.vec.transform([d[1] for d in docs])

    def search(self, query, k=1):
        sims = cosine_similarity(self.vec.transform([query]), self.matrix)[0]
        order = sims.argsort()[::-1][:k]
        return [(self.docs[i][0], float(sims[i])) for i in order]


_store = None


def retrieve_specialty(symptoms, threshold=0.08):
    global _store
    if _store is None:
        _store = VectorStore(KB)
    spec, score = _store.search(symptoms)[0]
    return (spec, score) if score >= threshold else (None, score)


# ---------------------------------------------------------------- long-term memory
def get_memory(pid):
    c = db.conn()
    out = {r["key"]: r["value"] for r in c.execute("SELECT key,value FROM memory WHERE patient_id=?", (pid,))}
    c.close()
    return out


def set_memory(pid, key, value):
    c = db.conn()
    c.execute("INSERT OR REPLACE INTO memory VALUES (?,?,?)", (pid, key, value))
    c.commit()
    c.close()


def feedback_counts(pid, doctor_id):
    c = db.conn()
    rows = c.execute("SELECT event, COUNT(*) n FROM feedback WHERE patient_id=? AND doctor_id=? GROUP BY event",
                     (pid, doctor_id)).fetchall()
    c.close()
    d = {r["event"]: r["n"] for r in rows}
    return d.get("accepted", 0), d.get("rejected", 0)


# ---------------------------------------------------------------- feedback-based adaptation
def record_feedback(pid, doctor_id, event):
    c = db.conn()
    c.execute("INSERT INTO feedback(patient_id,doctor_id,event) VALUES (?,?,?)", (pid, doctor_id, event))
    c.commit()
    c.close()
    acc, _ = feedback_counts(pid, doctor_id)
    if event == "accepted" and acc >= 2:  # learned preference after repeated acceptance
        set_memory(pid, "preferred_doctor", db.get_doctor(doctor_id)["name"])


def doctor_score(pid, doctor_id, doctor_name):
    """+2 preferred doctor, +1 per past acceptance (max 3), -2 per past rejection."""
    acc, rej = feedback_counts(pid, doctor_id)
    score = min(acc, 3) - 2 * rej
    if get_memory(pid).get("preferred_doctor") == doctor_name:
        score += 2
    return score