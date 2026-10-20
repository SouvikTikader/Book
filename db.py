"""SQLite layer + the hospital 'environment' tools (Unit 3: tools & environment interaction).

WAL mode + busy_timeout=30 s so init_db() and the waitlist watcher can coexist
without 'database is locked' on Windows.
"""
import os
import sqlite3
from datetime import date, timedelta

DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hospital.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS doctors(id INTEGER PRIMARY KEY, name TEXT, specialty TEXT);
CREATE TABLE IF NOT EXISTS slots(id INTEGER PRIMARY KEY AUTOINCREMENT, doctor_id INT, day TEXT, time TEXT, booked INT DEFAULT 0);
CREATE TABLE IF NOT EXISTS patients(id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, phone TEXT);
CREATE TABLE IF NOT EXISTS appointments(id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id INT, doctor_id INT, day TEXT, time TEXT, status TEXT);
CREATE TABLE IF NOT EXISTS memory(patient_id INT, key TEXT, value TEXT, PRIMARY KEY(patient_id, key));
CREATE TABLE IF NOT EXISTS feedback(id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id INT, doctor_id INT, event TEXT);
CREATE TABLE IF NOT EXISTS waitlist(id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id INT, specialty TEXT, day TEXT);
CREATE TABLE IF NOT EXISTS notifications(id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id INT, message TEXT);
"""

DOCTORS = [
    (1, "Dr. Mehta", "cardiology"), (2, "Dr. Rao", "cardiology"),
    (3, "Dr. Iyer", "cardiology"), (4, "Dr. Khan", "cardiology"),
    (5, "Dr. Sharma", "neurology"), (6, "Dr. Patil", "dermatology"),
    (7, "Dr. Joshi", "orthopedics"), (8, "Dr. Desai", "general"),
    (9, "Dr. Kulkarni", "pediatrics"),
]
BLOCK = {1: ("15:00", "16:30"), 2: ("15:00", "15:30"), 3: ("15:00", "23:59"), 4: ("15:00", "16:00")}


def conn():
    """Open a connection with WAL + 30 s busy timeout. Safe for concurrent readers/writers."""
    c = sqlite3.connect(DB, timeout=30.0, isolation_level=None)
    c.row_factory = sqlite3.Row
    # These are persistent settings on the DB file — cheap to re-apply on each connection.
    try:
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA busy_timeout=30000")
        c.execute("PRAGMA synchronous=NORMAL")
    except sqlite3.OperationalError:
        # If another process is currently holding the lock at startup, just proceed —
        # busy_timeout will kick in on the next statement.
        pass
    return c


def day_offset(n):
    return (date.today() + timedelta(days=n)).isoformat()


def _times():
    return [f"{h:02d}:{m:02d}" for h in range(9, 18) for m in (0, 30)]


def _seed_slots(c):
    c.execute("DELETE FROM slots")
    for off in (0, 1, 2):
        d = day_offset(off)
        for doc_id, _, _ in DOCTORS:
            lo, hi = BLOCK.get(doc_id, ("99", "99")) if off == 1 else ("99", "99")
            for t in _times():
                c.execute("INSERT INTO slots(doctor_id,day,time,booked) VALUES (?,?,?,?)",
                          (doc_id, d, t, 1 if lo <= t < hi else 0))


def init_db():
    """Idempotent. Safe to call from multiple processes (uses WAL + busy_timeout)."""
    c = conn()
    try:
        c.executescript(SCHEMA)
        c.executemany("INSERT OR IGNORE INTO doctors VALUES (?,?,?)", DOCTORS)
        if not c.execute("SELECT COUNT(*) FROM patients").fetchone()[0]:
            c.execute("INSERT INTO patients(name,phone) VALUES ('Anita','9876543210')")
            c.execute("INSERT INTO patients(name,phone) VALUES ('Ravi','9123456780')")
            c.execute("INSERT INTO memory VALUES (1,'preferred_doctor','Dr. Mehta')")
            c.executemany("INSERT INTO feedback(patient_id,doctor_id,event) VALUES (?,?,?)",
                          [(1, 1, "accepted"), (1, 1, "accepted"), (1, 2, "rejected")])
        if not c.execute("SELECT COUNT(*) FROM slots WHERE day=?", (day_offset(0),)).fetchone()[0]:
            _seed_slots(c)
    finally:
        c.close()


def reset():
    """Delete the DB *and its WAL/SHM sidecars*, then re-seed."""
    for suffix in ("", "-wal", "-shm"):
        p = DB + suffix
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass
    init_db()


# ------------------------------------------------------------------ TOOLS
def search_doctors(specialty):
    c = conn()
    try:
        return [dict(r) for r in c.execute("SELECT * FROM doctors WHERE specialty=?", (specialty,))]
    finally:
        c.close()


def get_slots(doctor_id, day, after="00:00"):
    c = conn()
    try:
        return [dict(r) for r in c.execute(
            "SELECT * FROM slots WHERE doctor_id=? AND day=? AND time>=? AND booked=0 ORDER BY time",
            (doctor_id, day, after))]
    finally:
        c.close()


def book_slot(slot_id, patient_id):
    """Atomic: the UPDATE only succeeds if the slot is still free."""
    c = conn()
    try:
        cur = c.execute("UPDATE slots SET booked=1 WHERE id=? AND booked=0", (slot_id,))
        if cur.rowcount == 0:
            return {"ok": False, "reason": "slot_taken"}
        s = c.execute("SELECT * FROM slots WHERE id=?", (slot_id,)).fetchone()
        cur = c.execute(
            "INSERT INTO appointments(patient_id,doctor_id,day,time,status) VALUES (?,?,?,?,'confirmed')",
            (patient_id, s["doctor_id"], s["day"], s["time"]),
        )
        return {"ok": True, "appointment_id": cur.lastrowid,
                "doctor_id": s["doctor_id"], "day": s["day"], "time": s["time"]}
    finally:
        c.close()


def simulate_conflict(slot_id):
    c = conn()
    try:
        c.execute("UPDATE slots SET booked=1 WHERE id=?", (slot_id,))
    finally:
        c.close()


def add_waitlist(patient_id, specialty, day):
    c = conn()
    try:
        c.execute("INSERT INTO waitlist(patient_id,specialty,day) VALUES (?,?,?)",
                  (patient_id, specialty, day))
    finally:
        c.close()


def mask_phone(p):
    return ("*" * (len(p) - 4) + p[-4:]) if p and len(p) > 4 else "unknown"


def send_notification(patient_id, message):
    c = conn()
    try:
        c.execute("INSERT INTO notifications(patient_id,message) VALUES (?,?)",
                  (patient_id, message))
    finally:
        c.close()
    p = get_patient(patient_id)
    return f"SMS sent to {mask_phone(p.get('phone'))}"


def get_patient(pid):
    c = conn()
    try:
        r = c.execute("SELECT * FROM patients WHERE id=?", (pid,)).fetchone()
        return dict(r) if r else {}
    finally:
        c.close()


def list_patients():
    c = conn()
    try:
        return [dict(r) for r in c.execute("SELECT * FROM patients")]
    finally:
        c.close()


def create_patient(name):
    c = conn()
    try:
        cur = c.execute("INSERT INTO patients(name,phone) VALUES (?,NULL)", (name,))
        return cur.lastrowid
    finally:
        c.close()


def set_phone(pid, phone):
    c = conn()
    try:
        c.execute("UPDATE patients SET phone=? WHERE id=?", (phone, pid))
    finally:
        c.close()


def get_doctor(did):
    c = conn()
    try:
        r = c.execute("SELECT * FROM doctors WHERE id=?", (did,)).fetchone()
        return dict(r) if r else {}
    finally:
        c.close()


def appointments_for(pid):
    c = conn()
    try:
        return [dict(r) for r in c.execute(
            "SELECT a.id, d.name AS doctor, a.day, a.time, a.status FROM appointments a "
            "JOIN doctors d ON d.id=a.doctor_id WHERE a.patient_id=? ORDER BY a.id DESC", (pid,))]
    finally:
        c.close()


def notifications_for(pid):
    c = conn()
    try:
        return [dict(r) for r in c.execute(
            "SELECT id, message FROM notifications WHERE patient_id=? ORDER BY id DESC LIMIT 20", (pid,))]
    finally:
        c.close()