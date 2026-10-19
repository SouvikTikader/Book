"""UNIT 4 MODULE: multi-agent system using the Contract Net Protocol over FIPA-ACL style messages.

Roles:
  ManagerAgent  - announces the task (cfp), evaluates proposals, awards the contract
  DoctorAgent   - one autonomous agent per doctor; owns its calendar; replies propose / refuse
Message performatives used: cfp, propose, refuse, accept-proposal, reject-proposal
"""
from dataclasses import dataclass, field

import db
import memory


@dataclass
class ACL:
    performative: str
    sender: str
    receiver: str
    content: dict = field(default_factory=dict)

    def __str__(self):
        return f"({self.performative} :sender {self.sender} :receiver {self.receiver} :content {self.content})"


def to_min(t):
    h, m = t.split(":")
    return int(h) * 60 + int(m)


class DoctorAgent:
    def __init__(self, doc):
        self.id, self.name, self.specialty = doc["id"], doc["name"], doc["specialty"]
        self.label = "dr_" + doc["name"].split()[-1].lower()

    def handle(self, msg):
        if msg.performative == "cfp":
            slots = db.get_slots(self.id, msg.content["day"], msg.content["after"])  # tool call on own calendar
            if not slots:
                return ACL("refuse", self.label, msg.sender, {"reason": "no free slot"})
            s = slots[0]
            return ACL("propose", self.label, msg.sender,
                       {"slot_id": s["id"], "time": s["time"], "day": s["day"]})


class ManagerAgent:
    label = "manager"

    def __init__(self, window=60):
        self.window = window  # minutes of tolerance in which memory may re-rank "earliest"

    def contract_net(self, specialty, day, after, patient_id):
        log, bids = [], []
        doctors = [DoctorAgent(d) for d in db.search_doctors(specialty)]
        for ag in doctors:  # 1. call for proposals
            cfp = ACL("cfp", self.label, ag.label, {"task": "book", "day": day, "after": after})
            log.append(str(cfp))
            reply = ag.handle(cfp)  # 2. participants respond
            log.append(str(reply))
            if reply.performative == "propose":
                bids.append({"doctor_id": ag.id, "doctor": ag.name, "slot_id": reply.content["slot_id"],
                             "time": reply.content["time"], "day": day, "minutes": to_min(reply.content["time"]),
                             "mem": memory.doctor_score(patient_id, ag.id, ag.name), "label": ag.label})
        if not bids:
            return [], log
        earliest = min(b["minutes"] for b in bids)
        for b in bids:
            b["in_window"] = b["minutes"] - earliest <= self.window
        # 3. evaluate: inside the tolerance window, patient memory decides; otherwise strictly earliest
        bids.sort(key=lambda b: (0, -b["mem"], b["minutes"]) if b["in_window"] else (1, 0, b["minutes"]))
        top = bids[0]
        if top["minutes"] == earliest:
            top["why"] = f"earliest slot across {len(bids)} cardiology bids" if specialty == "cardiology" \
                else f"earliest slot across {len(bids)} bids"
        else:
            top["why"] = (f"{top['minutes'] - earliest} min after the earliest bid, but preferred by this patient's "
                          f"memory (score {top['mem']:+d})")
        # 4. award
        log.append(str(ACL("accept-proposal", self.label, top["label"], {"slot_id": top["slot_id"]})))
        for b in bids[1:]:
            log.append(str(ACL("reject-proposal", self.label, b["label"], {"reason": "not awarded"})))
        return bids, log
