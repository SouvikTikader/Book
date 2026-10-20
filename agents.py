"""UNIT 4 MODULE: multi-agent system using the Contract Net Protocol over FIPA-ACL style messages.


"""
from dataclasses import dataclass, field
import asyncio
import concurrent.futures
import time

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


# ---------------------------------------------------------------- doctor-side agent
class DoctorAgent:
    """Autonomous agent. The ONLY component allowed to read/write its own calendar."""

    def __init__(self, doc):
        self.id = doc["id"]
        self.name = doc["name"]
        self.specialty = doc["specialty"]
        self.label = "dr_" + doc["name"].split()[-1].lower()
        self._holds: dict[int, float] = {}   # slot_id -> expiry (epoch seconds)

    # -- calendar (private) ----------------------------------------------------
    def _my_slots(self, day, after):
        return db.get_slots(self.id, day, after)

    def _is_held(self, slot_id):
        exp = self._holds.get(slot_id)
        if exp is None:
            return False
        if exp < time.time():
            self._holds.pop(slot_id, None)
            return False
        return True

    # -- public API ------------------------------------------------------------
    def handle(self, msg):
        if msg.performative == "cfp":
            slots = self._my_slots(msg.content["day"], msg.content["after"])
            free = [s for s in slots if not self._is_held(s["id"])]
            if not free:
                return ACL("refuse", self.label, msg.sender, {"reason": "no free slot"})
            s = free[0]
            return ACL("propose", self.label, msg.sender,
                       {"slot_id": s["id"], "time": s["time"], "day": s["day"]})
        return ACL("refuse", self.label, msg.sender,
                   {"reason": f"unsupported performative {msg.performative}"})

    def reserve(self, slot_id, ttl=180):
        """Hold the slot while the human decides."""
        self._holds[slot_id] = time.time() + ttl

    def release(self, slot_id):
        self._holds.pop(slot_id, None)


# ---------------------------------------------------------------- agent registry
_AGENT_REGISTRY: dict[int, DoctorAgent] = {}


def get_doctor_agent(doc):
    if doc["id"] not in _AGENT_REGISTRY:
        _AGENT_REGISTRY[doc["id"]] = DoctorAgent(doc)
    return _AGENT_REGISTRY[doc["id"]]


def reset_agents():
    _AGENT_REGISTRY.clear()


def release_hold(doctor_id, slot_id):
    ag = _AGENT_REGISTRY.get(doctor_id)
    if ag:
        ag.release(slot_id)


# ---------------------------------------------------------------- message bus
class MessageBus:
    """In-process async pub/sub. Swap for Redis/NATS without changing the agents."""

    def __init__(self):
        self._queues: dict[str, asyncio.Queue] = {}

    def register(self, label):
        self._queues[label] = asyncio.Queue()
        return self._queues[label]

    async def send(self, msg: ACL):
        q = self._queues.get(msg.receiver)
        if q is not None:
            await q.put(msg)

    async def receive(self, label, timeout):
        q = self._queues.get(label)
        if q is None:
            return ACL("refuse", label, "manager", {"reason": "unregistered"})
        try:
            return await asyncio.wait_for(q.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return ACL("refuse", label, "manager", {"reason": "timeout"})


# ---------------------------------------------------------------- sync wrapper
def _run_coro(coro):
    """Run a coroutine from synchronous code — works whether or not a loop is already running."""
    try:
        asyncio.get_running_loop()
        in_loop = True
    except RuntimeError:
        in_loop = False
    if not in_loop:
        return asyncio.run(coro)

    def _runner():
        return asyncio.run(coro)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_runner).result()


# ---------------------------------------------------------------- manager agent
class ManagerAgent:
    label = "manager"

    def __init__(self, window=60, timeout=2.0):
        self.window = window          # minutes of tolerance for memory re-ranking
        self.timeout = timeout        # seconds to wait for each doctor reply

    def contract_net(self, specialty, day, after, patient_id):
        return _run_coro(self._contract_net_async(specialty, day, after, patient_id))

    async def _contract_net_async(self, specialty, day, after, patient_id):
        log, bids = [], []
        docs = db.search_doctors(specialty)
        if not docs:
            return [], log
        agents = [get_doctor_agent(d) for d in docs]

        bus = MessageBus()
        bus.register(self.label)
        listener_tasks = []
        for ag in agents:
            bus.register(ag.label)
            listener_tasks.append(asyncio.create_task(self._doctor_listener(ag, bus)))

        try:
            # 1. call for proposals
            for ag in agents:
                cfp = ACL("cfp", self.label, ag.label,
                          {"task": "book", "day": day, "after": after})
                log.append(str(cfp))
                await bus.send(cfp)

            # 2. collect replies (with deadline)
            raw_replies = []
            for _ in agents:
                raw_replies.append(await bus.receive(self.label, self.timeout))
            by_sender = {r.sender: r for r in raw_replies}

            # 3. evaluate
            for ag in agents:
                reply = by_sender.get(
                    ag.label,
                    ACL("refuse", ag.label, self.label, {"reason": "timeout"}),
                )
                log.append(str(reply))
                if reply.performative == "propose":
                    bids.append({
                        "doctor_id": ag.id,
                        "doctor": ag.name,
                        "label": ag.label,
                        "slot_id": reply.content["slot_id"],
                        "time": reply.content["time"],
                        "day": day,
                        "minutes": to_min(reply.content["time"]),
                        "mem": memory.doctor_score(patient_id, ag.id, ag.name),
                    })
        finally:
            for t in listener_tasks:
                t.cancel()
            await asyncio.gather(*listener_tasks, return_exceptions=True)

        if not bids:
            return [], log

        earliest = min(b["minutes"] for b in bids)
        for b in bids:
            b["in_window"] = b["minutes"] - earliest <= self.window
        bids.sort(key=lambda b: (0, -b["mem"], b["minutes"]) if b["in_window"] else (1, 0, b["minutes"]))
        top = bids[0]
        if top["minutes"] == earliest:
            top["why"] = (f"earliest slot across {len(bids)} cardiology bids"
                          if specialty == "cardiology"
                          else f"earliest slot across {len(bids)} bids")
        else:
            top["why"] = (f"{top['minutes'] - earliest} min after the earliest bid, but preferred by "
                          f"this patient's memory (score {top['mem']:+d})")

        # 4. award (winner holds the slot until the human approves)
        winner = _AGENT_REGISTRY.get(top["doctor_id"])
        if winner:
            winner.reserve(top["slot_id"])
        log.append(str(ACL("accept-proposal", self.label, top["label"], {"slot_id": top["slot_id"]})))
        for b in bids[1:]:
            log.append(str(ACL("reject-proposal", self.label, b["label"], {"reason": "not awarded"})))
        return bids, log

    async def _doctor_listener(self, ag: DoctorAgent, bus: MessageBus):
        q = bus._queues[ag.label]
        try:
            while True:
                msg = await q.get()
                reply = ag.handle(msg)
                await bus.send(reply)
        except asyncio.CancelledError:
            pass