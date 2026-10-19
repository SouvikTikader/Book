# Hospital Appointment Agent: Extended Problem Statement (1 page)

## Original problem (unchanged)
"Book the earliest cardiologist appointment tomorrow after 3 PM." The agent must understand the request, identify
missing information, search doctors, check availability, book, notify the patient and retry alternatives.

## Base system
LangGraph state machine: Intent -> Gap detector -> Search/Availability -> Rank -> Human approval -> Book -> Notify,
with a bounded Retry loop (max 3) and waitlist fallback. Python tools do dates, filtering and atomic booking;
the Ollama LLM (llama3.1:8b) only parses intent and writes the SMS.

## Unit 3 extension: Memory, Tools, Environment Interaction (Demo Module 1)
Concepts: long-term memory, vector retrieval (RAG), feedback-based adaptation, tool/function calling.
- Long-term memory (SQLite): preferred doctor, accepted/rejected history persist across sessions.
- Vector retrieval: "chest pain" -> cardiology via embedded knowledge base (cosine similarity).
- Feedback adaptation: each approve/reject updates a per-doctor score; 2 acceptances create a learned preference.
- Demo: same request for Anita (history) vs Ravi (none) gives different doctors; rejecting a proposal lowers that doctor next time.

## Unit 4 extension: Multi-Agent Systems and Collaboration (Demo Module 2)
Concepts: role-based agents, Contract Net Protocol, FIPA-ACL performatives, human-agent collaboration, conflict resolution.
- One autonomous DoctorAgent per doctor + a ManagerAgent (roles/hierarchy).
- Manager sends `cfp`; doctors reply `propose`/`refuse`; manager sends `accept-proposal`/`reject-proposal`.
- Conflict resolution: slot stolen at booking time -> award to next bidder.
- Human-agent collaboration: approve/reject button, explainable "why this slot", medical-advice guardrail, masked phone.
- Demo: ACL message log tab + Simulate Conflict checkbox.

## Evaluation
Six scenarios (memory re-rank, strict earliest, conflict retry, RAG routing, guardrail, waitlist) in the Evaluation tab.

# Slide outline (5 slides)
1. Problem and original scope (request, 7 required capabilities)
2. Architecture: LangGraph flow diagram + tools + Ollama
3. Unit 3 module: memory/RAG/feedback, screenshot of Anita vs Ravi
4. Unit 4 module: Contract Net, ACL log screenshot, retry screenshot
5. Results (evaluation table) and conclusion / future work

# Viva answers
- Why Unit 3 module? Agents without memory repeat themselves; memory + retrieval personalises and grounds decisions, and feedback lets behaviour improve.
- Why Unit 4 module? Each doctor owns its own calendar, so negotiation (Contract Net) is natural; it scales by adding doctor agents and handles conflicts.
- Why not let the LLM decide everything? Small local models are unreliable at date math and tool calls; critical logic is deterministic, the LLM handles language.
- Is TF-IDF a "vector DB"? It is a vector store with cosine retrieval; the VectorStore class can be swapped for ChromaDB + nomic-embed-text with the same search() call.
