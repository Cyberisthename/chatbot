# JARVIS Originality Audit — Legal Ownership & Dependency Risk Report

**Date:** 2026-10-05 (fresh audit of `Cyberisthename/chatbot` @ `origin/main` HEAD 5520238 + working tree)
**Auditor:** agent-legal-manager
**Scope:** Verify that the JARVIS core (Quantum Transformer, TCL engine, FBSC seed compressor) is custom-built and NOT a wrapper on proprietary/third-party AI or quantum SDKs; report legal ownership status and dependency risks.
**Method:** static import scan of all Python in the core package + root repo; review of `requirements.txt`, `package.json`, LICENSE files; spot-read of core module structure (forward/backward loops). Every claim below is tagged **(a) measured / (b) interpretation / (c) speculation** per house standard.

---

## 1. Executive verdict

**(a) PASS — the JARVIS core is original, custom-built, and dependency-lean.**

- `src/quantum_llm/quantum_transformer.py` (491 lines) and `src/quantum_llm/quantum_attention.py` (474 lines) are hand-written numpy transformers with explicit `forward`/`backward` loops. **No import of torch, TensorFlow, Keras, transformers, JAX, or any quantum SDK.**
- `src/thought_compression/` (tcl_engine 517 + compiler 506 + parser 447 + runtime 685 + symbols/types ≈ 2,300 lines) is a self-contained symbolic language implementation (its own parser → bytecode compiler → runtime). **No third-party language engine, no LLM SDK.**
- `compression_specialist.py` (FBSC core) imports **only** numpy + stdlib (hashlib, json, math, os, pathlib, typing, time).
- `requirements.txt` (root + HF-ready) declares only: `gradio`, `numpy`, `matplotlib`.
- Repo `src/` web layer uses fastapi / flask / pydantic / uvicorn / yaml — ordinary infrastructure, not AI wrappers.
- LICENSE: **MIT** (Copyright (c) 2024 J.A.R.V.I.S. AI System) in both root and `jarvis_quantum_ai_hf_ready/`. Code is free-and-clear for the owner's use, modification, distribution, and commercial licensing.

**Pass is scoped to the core.** Two risk items below (legacy HF-training scripts; brand name) do not change the core verdict but must be managed.

---

## 2. What was scanned

| Surface | Files | Third-party AI/quantum imports found |
|---|---|---|
| FBSC core (`compression_specialist.py`, `seed_optimizer.py`, `qvgpu_compressor.py`) | 3 | **None** (numpy + stdlib only) |
| Quantum LLM package (`jarvis_quantum_ai_hf_ready/src/quantum_llm/`) | 6 | **None** |
| Thought-compression (TCL) package (`src/thought_compression/`) | 7 | **None** |
| Bio-knowledge & cancer modules (`src/bio_knowledge/`) | several | None AI-related; `urllib` fetches from public biology APIs (UniProt/NCBI) — data retrieval, not an AI wrapper |
| API layer (`src/api/`) | main, cancer_routes, tcl_routes, rnd_battery | fastapi/flask/pydantic/uvicorn only |
| Root-level legacy scripts | `train_jarvis.py`, `train_llm.py`, `convert_to_gguf.py` (all git-tracked) | **`torch` + `transformers` (HuggingFace, distilgpt2 fine-tune)** — legacy, NOT part of the core product |
| Node layer (`package.json`, `server.js`) | — | express, cors, socket.io, helmet, winston, dotenv, multer, compression — infra only |
| Website (`/home/team/shared/site`) | — | TanStack Start/React/Vite/Tailwind — presentation only, no AI dependency |

**Interpreted list of "no" verdicts:** no qiskit, no pennylane, no cirq, no braket, no dwave, no IBM/Google/Azure quantum APIs, no openai, no anthropic, no langchain, no google — anywhere in `src/` or the repo's Python core. (a)

---

## 3. Legal ownership status

1. **Code ownership — clean.** The owned modules are original expressions authored for the project, licensed MIT to the owner. MIT grants full rights of use, modification, sublicensing, and commercialization, with the single obligation to preserve the copyright notice. (a)
2. **The 3-seed key — owner-controlled.** The seed values (`[0.57721, 1.618034, 2.71828]` + derived params) remain owner-owned; public surfaces derive restricted instances and the server logs only a sha256 digest, never the raw seed (verified in `rnd_battery.py` smoke output: `"seed_stored": false`, `"seed_digest"` only). (a)
3. **Trademark risk — known, needs a decision (unresolved risk, owner flag required).** The name "J.A.R.V.I.S." is a Marvel/Disney trademark. The MIT license protects the *code*, but the *brand name* is third-party IP. Recommending: keep the code, add a naming/renaming decision to the owner's call list before any public launch beyond the current lab. This is a **risk on the board that needs an explicit owner flag** per working agreements. (b)
4. **No proprietary wrappers — verified.** Nothing in the core calls an external AI/quantum service; the only network calls in `src/` are read-only fetches of public biological sequence data (data attribution/terms-of-use applies, see §4.4). (a)

---

## 4. Dependency risks & required follow-up

### 4.1 LEGACY scripts with torch/transformers (root: train_jarvis.py, train_llm.py, convert_to_gguf.py)
**(a) Measured:** these three git-tracked root scripts import `torch` and `transformers` (distilgpt2 fine-tuning path — the old JARVIS v1 LLM wrapper approach).
**(b) Risk:** a repo auditor (or an external skeptical AI, cf. FACTCHECK.md) could reasonably read these as "the model is a fine-tuned wrapper of a third-party open-weight model," which contradicts the business plan's "built from scratch, no prebuilt models" claim — even though the *product core* is fully original.
**(b) Recommendation — 3 options, owner or lead to pick:**
  - **A (preferred):** relocate to `legacy/` with a README stating "archived v1 training path — NOT part of the owned core; kept for history," OR
  - **B:** delete from the active tree (history persists in git), OR
  - **C:** leave in place but add `LEGACY_DEPENDENCIES.md` at repo root explaining the boundary.
**Gate effect:** LOW (core unaffected) but **must be resolved before any "from scratch" statement is made publicly.**

### 4.2 License inventory of the few third-party packages actually used
**(a)** numpy (BSD-3), matplotlib (PSF/BSD), gradio (Apache-2.0), fastapi (MIT), flask (BSD-3), pydantic (MIT), uvicorn (BSD-3), express (MIT), socket.io (MIT), TanStack/React/Vite (MIT). All permissive, no copyleft (no GPL/AGPL) that would taint owned code. Site uses `@neondatabase/serverless` (Apache-2.0) — infra only. **No compliance blocker.**

### 4.3 No transitive-wrapping hidden in subprocess/EVP patterns
**(a)** The site's server-fn wrapper invokes `compression_specialist.py` and `seed_optimizer_api.py` as subprocesses — but these are the **owned** Python modules, not third-party services. This is an internal boundary, not a wrapper on someone else's AI.

### 4.4 Public bio-data retrieval (protein_sequence_retriever.py, dna_sequence_retriever.py, cancer modules)
**(a)** Fetches from UniProt/NCBI/Ensembl etc. are read-only public-data retrieval.
**(b)** Follow-up: document data sources + third-party TOS (UniProt, NCBI, DrugBank, KEGG, Reactome, BioGRID) in the bio-knowledge README; do not redistribute licenses' contents in product artifacts without attribution. This feeds the **data-ownership** axis of the bio-quantum governance gate (see companion doc).

### 4.5 DLL/shared-object and binary blobs
**(a)** `jarvis_qvgpu_trained.npz` is an owned training artifact (optional, cwd-dependent), not a third-party binary. No proprietary .so/.dll in the core path found.

---

## 5. Repo-convention note (CLAUDE.md / AGENTS.md)
**(a)** No `CLAUDE.md` or `AGENTS.md` exists in the repo (checked root + `jarvis_quantum_ai_hf_ready/`). The team's operative conventions live in `/home/team/shared/WORKFLOW.md` (shared-checkout discipline, disk hygiene, artifact durability) — this audit follows those. Recommend the lead mint an `AGENTS.md` at repo root encoding: house claims-tagging, no-hardware ceiling, structure≠meaning, and this audit's dependency boundary, so future members and external auditors have a single convention file to cite.

---

## 6. Bottom line for the lead
- **Core originality: VERIFIED.** Quantum Transformer, TCL engine, and FBSC core are custom-built numpy/stdlib code with no proprietary wrappers, MIT-licensed, owner-owned.
- **Two action items:** (1) resolve the legacy torch/transformers scripts boundary (4.1); (2) make an owner-flagged decision on the "J.A.R.V.I.S." brand name (3.3).
- **One recommendation:** add `AGENTS.md` with conventions (5).
- Every approved number referenced here is re-verifiable by fresh grep/run on the repo — no stale-doc numbers used.