# Legacy Dependency Boundary — torch/transformers in the Repository

**Status:** informational boundary statement (option‑C wording, see §4) · **Author:** agent‑legal‑manager · **Date:** 2026‑10‑05
**Claim tags per house standard:** (a) measured / (b) interpretation / (c) speculation.

---

## 1. The owned core (`src/`)

(a) The owned core — the Quantum Transformer (`src/quantum_llm/quantum_transformer.py`), the TCL engine (`src/thought_compression/`), and the FBSC seed compressor — is original custom code written against **numpy + Python stdlib only**. It is MIT‑licensed and imports **no third‑party quantum SDKs and no third‑party LLM SDKs**. This was verified by a fresh import audit on 2026‑10‑05 (`docs/artifacts/compliance/ORIGINALITY_AUDIT.md`) and is re‑verifiable on this repo by fresh grep/run.

## 2. The legacy root scripts (old fine‑tuned‑wrapper path)

(a) Exactly **three** tracked Python files import `torch` and/or `transformers`. Full importers list, verified by grep on this branch:

| file | third‑party imports |
|---|---|
| `train_jarvis.py` | `torch`; `transformers` (AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments); `datasets` |
| `train_llm.py` | `transformers` (AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling); `datasets` (distilgpt2 fine‑tune) |
| `convert_to_gguf.py` | `torch`; `transformers` (AutoModelForCausalLM, AutoTokenizer); `ctransformers` |

(b) Interpretation: these are the old fine‑tuned‑wrapper path (distilgpt2 fine‑tune → GGUF conversion), NOT the owned‑core runtime.

(a) No tracked file imports any of these three scripts (grep for their module names returns nothing outside the files themselves), so nothing in `src/` depends on them.

(a) No tracked dependency manifest declares `torch` or `transformers` (all `requirements*.txt`, `setup.py`, `pyproject.toml`, `Pipfile`, `vercel*.json` are clean; the root `requirements.txt` is gradio/numpy/matplotlib).

(a) No other tracked Python file imports `torch` or `transformers`. The only other occurrence of the string "transformers" is `demos/train_full_quantum_llm_production.py:8` ("Quantum‑inspired transformers with real backpropagation") — a feature‑bullet mention, not an import.

## 3. Scope of "built from scratch"

The "built from scratch" claim in any public README applies to the owned core of §1. The three legacy scripts of §2 are git‑tracked but are NOT part of that core; their third‑party imports describe an abandoned training path and do not make the owned core a wrapper. (b)

## 4. Owner decision — PENDING

The owner is deciding between:

- **(A)** relocate the three scripts to `legacy/` with a boundary README;
- **(B)** delete them from the active tree;
- **(C)** leave them in place with this note.

That decision is **PENDING** as of this date. This note is the wording for option (C); it is **not** a resolution. Until the owner decides, the boundary stands as documented in §1–§2. (a)

## 5. Verification

Each statement in §1–§2 is re‑verifiable on this repo with a fresh
`git grep -nE "(^|[^A-Za-z_])import (torch|transformers)\b" -- '*.py'`
and the import audit in `docs/artifacts/compliance/ORIGINALITY_AUDIT.md`. (a)