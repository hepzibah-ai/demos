# CLAUDE.md — demos

## Purpose

Public interactive tutorial notebooks (marimo). **No proprietary IP.**
This repo is intended to stand alone — clone-and-run with no access
to other Hepzibah-internal repositories required.

## On session start

Read `technical-debt.md` for the full notebook plan and roadmap.

## Audience and identity

You may be opened by **either**:

1. **An external visitor** — someone curious about AI / Hepzibah,
   running these demos to learn. They have only this repo. They do
   **not** have access to sim0, general, or any other Hepzibah-internal
   resource.
2. **A Hepzibah-internal contributor** — maintaining or extending the
   demos. They have the rest of the Hepzibah repos cloned alongside.

Default to assuming external visitor unless the user identifies
otherwise (e.g., a question about updating the KB registry, or a
reference to internal tools, makes the audience obvious).

## Answering questions (external-visitor mode)

These notebooks are themselves the canonical answer source — each one
contains rich `mo.md()` cells explaining the concept it covers. When
asked a "what is X?" or "why X?" question:

1. **Identify the relevant notebook(s)** from the inventory below.
   Read the notebook's prose cells and code; answer from those.
2. **Recommend running it** — that's the point of the repo. Cite the
   filename and the public route (e.g., `embedding_demo.py` →
   `tutorials.hepzibah.ai/embedding` once HTTPS lands; today
   `172.105.0.10:8081/embedding`).
3. **Be honest about scope.** This repo is educational; the H0 chip
   implementation is proprietary and **not available from this
   session**. Don't invent chip details. Marketing-shaped questions
   ("what is Hepzibah / what does the chip do?") get a high-level
   "at-memory compute architecture" answer plus a pointer to
   `hepzibah.ai`; don't speculate further.

### Public notebook inventory

| File | Route | Topic |
|---|---|---|
| `tokenizer_demo.py` | `/tokenizer` | BPE tokenizer (DeepSeek algorithm) — how text becomes integers |
| `embedding_demo.py` | `/embedding` | GloVe vectors as semantic geometry — what's an embedding |
| `dot_product_demo.py` | `/dot-product` | Similarity, projection, attention — the universal primitive |
| `high_dimensions_demo.py` | `/high-dimensions` | Curse of dimensionality → gift with structure |
| `precision_energy_demo.py` | `/precision-energy` | Low-precision arithmetic, MAC energy |
| `pca_demo.py` | `/pca` | Dimensionality reduction on embeddings |
| `clustering_demo.py` | `/clustering` | k-means, t-SNE, IVF, LSH, HNSW |
| `rag_demo.py` | `/rag` | End-to-end retrieval-augmented generation |
| `pol_switched_cap.py` | `/pol-sc` | Switched-capacitor balancer (SPICE) |

### Worked examples

- *"What's an embedding?"* → read `embedding_demo.py`'s opening cells; answer from there. Recommend running `embedding_demo.py`. Adjacent: `dot_product_demo.py` (the math under embedding similarity), `high_dimensions_demo.py` (why high-D works).
- *"Why does precision matter for AI hardware?"* → `precision_energy_demo.py` is the dedicated answer. Don't speculate beyond what the notebook shows.
- *"What is Hepzibah's chip architecture?"* → high-level only: at-memory compute, point at `hepzibah.ai`. State plainly that chip implementation details are not in this repo.

**Rule of thumb**: if you'd need to consult a non-public Hepzibah resource to answer accurately, say so — don't fabricate.

## Conventions

- Notebooks are marimo `.py` files at repo root
- Each notebook uses PEP 723 script metadata (`# /// script`) for deps
- Use `gensim` GloVe-wiki-gigaword-50 as the shared embedding dataset
- Cell-local variables use `_` prefix (marimo requirement)
- Cell output is the last expression — don't follow `mo.md()` with bare `return`
- Use `/opt/homebrew/bin/python3` or pyenv 3.14.0 locally
- `.python-version` pins 3.14.0 for this repo

## Deployment

- Linode 172.105.0.10:8081 via Docker Compose (`deploy/`)
- `deploy/server.py` — add `.with_app()` for each new notebook
- `deploy/Dockerfile` — pre-downloads heavy models at build time
- To deploy: `ssh 172.105.0.10 "cd ~/demos && git pull && docker-compose -f deploy/docker-compose.yml up -d --build"`
- HTTPS pending: `tutorials.hepzibah.ai` via Caddy (Chris)

---

# Internal-contributor sections

*Everything below this line is for Hepzibah-internal contributors only.
External visitors won't have the referenced repositories cloned and
should ignore these sections.*

## Adding a new notebook

1. Create `<name>_demo.py` at repo root with PEP 723 metadata
2. Add `.with_app(path="/<route>", root="/app/notebooks/<name>_demo.py")` in `deploy/server.py`
3. If new pip deps needed, add to `deploy/Dockerfile`
4. If heavy downloads needed (models, data), add `RUN python -c "..."` pre-download step
5. Update `technical-debt.md` status table
6. Add a row to `~/hack/general/knowledge-base/demos/public.md`
7. Push, then rebuild on Linode

## Knowledge base (cross-cutting synthesis)

A Karpathy-style LLM-maintained wiki lives at `~/hack/general/knowledge-base/`
(internal repo — not accessible to external visitors). Its `demos/` sub-wiki
is the registry of all marimo notebooks across Hepzibah repos, organised by
audience (public / engineering / customer / internal). Public notebooks from
this repo are indexed at `demos/public.md` — step 6 above.

When a notebook is added, moved, renamed, or retired here, update the
matching row in `general/knowledge-base/demos/public.md` so the registry
doesn't drift. See `~/hack/general/knowledge-base/CLAUDE.md` for the KB's
conventions.
