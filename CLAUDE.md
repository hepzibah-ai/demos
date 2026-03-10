# CLAUDE.md — demos

## Purpose

Public interactive tutorial notebooks (marimo). No proprietary IP — H0
implementation details stay in sim0.

## On session start

Read `technical-debt.md` for the full notebook plan and roadmap.

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

## Adding a new notebook

1. Create `<name>_demo.py` at repo root with PEP 723 metadata
2. Add `.with_app(path="/<route>", root="/app/notebooks/<name>_demo.py")` in `deploy/server.py`
3. If new pip deps needed, add to `deploy/Dockerfile`
4. If heavy downloads needed (models, data), add `RUN python -c "..."` pre-download step
5. Update `technical-debt.md` status table
6. Push, then rebuild on Linode
