# Tutorial & Notebook Server — Setup Brief for IT

**Audience**: Chris (IT/infrastructure).
**Last updated**: 2026-03-16.

This describes what we need on the corporate server to host interactive
tutorial notebooks and internal documentation. We're moving off the
temporary Linode (172.105.0.10) onto corporate infrastructure.

---

## What this is

Interactive Python notebooks served as web pages. Engineers open a URL
in their browser and get a live tutorial with plots, sliders, and text.
No software installation needed on the user's laptop — it all runs
server-side.

There are currently two repos that need serving:

| Repo | What | Notebooks | Audience |
|------|------|-----------|----------|
| `demos` | Math/ML tutorial series (tokenizer, embeddings, PCA, etc.) | 9 marimo notebooks | Everyone — onboarding, customer demos |
| `sim0/tutorials` | Chip architecture walkthroughs | TBD (will migrate to marimo) | Engineering team |

Both are pure Python — no GPUs, no CUDA, no heavy compute. The
workload is a web server delivering pre-rendered plots and lightweight
interactive calculations (dot products on 50-dimensional vectors, small
matrix decompositions).

---

## What we need

### 1. A Docker host

The server runs as a single Docker container per repo. Requirements:

| Resource | demos (current) | sim0/tutorials (future) |
|----------|----------------|------------------------|
| Docker image size | ~1 GB | TBD, similar |
| RAM at rest | ~800 MB | TBD |
| RAM under load | ~1.2 GB (GloVe model + concurrent users) | TBD |
| CPU | Negligible (<1% idle, brief spikes on page load) | TBD |
| Disk | ~2 GB (image + model cache) | TBD |
| Ports | 8081 (tutorials) | 8082 or similar |

Total: a **2-core, 4 GB RAM** VM would be comfortable for both. Even
a shared machine with other lightweight services would be fine — the
load is bursty and small.

Docker and Docker Compose are the only dependencies on the host.
(`docker-compose` v2 or `docker compose` — either works.)

### 2. HTTPS reverse proxy

We need a domain (e.g. `tutorials.hepzibah.ai`) with:
- DNS A record → corporate server IP
- HTTPS termination (Let's Encrypt via Caddy is simplest, but any
  reverse proxy works — nginx, Traefik, whatever you already use for
  the CAD tools)
- Reverse proxy: `tutorials.hepzibah.ai/` → `localhost:8081`

Optional: basic auth or OAuth if we want to restrict access. The
notebooks contain no secrets, but they're internal content.

### 3. Git access for deployments

We deploy by pushing to GitHub and pulling on the server. The workflow:

```
Developer laptop                    Server
      │                                │
      │  git push origin main          │
      │──────────────────────────────► │
      │                                │  git pull
      │                                │  docker-compose up -d --build
      │                                │  (rebuilds container, ~30s)
      │                                │
```

The server needs:
- **Git** installed
- **Read access** to `github.com/hepzibah-ai/demos` (deploy key or
  personal access token — whatever you use for the CAD repos)
- A clone of the repo (one-time: `git clone ... ~/demos`)

Currently we SSH in and run the deploy commands manually. If you
want to set up a webhook or CI/CD that's great, but manual is fine
for now — we deploy a few times a week at most.

### 4. The deploy command

After any notebook change, the full deploy is:

```bash
cd ~/demos
git pull
docker-compose -f deploy/docker-compose.yml up -d --build
```

This rebuilds the Docker image (copies new notebook files, ~30 seconds)
and restarts the container with zero downtime. The Docker image
pre-downloads all model files during build, so container startup is
instant.

---

## What's in the Docker image

The container is self-contained — no external dependencies at runtime:

- **Python 3.12** (slim base)
- **marimo** — the notebook runtime (serves notebooks as web apps)
- **FastAPI + uvicorn** — web server
- **gensim** — loads GloVe word vectors (~70 MB model, pre-downloaded)
- **tokenizers + huggingface_hub** — loads DeepSeek tokenizer data
- **numpy, matplotlib, plotly** — computation and plotting
- **ngspice** — circuit simulator (one notebook uses it)

No GPU drivers. No CUDA. No PyTorch/TensorFlow.

---

## Current notebook inventory

All served from a single container on port 8081:

| Route | Title | What it does |
|-------|-------|-------------|
| `/tokenizer` | What's a Token? | Compare tokenizers (GPT-4, DeepSeek) |
| `/embedding` | What's an Embedding? | Explore GloVe word vectors |
| `/dot-product` | The Dot Product | Cosine similarity, projection |
| `/high-dimensions` | High Dimensions | Curse of dimensionality, concentration |
| `/precision-energy` | Precision and Energy | Number formats, MAC energy, custom silicon |
| `/pca` | PCA | Dimensionality reduction on embeddings |
| `/clustering` | Clustering & Search | k-means, t-SNE, IVF, LSH, HNSW |
| `/rag` | RAG: From Search to Answers | Sentence embeddings, chunking, retrieval |
| `/pol-sc` | Switched Capacitor | SPICE simulation (engineering) |

New notebooks get added roughly weekly. Adding one is a code change
(new `.py` file + one line in `server.py`) — the deploy process is
the same.

---

## Migration checklist

- [ ] Provision VM or allocate resources on existing server
- [ ] Install Docker + Docker Compose
- [ ] Set up DNS A record for `tutorials.hepzibah.ai`
- [ ] Set up HTTPS reverse proxy → `localhost:8081`
- [ ] Clone `github.com/hepzibah-ai/demos` with read access
- [ ] First build: `docker-compose -f deploy/docker-compose.yml up -d --build`
  (first build takes ~5 min to download Python packages and models; subsequent
  rebuilds are ~30s due to Docker layer caching)
- [ ] Verify: `curl -I https://tutorials.hepzibah.ai/tokenizer`
- [ ] Optional: basic auth, monitoring, log rotation
- [ ] Decommission Linode 172.105.0.10 (after verifying everything works)

---

## Questions for Chris

1. Do we have a preferred domain? `tutorials.hepzibah.ai` is what we've
   been planning, but open to alternatives.
2. Do you already have a reverse proxy running on the corporate servers
   that we should integrate with, or should we set up Caddy standalone?
3. Any firewall rules needed for inbound HTTPS (port 443)?
4. Should we set up a deploy key, or use an existing GitHub access token?
