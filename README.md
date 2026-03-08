# Demos

Interactive notebooks for onboarding and education.
No proprietary content — this repo is public.

## Notebooks

| Notebook | Path | What it covers |
|----------|------|----------------|
| [tokenizer_demo.py](tokenizer_demo.py) | `/tokenizer` | DeepSeek's BPE tokenizer — see how LLMs break text into tokens |

## Running locally

```bash
pip install marimo transformers torch
marimo run tokenizer_demo.py
```

## Deploying (Linode)

The tutorial server runs alongside the customer demo server (sim0/deploy)
on 172.105.0.10. Tutorials use port 8081; customer demos use port 8080.

### Quick start

```bash
ssh snelgar@172.105.0.10
cd ~/demos

# Build and run
docker compose -f deploy/docker-compose.yml up -d

# Test
curl -I http://localhost:8081/tokenizer
```

### With HTTPS (via Caddy)

Add to `/etc/caddy/Caddyfile`:

```
tutorials.hepzibah.ai {
    reverse_proxy localhost:8081
}
```

```bash
sudo systemctl reload caddy
```

Caddy auto-provisions a Let's Encrypt cert. Team visits
`https://tutorials.hepzibah.ai/tokenizer`.

### Adding notebooks

1. Add a marimo `.py` file to this repo
2. Add a `.with_app()` line in `deploy/server.py`
3. Rebuild: `docker compose -f deploy/docker-compose.yml up -d --build`

### Architecture

```
Linode 172.105.0.10
├── ~/sim0/     → customer demos (port 8080, token-protected)
└── ~/demos/    → tutorial notebooks (port 8081, behind Caddy auth)
    └── deploy/
        ├── Dockerfile         # Python + marimo + transformers
        ├── server.py          # FastAPI multi-notebook server
        └── docker-compose.yml
```
