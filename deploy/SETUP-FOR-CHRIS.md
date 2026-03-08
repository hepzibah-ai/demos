# Tutorial server — DNS + HTTPS setup

The tutorial server is already running on the Linode at
`http://172.105.0.10:8081/`. This doc covers what's needed to put it
behind `https://tutorials.hepzibah.ai` with proper auth.

## What's already done

- Docker container serving marimo notebooks on port 8081
- Caddy is already installed on the Linode (used by sim0 customer demos on 8080)
- Container auto-restarts (`restart: unless-stopped`)

## What's needed

### 1. DNS record

Add an A record for `tutorials.hepzibah.ai` pointing to `172.105.0.10`.

Where: wherever we manage DNS for hepzibah.ai (same place as the existing
records for the sim0 customer demos).

### 2. Caddy reverse proxy + auto-HTTPS

SSH into the Linode and add a block to `/etc/caddy/Caddyfile`:

```
tutorials.hepzibah.ai {
    reverse_proxy localhost:8081
}
```

Then reload:

```bash
sudo systemctl reload caddy
```

Caddy auto-provisions a Let's Encrypt certificate — no manual cert setup needed.
Verify with `curl -I https://tutorials.hepzibah.ai/tokenizer`.

### 3. Authentication (not urgent, but do it while you're there)

These are internal onboarding tutorials, not public. Options in rough order
of preference:

**a) Caddy basic auth** (simplest, fine for now):

```bash
# Generate a hashed password
caddy hash-password
```

```
tutorials.hepzibah.ai {
    basicauth /* {
        hepzibah <hashed-password>
    }
    reverse_proxy localhost:8081
}
```

Shared credentials are fine for a small team. Put the password in the
usual place (1Password / shared vault).

**b) Caddy + OAuth proxy** (if we want individual logins later):
Use `caddy-security` plugin with Google/GitHub OAuth. More setup but
gives per-user access. Worth doing when we also set up auth for the
sim0 customer demo server.

### 4. Future: customer demo auth

When we're ready, the sim0 customer demos (port 8080) need proper
per-customer auth with token-based access. That's a separate task but
the Caddy + OAuth infrastructure would serve both. Worth planning them
together.

## Current architecture

```
Linode 172.105.0.10
├── Caddy (ports 80/443) ─── reverse proxy + auto-HTTPS
│   ├── tutorials.hepzibah.ai → localhost:8081
│   └── [customer demo domain] → localhost:8080
│
├── ~/demos/    → tutorial notebooks (this repo)
│   └── docker container on port 8081
│
└── ~/sim0/     → customer demos
    └── docker container on port 8080
```

## Useful commands

```bash
# Check tutorial container status
docker ps | grep tutorial

# Rebuild after notebook changes
cd ~/demos
git pull
docker-compose -f deploy/docker-compose.yml up -d --build

# Check Caddy status
sudo systemctl status caddy
sudo journalctl -u caddy --since "10 min ago"

# Test locally
curl -I http://localhost:8081/tokenizer
```
