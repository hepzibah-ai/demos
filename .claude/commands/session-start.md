# Session Start

Sync and load context for a demos session.

## Steps

### 1. Sync

```bash
git fetch origin
git status
```

**If behind remote**: `git pull` and review what changed.
**If diverged**: Stop and report to user.

### 2. Read Context

1. `technical-debt.md` — notebook roadmap and status
2. `deploy/server.py` — which notebooks are currently deployed

### 3. Report

- Sync status
- Next notebooks to build (from technical-debt.md)
- Any files modified since last session
