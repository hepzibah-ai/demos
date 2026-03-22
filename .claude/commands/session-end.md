# Session End

Prepare for handoff before ending a session.

## Checklist

### 1. Commit and Push

- [ ] All changes committed with clear messages
- [ ] Pushed to GitHub

### 2. Update Tracking

- [ ] Update `technical-debt.md` with status changes, new items, or completed work
- [ ] If new notebook added, verify it's registered in `deploy/server.py`

### 3. Verification

```bash
git status          # Nothing to commit
git push            # Up to date
```

## Report

Before ending, report:
- What was accomplished
- What was committed (commit hashes)
- Any items added to technical debt
- Any blockers for next session
