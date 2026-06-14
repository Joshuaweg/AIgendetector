# how/ — Agent Guide

## What's Here

Sessions, missions, and operational procedures for AIgendetector work.

## Structure

```
how/
├── sessions/    ← bounded units of agent work (active + archive)
├── missions/    ← multi-session decomposed work with artifacts
└── AGENTS.md   ← this file
```

## Working Rules

- Sessions → `how/sessions/` (filename: `session_YYYYMMDD_topic.md`)
- Close sessions with a SITREP before archiving
- Missions → `how/missions/` for work spanning multiple sessions
- Required frontmatter: `type`, `status`, `created`, `updated`, `last_edited_by`, `tags`
- Read before write; set `updated` and `last_edited_by` on every edit

## Existing Workflows (at project root — do not move)

| File | Purpose |
|------|---------|
| `../../full_train.py` | Main training pipeline |
| `../../sm_train_v3.py` | AWS SageMaker training |
| `../../DEPLOYMENT.md` | Production deployment guide |
| `../../QUICKSTART.md` | Fast API + Next.js setup |

## When to Add Files Here

- Starting a new agent work session (create a session file)
- Beginning multi-session work like a new experiment or feature (create a mission)
