# what/ — Agent Guide

## What's Here

Knowledge, research findings, experiment results, and architecture decisions for AIgendetector.

## Working Rules

- Decisions → `what/decisions/` (use ADR format)
- Context summaries → `what/context/`
- Reference docs → `what/docs/`
- Required frontmatter: `type`, `created`, `updated`, `last_edited_by`, `tags`
- Read before write; set `updated` and `last_edited_by` on every edit

## Existing Documentation (at project root — do not move)

| File | Topic |
|------|-------|
| `../../EXPERIMENTS_SUMMARY.md` | Research results and experiment tracking |
| `../../EXPERIMENTS_GUIDE.md` | Experiment methodology |
| `../../SAE_INTERPRETABILITY.md` | Sparse autoencoder explanation |
| `../../SPECTRAL_ANALYSIS_README.md` | Spectral analysis methodology |
| `../../APP_SUMMARY.md` | Frontend application overview |

These live at the project root by convention. Reference them from here rather than moving them.

## When to Add Files Here

- A new architecture decision needs documenting (ADR)
- An experiment produces findings worth preserving
- You want to create a context file summarizing a domain concept for future agents
