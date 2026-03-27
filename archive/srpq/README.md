# Archived SRPQ Work

This folder contains the earlier `SRPQ` line of the project.

Files here are kept for provenance, not as the recommended current path.

Included:

- `srpq_compress.py`
- `srpq_validate.py`
- `srpq_hybrid.py`
- `srpq_v11_candidate.py`

Why these were archived:

- the low-rank + sparse decomposition did not hold up as the primary KV codec on real model data
- the spectral machinery turned out to be more useful as a selector feature than as a standalone codec
- the project later pivoted to `CARP`, then further to a diagnostics-first framing

Use these only if you want to trace the historical development of the repo.
