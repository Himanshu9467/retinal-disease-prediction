# Cleanup Report

## Verified Generated Files

These paths are generated/runtime artifacts and should be excluded from GitHub:

- `venv/`
- `frontend/node_modules/`
- `frontend/dist/`
- `__pycache__/` folders
- `uploads/`
- `api/uploads/`
- `outputs/gradcam/`
- `*.db`

## Removed

- Project Python bytecode caches under:
  - `api/__pycache__/`
  - `database/__pycache__/`
  - `src/__pycache__/`
  - `src/ensemble/__pycache__/`
- QA smoke-test upload image generated during this audit.
- QA smoke-test GradCAM image generated during this audit.

## Not Removed

The following were not deleted because they may be useful for demo, reproducibility, or because the repository has no `.git` metadata to confirm intent:

- `outputs/*.pth`, `outputs/*.json`, and confusion matrices: production artifact source.
- `research_patient_safe_aug_dataset/`: contains patient-safe split and audit artifacts.
- `research_aug_dataset/`: likely experiment/dataset artifact; not deleted because usage intent is uncertain.
- Root training/preparation scripts such as `patient_safe_split.py`, `rebalance_cleaned.py`, `generate_research_reports.py`, `data_aug.py`, `final.py`, and `last.py`: not imported by production paths, but retained because they may document research workflow.
- Existing non-QA files in `uploads/` and `api/uploads/`: retained because they may support demo/history.

## GitHub Recommendation

Publish source code, documentation, `requirements.txt`, `frontend/package*.json`, and the validated `outputs/` production artifacts if file size policy allows. Exclude local virtual environments, node modules, runtime uploads, generated GradCAM cache, SQLite databases, and build output.
