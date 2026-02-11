# Copyright 2026 Nearby Computing S.L.

**EUCNC Simulations — Functionality & User Guide**

Overview
- **What:** Two Python simulation scripts used for the EUCNC paper. They generate results (CSV/JSON) and plots (PDF) for migration and energy evaluations.
- **Location:** [EUCNC](EUCNC)

Files
- `EUCNC/EUCNC_2025_Migration_affected_users_simulation.py`: migration impact simulation. Outputs `migration_results.csv` and `users_affected_migration_comparison_percentage.pdf`.
- `EUCNC/EUCNC_2025_NASO_Energy_simulation.py`: energy and utilization simulation. Outputs `energy_reduction_log.json` and `energy_reduction_comparison.pdf`.

Prerequisites
- Python 3.9+ recommended
- Python packages: `numpy`, `matplotlib`, `seaborn` (see `EUCNC/requirements.txt`)

Quick setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r EUCNC/requirements.txt
```

Run the simulations
- Migration script:

```bash
python3 EUCNC/EUCNC_2025_Migration_affected_users_simulation.py
```

- Energy script:

```bash
python3 EUCNC/EUCNC_2025_NASO_Energy_simulation.py
```

Notes
- Both scripts are self-contained and use random seeds for reproducibility (see top of each file).
- Outputs are saved in the working directory. Open the generated PDF files for visual results and the CSV/JSON files for raw numbers.
- If running headless (no X server), set matplotlib backend or run in an environment that supports display; the scripts save PDFs so headless runs should still succeed.
