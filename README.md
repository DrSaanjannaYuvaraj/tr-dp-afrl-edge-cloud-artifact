# Telemetry-Driven AFRL Artifact

This repository contains the code, configuration, sample data, and helper scripts for the telemetry-driven adaptive federated reinforcement learning experiments.

## Repository Structure

- `src/` – core implementation for server experiments, plotting, merging, and client API
- `config/` – configuration files such as `class_map.json` and `scaler.json`
- `data/` – raw per-client CSV files and the merged dataset created locally
- `scripts/` – helper scripts matching the experiment execution workflow
- `docs/` – supplementary artifact notes

## Important Before Running

This public artifact intentionally removes private AWS-specific details.

Before running any script in `scripts/`, update:

1. `CLIENT_ID1` to `CLIENT_ID10` placeholders with your actual client IP addresses.
2. Any local paths if your repo layout differs.
3. The Python environment activation only if you want the script to do it for you.

### Example placeholder replacement

Before:
```bash
CLIENT_ID1="<CLIENT_ID1>"
CLIENT_ID2="<CLIENT_ID2>"
```

After:
```bash
CLIENT_ID1="172.31.44.13"
CLIENT_ID2="172.31.29.86"
```

## About `cd` and virtual environment activation

The scripts do **not** hardcode `cd /home/ubuntu/afrl` anymore.

Instead, they use this logic:

- default repository root = parent directory of the script location
- optional override with:
```bash
export REPO_ROOT="/your/repo/path"
```

This makes the scripts safer for GitHub users whose directory structure differs from the original private setup.

The scripts also do **not** force a virtual environment activation line.

They support an optional variable:
```bash
export VENV_ACTIVATE="/full/path/to/venv/bin/activate"
```

If you do not set it, the script assumes your environment is already active.

That is usually better for a public GitHub artifact because environment names differ across users and machines.

## Configuration Files

Required:
- `config/class_map.json`
- `config/scaler.json`

Optional but useful:
- `config/feature_cols.json`

## Data Preparation

The full merged dataset is not included in GitHub because of file size limits.

1. Place per-client CSV files in:
```text
data/raw/
```

2. Create the merged dataset:
```bash
python3 src/merge_telemetry.py --input_dir data/raw --output data/telemetry_merged.csv
```

This will create:
```text
data/telemetry_merged.csv
```

## Reproducing the Main Experimental Workflow

### 1. Activate your environment manually
```bash
source /path/to/venv/bin/activate
```

Or let the script do it:
```bash
export VENV_ACTIVATE="/path/to/venv/bin/activate"
```

### 2. Start client APIs
```bash
bash scripts/start_client.sh
```

### 3. Optional: verify client reachability
```bash
bash scripts/check_clients.sh
```

### 4. Main N=10 run
```bash
bash scripts/run_main.sh
```

### 5. Optional ablation run
```bash
bash scripts/run_ablation.sh
```

### 6. Scalability experiments
```bash
bash scripts/run_scalability.sh
```

### 7. Robustness experiments
```bash
bash scripts/run_robustness.sh
```

### 8. Plotting / export
Use your plotting command after results are generated.

## Why the scripts are split this way

The public artifact exposes the same logical runs as the execution workflow:
- one main run
- one optional ablation run
- scalability runs for `N=4`, `N=6`, `N=8`, and `N=10`
- robustness runs for severities `A` through `E`

For convenience, `run_scalability.sh` includes all four scalability commands and `run_robustness.sh` includes all five severity commands.

## Notes on nohup

The original experiments were launched with `nohup`, and the public scripts preserve that style. Each block writes to a separate output folder and log file.

## Exact experiment groupings reflected in the scripts

### Scalability
- `N=4` → clients `1,2,7,8`, `K=3`, `min_participants=2`
- `N=6` → clients `1,2,5,6,7,8`, `K=5`, `min_participants=3`
- `N=8` → clients `1..8`, `K=6`, `min_participants=4`
- `N=10` → clients `1..10`, `K=8`, `min_participants=5`

### Robustness
- Severity A → delay `90`, loss `1`, cpu workers `1`
- Severity B → delay `90`, loss `3`, cpu workers `2`
- Severity C → delay `180`, loss `5`, cpu workers `3`
- Severity D → delay `180`, loss `8`, cpu workers `4`
- Severity E → delay `240`, loss `12`, cpu workers `4`

## Minimal Pre-Run Checklist

Before running the scripts:
- update all `CLIENT_ID*` placeholders
- ensure `config/class_map.json` exists
- ensure `config/scaler.json` exists
- create `data/telemetry_merged.csv`
- confirm all client APIs are reachable on port `8000`
- confirm the output directory is writable

## Public Release Hygiene

Do not commit:
- private AWS IPs
- PEM keys
- virtual environments
- large result folders
- logs generated during local execution
- oversized merged datasets if GitHub rejects them

## License

Place the `LICENSE` file in the repository root. The MIT License is a simple common choice for code release.
