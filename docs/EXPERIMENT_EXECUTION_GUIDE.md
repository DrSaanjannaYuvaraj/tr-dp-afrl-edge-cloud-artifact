# Experiment execution: Core, Core Matched, E1–E7

Run all commands from the repository root on the **server** after following [AWS_DEPLOYMENT_GUIDE.md](AWS_DEPLOYMENT_GUIDE.md). First configure/start the matching client API on each participating client. Server wrappers dry-run by default; set `FGCS_CONFIRM_RUN=YES` only when deliberately launching an experiment. New outputs go to `outputs/manual_runs/`, **never** `results/frozen/`.

## Universal command pattern

```bash
# Safe command construction (creates a new manual-run folder; no training)
bash scripts/server/run_EXPERIMENT.sh STRATEGY

# Intentional live execution
FGCS_CONFIRM_RUN=YES bash scripts/server/run_EXPERIMENT.sh STRATEGY
```

Replace `run_EXPERIMENT.sh` with one of the actual filenames below. Set `SEED` in `config/experiments.env` before each run. The runner's shared argument set was checked against all ten Python `argparse` definitions. The wrapper blocks direct overrides of output paths, client URLs/IDs, and dataset/map/scaler paths through extra CLI arguments; edit local configuration instead. Other supported runner options may be passed through as additional arguments after the strategy.

## Verified archived matrix

All archived groups use seeds **42, 43, 44, 45, 46**. The table records what the JSONs establish, not every exact historical hyperparameter.

| Group | Server launcher | Strategy/condition and count |
|---|---|---|
| Core (30) | `run_core.sh` | `ddqn`, `dqn`, `fedavg`, `heuristic`, `tl_ppo`, `tr_dp_afrl`; five seeds each |
| Core Matched (15) | `run_core_matched.sh` | `fedavg`, `tl_ppo`, `tr_dp_afrl`; five seeds each |
| E1 original (30) | `run_e1.sh` | Same six strategies as Core; five seeds each |
| E1 final rerun (10) | `run_e1.sh` | `tl_ppo`, `tr_dp_afrl`; five seeds each |
| E2 original (30) and final rerun (30) | `run_e2.sh` | `execution_only`, `no_kl`, `no_offload_kl`, `no_selection_kl`, `selection_only`, `tr_dp_afrl`; five seeds each |
| E3 original (45) and final rerun (45) | `run_e3.sh` | `tr_dp_afrl`; reference plus low/high for accuracy, latency, offload, stability; five seeds each |
| E4 original (10) and final rerun (10) | `run_e4.sh` | `tr_dp_afrl` with/without telemetry; five seeds each |
| E5 final rerun (5) | `run_e5.sh` | `tr_dp_afrl`, N=10; five seeds |
| E6A physical (15) | `run_e6a.sh` | `tr_dp_afrl`, N=5, 8, 10; five seeds each |
| E6B controller (5) | `run_e6b.sh` | `controller`; five seeds, no live clients |
| E7 final (45) | `run_e7.sh` | `fedavg`, `tl_ppo`, `tr_dp_afrl` × `none`, `cpu`, `network`; five seeds |

## Launch examples

Set `EXPERIMENT` in every client's `scripts/client/client_runtime.env` to match the server experiment, then stop and restart the client wrapper for that experiment. For E5, `start_e5.sh` deliberately invokes the E1 client API. For E6B, **do not start clients**.

```bash
# Core
bash scripts/server/run_core.sh tr_dp_afrl
bash scripts/server/run_core.sh fedavg

# Core matched
bash scripts/server/run_core_matched.sh tl_ppo

# E1 comparison
bash scripts/server/run_e1.sh tr_dp_afrl

# E2 component ablation
bash scripts/server/run_e2.sh no_kl

# E3 reward sensitivity — must supply the matching weight overrides
bash scripts/server/run_e3.sh tr_dp_afrl

# E4 telemetry on/off — must configure the matching ablation flag
bash scripts/server/run_e4.sh tr_dp_afrl

# E5 ten-client study
bash scripts/server/run_e5.sh tr_dp_afrl

# E6A physical scalability — set NUM_CLIENTS to 5, 8, or 10
bash scripts/server/run_e6a.sh tr_dp_afrl

# E6B controller-only
bash scripts/server/run_e6b.sh controller

# E7 robustness — configure IMPAIR_CONDITION and impairment parameters
bash scripts/server/run_e7.sh tr_dp_afrl
```

These examples are **dry runs**. After checking the generated command and all configuration values, prefix the chosen command with `FGCS_CONFIRM_RUN=YES` to launch. Each wrapper defaults to `tr_dp_afrl`, except E6B defaults to `controller`; explicit strategies are clearer in a reproducibility guide.

## Experiment-specific settings that matter

**Core / Core Matched:** use the appropriate API (`live_client_api_core.py`) and the correct dataset, rounds and strategy. Archived Core Matched variant label: `CORE_MATCHED_300R_5SEED`.

**E1:** six original strategies versus the two-strategy final comparison. Client APIs use `--client_index`.

**E2:** run each of the six actual ablation strategy names; no separate wrapper per ablation is required.

**E3:** the nine archived variants are `reference`, `accuracy_low/high`, `latency_low/high`, `offload_low/high`, and `stability_low/high`. The runner exposes `--alpha_acc`, `--beta_lat`, `--gamma_off`, and `--lambda_stab`; the precise variant-specific numerical overrides were **not established** from the supplied inventory. Do not label nine identical-default runs as nine sensitivity conditions.

**E4:** archived variants are `E4_TELEMETRY_ABLATION_WITH_TELEMETRY` and `E4_TELEMETRY_ABLATION_NO_TELEMETRY`. The Python runner accepts `--ablate_telemetry`; verify the flag's Boolean parsing and intended semantics from the runner source before using it. Do not infer that a variant label alone changes the code path.

**E5:** N=10, five seeds; client wrapper uses `live_client_api_e1.py`. The exact experimental treatment beyond the archived N10 label requires source/manuscript confirmation.

**E6A:** set `NUM_CLIENTS` (or `FGCS_NUM_CLIENTS`) to 5, 8, or 10 and ensure those physical clients are live. Set `K` and `MIN_PARTICIPANTS` consistently with the intended experiment; they are not automatically inferred from N.

**E6B:** `bash scripts/server/run_e6b.sh controller` uses `E6B_SIZES`, `E6B_WARMUP`, `E6B_REPETITIONS`, and `SEED`/controller-specific configuration. The template defaults to `E6B_SIZES="10 20 50 100"`, whereas the archived paper configuration was previously reported as `8, 16, 40, 80`; **verify exact historical values before a reproduction claim**. No dataset or client URLs are loaded for E6B.

**E7:** exactly ten clients; set `IMPAIR_CONDITION` to `none`, `cpu`, or `network`, and configure the impairment interval and severity. The wrapper passes `--impair_condition`, `--impair_start_round`, `--impair_stop_round`, `--net_delay_ms`, `--net_loss_pct`. The mere presence of these arguments does not establish that real CPU/network impairment is occurring; verify implementation and client-side controls before claiming the robustness condition was reproduced.

## Seeds and variants

To execute five seeds, change `SEED` in `config/experiments.env` to each of `42`, `43`, `44`, `45`, `46`, and run the chosen command again. For E3, E4 and E7, change the relevant condition-specific settings between groups. The historical `variant_label` field is an archived metadata label, not a complete executable recipe.

## Verify results

The wrapper records a command file, log and PID in a unique `outputs/manual_runs/` directory. Confirm the process exited successfully and that the output JSONs contain the expected seed, strategy and round counts. Record actual instance types, operating system, package versions, dataset hashes, private topology (privately), and timestamps for the fresh run. Do not overwrite, reformat or regenerate the frozen JSONs in place.
