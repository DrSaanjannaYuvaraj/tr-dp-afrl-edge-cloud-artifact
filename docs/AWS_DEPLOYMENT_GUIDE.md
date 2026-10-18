# AWS deployment: one server and up to ten client instances

This is a **fresh-run operational guide**, not a claim that the example environment reproduces every published hardware or hyperparameter setting. It follows the inspected repository wrappers. The operator supplies AWS resources, permitted data, and network details.

## 1. Architecture and prerequisites

Use one Linux EC2 server to coordinate experiments and up to ten Linux EC2 client machines (`client1` … `client10`). A smaller subset is applicable to E6A; **E7 requires exactly ten configured clients**. E6B is **controller-only**, with no live clients or dataset requirement in the wrapper.

Choose an AWS region, Linux AMI, instance types, storage, security groups, and costs appropriate to the workload; the original instance types and AMI were **not verified**. Use private VPC addressing and security-group-to-security-group access where possible. Avoid opening port 8000 to the public internet. Permit SSH only from trusted administrator addresses or use AWS Systems Manager Session Manager. Configure outbound access as needed for packages and datasets. Never publish credentials or machine IPs.

Recommended connectivity: server can reach each client on TCP port `8000`; administrators can securely access the instances. The client API binds to `0.0.0.0:8000` by default, so network filtering is essential. Prefer dedicated IAM roles and temporary credentials; no AWS key is needed inside the repository.

## 2. Install system and Python prerequisites on each EC2 instance

Example for Ubuntu/Debian-based Linux (adjust for your AMI):

```bash
sudo apt-get update
sudo apt-get install -y git python3 python3-venv python3-pip
```

Obtain the **sanitized public artifact release** on each machine, for example:

```bash
git clone https://github.com/DrSaanjannaYuvaraj/tr-dp-afrl-edge-cloud-artifact.git
cd tr-dp-afrl-edge-cloud-artifact
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**Publication warning:** Do not publicly push or recommend cloning an unsanitized Git history. The URL above identifies the intended repository; confirm that a safe release exists before sharing it with reviewers. GPU-enabled PyTorch installations may require platform-specific wheels; no CUDA version was verified.

On each client, retain the full repository or at least `src/client/` plus `scripts/client/` and dependencies. The wrapper uses the repository-relative client source by default. If deploying only a client source bundle, set `FGCS_CLIENT_SOURCE_ROOT` to the actual client Python source directory.

## 3. Prepare data and preprocessing artifacts

Follow [DATA_PREPARATION.md](DATA_PREPARATION.md). The server runner requires `DATASET_CSV`, `CLASS_MAP_PATH`, and `SCALER_PATH` pointing to **existing files**. Each client requires a readable `CLIENT_CSV` plus the class map and scaler. The client CSV is not automatically generated or transferred by the startup wrapper. Do not substitute raw merged telemetry for a prepared model-input CSV without confirming the preprocessing schema.

On each client, choose absolute paths for its own input CSV and preprocessing JSONs. An example directory layout is:

```text
/home/ubuntu/tr-dp-afrl-edge-cloud-artifact/
  src/client/
  scripts/client/
  private_data/client1.csv       # operator-supplied; not committed
  private_data/class_map.json    # operator-supplied or verified repo asset
  private_data/scaler.json       # operator-supplied or verified repo asset
```

## 4. Configure and start each client

On **client1**, from the repository root:

```bash
cp scripts/client/client_runtime.env.example scripts/client/client_runtime.env
```

Edit `scripts/client/client_runtime.env` to set real absolute paths:

```bash
CLIENT_ID=client1
EXPERIMENT=core
CLIENT_CSV="/absolute/path/to/client1.csv"
CLASS_MAP_PATH="/absolute/path/to/class_map.json"
SCALER_PATH="/absolute/path/to/scaler.json"
CLIENT_HOST=0.0.0.0
CLIENT_PORT=8000
CLIENT_WIN=10
PYTHON_BIN=python3
```

For client2 … client10, repeat on the respective EC2 machine, changing `CLIENT_ID` and `CLIENT_CSV`. Keep `CLIENT_PORT=8000` unless both sides are intentionally reconfigured. Set `EXPERIMENT` to the experiment being launched (`core`, `core_matched`, `e1`, `e2`, `e3`, `e4`, `e5`, `e6a`, or `e7`). **E6B has no client launcher.**

Run a **dry run** on each client first:

```bash
bash scripts/client/start_core.sh
```

For a deliberate launch:

```bash
FGCS_CONFIRM_RUN=YES bash scripts/client/start_core.sh
```

Use `start_core_matched.sh`, `start_e1.sh` … `start_e7.sh` for other experiments, except E6B. The wrapper refuses an incompatible `EXPERIMENT` value; update `client_runtime.env` and stop the previous API before switching experiments. **E5 deliberately runs `live_client_api_e1.py`** via its wrapper. Core/Core Matched use `live_client_api_core.py`. E1–E4, E6A, and E7 use their matching client API modules.

Check client processes and logs using:

```bash
bash scripts/client/status_client.sh
```

Stop before reconfiguration:

```bash
bash scripts/client/stop_client.sh
```

The main start script records PID/logs under `scripts/client/runtime/`. Inspect logs if a process fails; do not delete PID files blindly. Verify the API is reachable from the server over the private VPC network before beginning a full run. Endpoint paths are not established by the wrapper inventory; inspect the API source before relying on a particular health-check route.

## 5. Configure the coordinating server

From the server repository root:

```bash
cp config/experiments.env.example config/experiments.env
cp config/clients.env.example config/clients.env
```

Edit `config/clients.env`: replace `CLIENT1_IP` … `CLIENT10_IP` with each client's **actual private, reachable address**; keep `CLIENT_PORT=8000` and set `NUM_CLIENTS` to the intended physical-client count. The documentation-only `192.0.2.x` addresses in the template are **not real targets** and are rejected by the validation script. For E7 set `NUM_CLIENTS=10`. For E6A set 5, 8, or 10 to match the run; ensure corresponding clients are started. The server builds `CLIENT_URLS` and `CLIENT_IDS` in client-number order.

Edit `config/experiments.env`: set absolute paths for `DATASET_CSV`, `CLASS_MAP_PATH`, and `SCALER_PATH`, and set `ROUNDS`, `SEED`, `K`, `MIN_PARTICIPANTS`, `WIN`, local training and reward parameters as appropriate. `MIN_PARTICIPANTS` cannot exceed `K`. The example values (`ROUNDS=300`, `K=5`, etc.) are **not independently verified paper settings**. Reconcile each run with archived metadata and the manuscript before claiming published-run replication. Set `PYTHON_BIN` to the environment's interpreter if required.

Keep `config/clients.env`, `config/experiments.env`, and `scripts/client/client_runtime.env` **local and uncommitted**.

## 6. Dry run and deliberately execute

For Core with the default strategy:

```bash
bash scripts/server/run_core.sh tr_dp_afrl
```

This constructs the command and creates a manual-run directory containing `command.sh`, but **does not launch training** without explicit confirmation. Inspect the generated command and confirm paths and arguments. To start intentionally:

```bash
FGCS_CONFIRM_RUN=YES bash scripts/server/run_core.sh tr_dp_afrl
```

The launcher writes under `outputs/manual_runs/<experiment>_<strategy>_<unique-suffix>/` and records `command.sh`, `nohup.log`, and `pid`. Do **not** direct outputs to `results/frozen/`. See the [execution guide](EXPERIMENT_EXECUTION_GUIDE.md) for all experiments and seed/variant matrices.

## 7. Collect outputs and shut down

Check the generated run directory, `nohup.log`, process PID, and experiment-produced JSON/tables. Preserve run-specific configuration and timing/environment notes for reproducibility. Stop client processes using the supplied `stop_client.sh` and shut down unused EC2 instances to avoid charges. Use secure copy or a controlled artifact store to retrieve new outputs. Never publish logs or CSVs containing private IPs without inspection and redaction.

## Troubleshooting

- **Missing client:** confirm VPC route, security group, private address, TCP 8000, API process, and client-specific log.
- **Rejected IP:** replace template documentation IPs with actual private instance addresses.
- **Missing dataset/map/scaler:** configure absolute paths and verify files exist on the machine executing that component.
- **Wrong experiment API:** stop the previous client API, change `EXPERIMENT`, then launch the matching wrapper.
- **Nothing started:** expected unless `FGCS_CONFIRM_RUN=YES` is set.
- **E7 rejected:** configure exactly ten clients.
- **E6B:** launch controller only; do not start client APIs or expect a dataset CSV.
