# Data preparation, ten-client distribution, and telemetry merging

## Understand the three distinct data roles

1. **Raw telemetry CSVs:** `telemetry_client1.csv` … `telemetry_client10.csv`, collected or provided separately; potentially contain private addresses and infrastructure details.
2. **Merged telemetry CSV:** a chronological aggregation produced by the repository merger, not necessarily a model-ready feature matrix.
3. **Prepared experiment dataset and per-client CSVs:** the server's `DATASET_CSV` and each client's `CLIENT_CSV`, with class-map/scaler artifacts. The repository wrappers require these paths but do **not** generate the CSVs or perform preprocessing/distribution automatically.

The exact source URL and full transformation from raw telemetry to the published model-ready dataset were not verified from the inspected materials. Do not invent a Zenodo DOI or claim that merging alone recreates the published training CSV. Check the manuscript, original data release, and `src/server/telemetry_dataset.py` / `src/client/telemetry_dataset.py` before substituting a dataset.

## Collect or obtain ten raw telemetry files

The repository includes `src/client/telemetry_client.py` and `src/client/run_scenarios.sh`, but their full runtime protocol and scenario parameters were not verified here. Inspect these scripts and obtain necessary permissions before using them to generate new data. Alternatively, use an authorized copy of the original ten CSVs. Do not commit private raw telemetry.

## Transfer the CSVs to the server

For the **primary** merger, create `src/data/` on the server, then securely transfer the ten files so the server has:

```text
src/data/telemetry_client1.csv
src/data/telemetry_client2.csv
...
src/data/telemetry_client10.csv
```

For example, from an authorized administrative machine, using placeholder names and a private SSH key:

```bash
scp -i /path/to/authorized-key.pem /path/to/telemetry_client1.csv ubuntu@SERVER_HOST:/home/ubuntu/tr-dp-afrl-edge-cloud-artifact/src/data/
```

Repeat for clients 2–10, or use an approved private S3 bucket with restricted access. `SERVER_HOST` is a placeholder, not a real address. For private-only EC2 machines, use an SSH bastion/SSM workflow instead of exposing SSH publicly. Verify filenames and schema after transfer.

## Validate and merge on the server

The inspected script `src/merge_telemetry.py` looks for `src/data/telemetry_client*.csv` and writes `src/data/telemetry_merged.csv`:

```bash
python3 src/merge_telemetry.py
```

It reads CSVs using pandas, requires **identical column names in identical order**, appends a `source_file` column, parses `timestamp_utc` (invalid timestamps become missing values and trigger a warning), sorts by `timestamp_utc` and `client_id`, then prints row counts per `client_id` and `scenario`. The script does **not** enforce exactly ten files: ensure the input directory contains only the intended `telemetry_client1.csv` … `telemetry_client10.csv` files. An extra `telemetry_client11.csv` would also match its glob. Check invalid-timestamp warnings and expected per-client counts before using the merged output.

A second copy, `src/telemetry/merge_telemetry.py`, has the same logic but reads and writes under `src/telemetry/data/`. **Do not mix the two locations.** The recommended documented workflow uses `src/merge_telemetry.py`.

## Prepare and distribute model-input files

The server requires `DATASET_CSV`, `CLASS_MAP_PATH`, `SCALER_PATH` in `config/experiments.env`. Each client requires its own `CLIENT_CSV`, `CLASS_MAP_PATH`, `SCALER_PATH` in `scripts/client/client_runtime.env`. Use a documented, consistent preprocessing/partitioning procedure from the original study; copy the resulting client-specific CSV and the matching class-map/scaler artifacts to each EC2 client via a private channel. The existence of `src/client/class_map.json`, `src/client/scaler.json`, and `src/client/feature_cols.json` does not prove that these match every experiment/dataset; verify before reuse.

Client indexing for E1–E7 except E6B is zero-based (`client1` → 0, `client10` → 9). Core and Core Matched do not supply `--client_index`. Keep client IDs and CSV assignments aligned.

## Privacy

Raw telemetry can contain private IPv4 addresses, hostnames, timestamps, or operational metadata. Keep it outside Git and public archives unless appropriately de-identified and cleared for release. Do not expose AWS credentials, SSH keys, or live endpoint addresses in README examples or archived JSONs.
