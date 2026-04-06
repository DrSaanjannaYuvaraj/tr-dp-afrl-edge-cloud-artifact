# Artifact Description

This artifact contains the source code, configuration files, and client-side CSV inputs needed to reproduce the telemetry-driven AFRL workflow.

## Included

- Server-side experiment scripts
- Client-side API script
- Shared model and dataset utilities
- Configuration JSON files
- Per-client CSV files used to reconstruct the merged dataset locally

## Excluded

- Large merged dataset file (`data/telemetry_merged.csv`)
- Virtual environments
- Logs and generated outputs

## Reproduction summary

1. Install dependencies from `requirements.txt`.
2. Merge the client CSV files into `data/telemetry_merged.csv` using `src/merge_telemetry.py`.
3. Start one or more client APIs with `scripts/start_client.sh`.
4. Run server-side experiments with the provided scripts.
5. Generate figures and tables using `src/make_journal_figs_and_table.py`.
