# TR-DP-AFRL edge–cloud reproducibility artifact

This repository provides the implementation, preserved experimental outputs, and analysis assets for the FGCS TR-DP-AFRL study. The **325 frozen JSON results** are historical evidence; new executions must write elsewhere.

## Start here

- [AWS server and ten-client deployment](docs/AWS_DEPLOYMENT_GUIDE.md)
- [Dataset, client CSVs, and telemetry merging](docs/DATA_PREPARATION.md)
- [Core, Core Matched, and E1–E7 execution](docs/EXPERIMENT_EXECUTION_GUIDE.md)
- [Analysis and reproducibility](REPRODUCIBILITY.md)
- [Artifact inventory and checksums](ARTIFACT_MANIFEST.md)
- [Data availability and privacy](docs/data_availability.md)

## Two distinct reproducibility paths

**Analysis-only:** inspect `results/frozen/` and `analysis/`; no AWS machines are needed to examine archived outputs. The existing publication-asset and frozen-result manifests provide file-integrity checks.

**Fresh distributed execution:** provision a server and up to ten reachable client machines, supply the required dataset and preprocessing artifacts, configure local `.env` files, launch the appropriate client APIs, then invoke a server wrapper. This requires resources and inputs not bundled with the public artifact. Read the AWS guide first.

## Important limitations

The checked-in `.env.example` files are **illustrative**, not a reconstruction of every paper configuration. The frozen experiment matrix establishes strategies, variants, and seeds 42–46, but does not alone establish all hyperparameters, cloud instance specifications, or dataset download URL. Never claim byte-for-byte reproduction solely from the example values.

**Do not publish private IP addresses, cloud credentials, raw IP-bearing telemetry, or private `.env` files.** The original Git history may contain private infrastructure data: review and sanitize the release history before any public push. See [publication safety](docs/PUBLICATION_SAFETY.md).

## License

See the repository `LICENSE` file.
