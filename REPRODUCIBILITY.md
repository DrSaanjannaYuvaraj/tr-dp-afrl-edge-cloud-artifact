# Reproducibility and evidence boundaries

## Frozen evidence

The repository holds **325 archived JSON results** in `results/frozen/` across 14 experiment groups. These are not outputs of a new run. Keep them unchanged, and verify them against `manifests/frozen_results_sha256.txt`.

```bash
sha256sum -c manifests/frozen_results_sha256.txt
```

On macOS, use `shasum -a 256 -c manifests/frozen_results_sha256.txt`. Run the check from the repository root. Also verify `manifests/publication_assets_sha256.txt` for publication assets and consult `src/client/SHA256_MANIFEST.json` for client-source provenance.

## Re-running analysis

The `analysis/` directory contains archived analysis and figure-generation scripts; `outputs/regenerated_analysis/` contains previously regenerated analysis tables. Inspect script arguments and input/output paths before execution. Some historical analysis scripts retain original AWS absolute paths and may require explicit local path adjustments. Do not imply that every historical script is plug-and-play.

Previously verified regenerated outputs include Core descriptive/inferential tables, E4 telemetry-ablation statistics, E6A summaries, and E7 per-run and five-seed summaries. They do not constitute fresh AWS training runs.

## Re-running distributed experiments

Follow [AWS deployment](docs/AWS_DEPLOYMENT_GUIDE.md), [data preparation](docs/DATA_PREPARATION.md), and [experiment execution](docs/EXPERIMENT_EXECUTION_GUIDE.md). These are operational instructions based on the inspected wrappers, not a guarantee of exact replication of all historical environmental conditions.

## Scientific caveats

Seeds 42–46 and the archived strategy/variant matrix were verified from JSON contents. Exact historical hardware, dataset availability, all reward-weight overrides, and E7 impairment implementations must be confirmed from primary experimental records or the manuscript before asserting numerical equivalence. Hardware/network nondeterminism can affect fresh runs.
