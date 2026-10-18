# Artifact inventory

## Frozen results — 325 JSON files

| Directory under `results/frozen/` | Count | Archived design |
|---|---:|---|
| `core` | 30 | Six strategies × five seeds |
| `core_matched` | 15 | Three strategies × five seeds |
| `e1_original` | 30 | Six strategies × five seeds |
| `e1_final_rerun` | 10 | Two strategies × five seeds |
| `e2_original` | 30 | Six ablations × five seeds |
| `e2_final_rerun` | 30 | Six ablations × five seeds |
| `e3_original` | 45 | Nine reward variants × five seeds |
| `e3_final_rerun` | 45 | Nine reward variants × five seeds |
| `e4_original` | 10 | Telemetry on/off × five seeds |
| `e4_final_rerun` | 10 | Telemetry on/off × five seeds |
| `e5_final_rerun` | 5 | N=10 × five seeds |
| `e6a_physical` | 15 | N=5, 8, 10 × five seeds |
| `e6b_controller` | 5 | Controller-only × five seeds |
| `e7_final` | 45 | Three strategies × three conditions × five seeds |
| **Total** | **325** | |

Seeds: **42, 43, 44, 45, 46**.

## Key paths

- `src/server/`: Python server experiment implementations.
- `src/client/`: Python client APIs, preprocessing assets, telemetry generation utilities.
- `scripts/server/`: experiment wrappers and shared runner.
- `scripts/client/`: client start, status and stop wrappers.
- `scripts/lib/`: experiment and client configuration validation.
- `config/*.env.example`: safe, illustrative templates.
- `analysis/`: analysis and plotting sources.
- `outputs/regenerated_analysis/`: previously regenerated tables and analysis outputs.
- `manifests/`: SHA-256 integrity inventories.

Some historical assets are archived rather than fully portable; check the documentation before running them.
