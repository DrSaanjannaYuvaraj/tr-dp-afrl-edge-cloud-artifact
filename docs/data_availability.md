# Data availability and privacy

The repository preserves 325 frozen experiment-result JSONs and supporting analysis assets. The ten original raw telemetry CSVs are **not included in the prepared public working tree** because they contain private infrastructure-address information. The repository does not guarantee availability of a ready-to-use public training dataset or all per-client CSV partitions.

A dataset reference may be described in older configuration comments, but its definitive public URL, license and preprocessing steps were **not verified** for this documentation package. Do not invent a DOI or claim a specific download is available without checking the primary data-release record. Operators must obtain data they are authorized to use, prepare compatible server/client inputs, and keep sensitive files private.

`src/merge_telemetry.py` merges ten correctly named raw telemetry CSVs placed in `src/data/`; it does not by itself generate the model-ready dataset. See [DATA_PREPARATION.md](DATA_PREPARATION.md).
