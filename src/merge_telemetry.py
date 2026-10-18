import glob
import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
pattern = os.path.join(DATA_DIR, "telemetry_client*.csv")
files = sorted(glob.glob(pattern))

if not files:
    raise SystemExit(f"No files found: {pattern}")

print(f"[INFO] Found {len(files)} files")
for f in files:
    print(" -", os.path.basename(f))

# Read and validate schema
dfs = []
base_cols = None

for fp in files:
    df = pd.read_csv(fp)
    cols = list(df.columns)

    if base_cols is None:
        base_cols = cols
    elif cols != base_cols:
        raise SystemExit(
            f"[ERROR] Column mismatch in {os.path.basename(fp)}\n"
            f"Expected: {base_cols}\nGot:      {cols}"
        )

    df["source_file"] = os.path.basename(fp)
    dfs.append(df)

merged = pd.concat(dfs, ignore_index=True)

# Parse timestamp and sort (recommended for analysis)
merged["timestamp_utc"] = pd.to_datetime(merged["timestamp_utc"], errors="coerce")
bad_ts = merged["timestamp_utc"].isna().sum()
if bad_ts:
    print(f"[WARN] {bad_ts} rows have invalid timestamp_utc")

merged = merged.sort_values(["timestamp_utc", "client_id"], kind="mergesort")

out_path = os.path.join(DATA_DIR, "telemetry_merged.csv")
merged.to_csv(out_path, index=False)

print(f"[DONE] Merged rows: {len(merged):,}")
print(f"[DONE] Saved: {out_path}")

# Optional summary
print("\n[SUMMARY] Rows per client_id:")
print(merged["client_id"].value_counts().sort_index())
print("\n[SUMMARY] Rows per scenario:")
print(merged["scenario"].value_counts())
