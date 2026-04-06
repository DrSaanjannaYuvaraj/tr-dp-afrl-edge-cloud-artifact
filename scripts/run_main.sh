#!/usr/bin/env bash
set -euo pipefail

nohup python3 src/afrl_idea1_experiment_all.py   --csv data/telemetry_merged.csv   --class_map config/class_map.json   --scaler_path config/scaler.json   --clients "http://CLIENT1_IP:8000,http://CLIENT2_IP:8000"   --rounds 300   --seeds 42,43,44,45,46   --strategies fedavg,heuristic,dqn,ddqn,tl_ppo,tr_dp_afrl   --K 8   --min_participants 5   --outdir outputs/N10_300r_main   --variant_label N10
