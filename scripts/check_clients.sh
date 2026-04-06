#!/usr/bin/env bash
set -euo pipefail

echo "Update the CLIENT*_IP placeholders before running this reachability check."

echo "== N=4 =="
for ip in CLIENT1_IP CLIENT2_IP CLIENT3_IP CLIENT4_IP; do
  echo "== $ip =="
  curl -s --max-time 3 "http://$ip:8000/health" | head -c 200 || true
  echo
done

echo "== N=6 =="
for ip in CLIENT1_IP CLIENT2_IP CLIENT5_IP CLIENT6_IP CLIENT7_IP CLIENT8_IP; do
  echo "== $ip =="
  curl -s --max-time 3 "http://$ip:8000/health" | head -c 200 || true
  echo
done

echo "== N=8 =="
for ip in CLIENT1_IP CLIENT2_IP CLIENT3_IP CLIENT4_IP CLIENT5_IP CLIENT6_IP CLIENT7_IP CLIENT8_IP; do
  echo "== $ip =="
  curl -s --max-time 3 "http://$ip:8000/health" | head -c 200 || true
  echo
done

echo "== N=10 =="
for ip in CLIENT1_IP CLIENT2_IP CLIENT3_IP CLIENT4_IP CLIENT5_IP CLIENT6_IP CLIENT7_IP CLIENT8_IP CLIENT9_IP CLIENT10_IP; do
  echo "== $ip =="
  curl -s --max-time 3 "http://$ip:8000/health" | head -c 200 || true
  echo
done
