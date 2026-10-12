#!/usr/bin/env bash
# Wait (up to WAIT_S seconds, default 180) until two GPUs are visible and each uses less than 2 GiB. Useful as a systemd ExecStartPre.
WAIT_S="${WAIT_S:-180}"
for _ in $(seq 1 $((WAIT_S / 3))); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null) || used=""
  n=$(printf '%s\n' "$used" | grep -c '[0-9]')
  if [ "$n" -ge 2 ] && [ "$(printf '%s\n' "$used" | sort -n | tail -1)" -lt 2048 ]; then exit 0; fi
  sleep 3
done
echo "two free GPUs not available after ${WAIT_S}s" >&2; exit 1
