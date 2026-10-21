#!/usr/bin/env bash
# run_training_safe.sh — VRAM handoff wrapper for the V620 training smoke test.
#
# Safety contract: this script ALWAYS restores the Hermes backend by calling
# `moe-backend resume` in a trap, on every exit path (success, crash, kill).
# Run it via a persisted background terminal so it also survives agent-session
# cleanup. If the agent is mid-conversation when the backend stops, the agent
# switches to its configured API fallback and reloads on `resumed`.
#
# Refuses to run if the backend controller reports a pending operation, and
# never kills GPU processes itself (controller-owned semantics).
set -euo pipefail

MB=/home/praxisog/.local/bin/moe-backend
CMD=("$@")
[ "${#CMD[@]}" -gt 0 ] || { echo "usage: $0 <training command...>"; exit 2; }

# Precondition: controller idle and healthy enough to stop from.
state=$("$MB" status)
echo "$state" | grep -q '"operation": null' || {
  echo "refusing: controller has a pending operation"; exit 3; }
echo "$state" | grep -q '"stop_pending": \[\]' || {
  echo "refusing: a stop is pending"; exit 3; }

restore() {
  code=$?
  echo "[safe-runner] exit=$code — resuming backend (idempotent if already up)"
  "$MB" resume --timeout 1800 || echo "[safe-runner] resume FAILED — run manually: $MB resume"
}
trap restore EXIT

echo "[safe-runner] stopping backend to free VRAM"
"$MB" stop
# Wait for VRAM release (controller stop is async at docker level).
for _ in $(seq 1 60); do
  used=$(rocm-smi --showmeminfo vram 2>/dev/null | grep -oE 'Used.*: [0-9]+' | grep -oE '[0-9]+$' | sort -rn | head -1)
  [ -n "$used" ] && [ "$used" -lt 2000000000 ] && break
  sleep 2
done
echo "[safe-runner] VRAM released (~$(( ${used:-0} / 2**30 )) GiB/card used); starting: ${CMD[*]}"
# NOT exec: exec would replace this shell and skip the EXIT trap, leaving
# the backend stopped forever. Child process + wait keeps the trap armed.
"${CMD[@]}"
rc=$?
echo "[safe-runner] training command exited rc=$rc"
exit "$rc"
