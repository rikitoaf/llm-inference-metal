# Sourced by the GPU measurement scripts. Microsoft Defender's real-time scan (wdavdaemon_unprivileged at 150-200%
# CPU) takes 20-45% off the GPU while it runs, and our own file writes (builds, flattened kernel copies, the
# persistent cache) start it. So: build first, then wait for the machine to go quiet, and record Defender's CPU
# next to every measurement.

# Defender's current CPU percent (0 when it is not running).
defender_cpu() {
  ps -axo %cpu=,comm= | awk '/wdavdaemon_unprivileged/ {print int($1); found=1} END {if (!found) print 0}' | head -1
}

# Waits until Defender stays at or below $1 % CPU (default 30) for $2 consecutive seconds (default 30), at most
# $3 seconds in all (default 1200). Returns 1 on timeout.
wait_quiet() {
  local limit=${1:-30} hold=${2:-30} timeout=${3:-1200} quiet=0 waited=0
  while (( quiet < hold )); do
    if (( $(defender_cpu) <= limit )); then quiet=$((quiet + 5)); else quiet=0; fi
    (( waited >= timeout )) && { echo "wait_quiet: Defender still busy after ${timeout}s" >&2; return 1; }
    sleep 5; waited=$((waited + 5))
  done
  echo "wait_quiet: quiet after ${waited}s" >&2
}

# Fails unless on AC power, with no model server or suite running.
require_measurable() {
  pmset -g batt | grep -q "AC Power" || { echo "error: on battery (Low Power mode throttles the GPU)" >&2; return 1; }
  if pgrep -f "serve-native|run_kv_suite" >/dev/null; then echo "error: a model server or suite is running" >&2; return 1; fi
}
