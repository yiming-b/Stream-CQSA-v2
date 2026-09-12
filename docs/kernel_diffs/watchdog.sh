#!/bin/bash
# Kill any ptxas that has run longer than LIMIT minutes (default 90): a healthy
# hdim64/128 file finishes in < 20 min; anything longer is the mis-compiling
# instantiation (see LOG.md, v10/v11 first builds: 5+ h with no end).
#   nohup bash next/kernel/watchdog.sh > next/logs/watchdog.log 2>&1 &
LIMIT=${1:-90}
while true; do
  for pid in $(pgrep -x ptxas); do
    mins=$(ps -o etimes= -p "$pid" 2>/dev/null | awk '{print int($1/60)}')
    if [ -n "$mins" ] && [ "$mins" -ge "$LIMIT" ]; then
      echo "$(date) killing ptxas $pid after $mins min: $(tr '\0' ' ' < /proc/$pid/cmdline | grep -o 'flash_[a-z0-9_]*' | head -1)"
      kill -9 "$pid"
    fi
  done
  sleep 120
done
