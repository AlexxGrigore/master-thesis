#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$SCRIPT_DIR/overnight.log"

SMOKE=""
if [[ "${1:-}" == "--smoke-test" ]]; then
    SMOKE="--smoke-test"
fi

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "=== Overnight run started ${SMOKE} ==="

log "--- Run 1: synthetic dataset ---"
python "$SCRIPT_DIR/main.py" --dataset-type synthetic $SMOKE 2>&1 | tee -a "$LOG"
log "--- Run 1 finished ---"

log "--- Run 2: real dataset ---"
python "$SCRIPT_DIR/main.py" --dataset-type real $SMOKE 2>&1 | tee -a "$LOG"
log "--- Run 2 finished ---"

log "=== All done ==="
