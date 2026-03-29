#!/usr/bin/env bash
# Daily crypto analysis runner
# Scheduled via cron at 16:00 Europe/Paris

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
LOG_FILE="$LOG_DIR/analysis_$(date +%Y-%m-%d).log"

mkdir -p "$LOG_DIR"

echo "========================================" | tee -a "$LOG_FILE"
echo "  Run started: $(date '+%Y-%m-%d %H:%M %Z')" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

cd "$SCRIPT_DIR"
source .venv/bin/activate

python main.py --gemini --email 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "  Run finished: $(date '+%Y-%m-%d %H:%M %Z')" | tee -a "$LOG_FILE"

# Keep only the last 30 log files
find "$LOG_DIR" -name "analysis_*.log" | sort | head -n -30 | xargs -r rm --
