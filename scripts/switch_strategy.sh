#!/usr/bin/env bash
# switch_strategy.sh — hot-swap the active trading strategy via ConfigMap patch
#
# Usage: ./scripts/switch_strategy.sh <strategy_name> <params_json_file>
# Example: ./scripts/switch_strategy.sh zscore_reversion params/zscore.json
set -euo pipefail

NAMESPACE="trading"
CONFIGMAP="trading-config"
DEPLOYMENT="signal-engine"

# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------
if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <strategy_name> <params_json_file>" >&2
  exit 1
fi

STRATEGY_NAME="$1"
PARAMS_FILE="$2"

if [[ ! -f "$PARAMS_FILE" ]]; then
  echo "ERROR: params file not found: $PARAMS_FILE" >&2
  exit 1
fi

STRATEGY_PARAMS=$(cat "$PARAMS_FILE")

# Validate JSON
if ! echo "$STRATEGY_PARAMS" | python3 -c "import sys, json; json.load(sys.stdin)" 2>/dev/null; then
  echo "ERROR: $PARAMS_FILE is not valid JSON" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# 1. Check open positions
# ---------------------------------------------------------------------------
echo "==> Checking open positions..."

OPEN_POSITIONS=$(kubectl exec -n "$NAMESPACE" deployment/"$DEPLOYMENT" -- \
  python3 -c "
import sqlite3, sys
try:
    conn = sqlite3.connect('/data/trading.db')
    count = conn.execute(\"SELECT COUNT(*) FROM positions WHERE status='open'\").fetchone()[0]
    print(count)
except Exception as e:
    print(0)
" 2>/dev/null || echo "0")

if [[ "$OPEN_POSITIONS" -gt 0 ]]; then
  echo ""
  echo "WARNING: There are $OPEN_POSITIONS open position(s)."
  echo "Switching strategy while positions are open may cause inconsistent risk management."
  read -rp "Continue anyway? [y/N] " CONFIRM
  if [[ "${CONFIRM,,}" != "y" ]]; then
    echo "Aborted."
    exit 0
  fi
fi

# ---------------------------------------------------------------------------
# 2. Patch the ConfigMap
# ---------------------------------------------------------------------------
echo ""
echo "==> Patching ConfigMap '$CONFIGMAP' in namespace '$NAMESPACE'..."

kubectl patch configmap "$CONFIGMAP" \
  -n "$NAMESPACE" \
  --type merge \
  -p "{\"data\":{\"ACTIVE_STRATEGY\":\"$STRATEGY_NAME\",\"STRATEGY_PARAMS\":$(echo "$STRATEGY_PARAMS" | python3 -c "import sys,json; print(json.dumps(sys.stdin.read()))")}}"

echo "    ACTIVE_STRATEGY = $STRATEGY_NAME"
echo "    STRATEGY_PARAMS = $STRATEGY_PARAMS"

# ---------------------------------------------------------------------------
# 3. Restart the deployment to pick up the new ConfigMap values
# ---------------------------------------------------------------------------
echo ""
echo "==> Restarting deployment/$DEPLOYMENT..."
kubectl rollout restart deployment/"$DEPLOYMENT" -n "$NAMESPACE"

# ---------------------------------------------------------------------------
# 4. Wait for rollout to complete
# ---------------------------------------------------------------------------
echo ""
echo "==> Waiting for rollout to complete..."
kubectl rollout status deployment/"$DEPLOYMENT" -n "$NAMESPACE" --timeout=120s

# ---------------------------------------------------------------------------
# 5. Done
# ---------------------------------------------------------------------------
echo ""
echo "✓ Strategy switched successfully."
echo "  Active strategy : $STRATEGY_NAME"
echo "  Namespace       : $NAMESPACE"
echo "  Deployment      : $DEPLOYMENT"
