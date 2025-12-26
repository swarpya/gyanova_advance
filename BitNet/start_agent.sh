#!/bin/bash

# --- 1. CONFIGURATION ---
export PROJECT_ROOT="/home/user/gyanova/BitNet"
export AIRFLOW_HOME="$PROJECT_ROOT/airflow"
export PYTHONPATH="$PROJECT_ROOT/.venv/lib/python3.11/site-packages:$PROJECT_ROOT"

# Security Settings
export AIRFLOW__CORE__LOAD_EXAMPLES=False
export AIRFLOW__WEBSERVER__ENABLE_PROXY_FIX=True
export AIRFLOW__WEBSERVER__BASE_URL="https://8080-firebase-gyanova-1766439470373.cluster-lrhkgnsygfb7ovnbvdt2fflxme.cloudworkstations.dev"

# --- 2. ACTIVATION ---
cd "$PROJECT_ROOT"  # Ensure we are in the root so relative paths (like 'models/') work
source "$PROJECT_ROOT/.venv/bin/activate"

# --- 3. ENSURE PASSWORD IS SET ---
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password V9fDuWvveYaqVxka > /dev/null 2>&1

# --- 4. INFO DISPLAY ---
LOGIN_URL="${AIRFLOW__WEBSERVER__BASE_URL}/api/v2/auth/login"

echo "=========================================================="
echo "🚀 Starting BitNet Agent..."
echo "📂 Project Root: $PROJECT_ROOT"
echo "----------------------------------------------------------"
echo "👉 CLICK TO LOGIN:  $LOGIN_URL"
echo "----------------------------------------------------------"
echo "🔑 User: admin"
echo "🔑 Pass: V9fDuWvveYaqVxka"
echo "=========================================================="

# --- 4.5 START AI MODEL SERVER (Background) ---
# Check if agent_server.py exists before trying to run it
if [ -f "agent_server.py" ]; then
    echo "🧠 Booting AI Model Server (Port 5000)..."
    python agent_server.py > agent_server.log 2>&1 &
    SERVER_PID=$!
    echo "   [PID: $SERVER_PID] Server is loading in background."
    echo "   (Logs are being saved to $PROJECT_ROOT/agent_server.log)"
else
    echo "⚠️  WARNING: agent_server.py not found in $PROJECT_ROOT"
    echo "   The Fast Agent DAG will fail properly without this."
fi

echo "=========================================================="

# --- 5. START AIRFLOW ---
# Trap commands to kill the background server when you stop Airflow (Ctrl+C)
trap "kill $SERVER_PID" EXIT

airflow standalone