#!/bin/bash
set -e

echo "Starting agent server..."
uv run run_api &
SERVER_PID=$!

echo "Starting Streamlit client..."
uv run run_app &
CLIENT_PID=$!

cleanup() {
    echo "Shutting down..."
    kill $SERVER_PID $CLIENT_PID 2>/dev/null || true
    wait
}
trap cleanup SIGTERM SIGINT
wait