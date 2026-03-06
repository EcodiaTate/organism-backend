#!/usr/bin/env bash
# Kill all uvicorn processes (run from the backend directory or anywhere)
pids=$(pgrep -f uvicorn)
if [ -z "$pids" ]; then
    echo "No uvicorn processes found."
else
    echo "Killing uvicorn PIDs: $pids"
    kill $pids
    sleep 1
    # Force-kill any that didn't exit cleanly
    remaining=$(pgrep -f uvicorn)
    if [ -n "$remaining" ]; then
        echo "Force-killing: $remaining"
        kill -9 $remaining
    fi
    echo "Done."
fi
