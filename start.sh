#!/bin/bash
cd /home/tate/organism
exec /home/tate/organism/.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
