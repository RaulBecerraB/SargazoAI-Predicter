#!/usr/bin/env bash
MODULE="sargazo_predictor_service.app.main:app"
HOST="0.0.0.0"
PORT=8000
RELOAD=${1:-false}

if [ "$RELOAD" = "true" ]; then
  RELOAD_FLAG="--reload"
else
  RELOAD_FLAG=""
fi

echo "Starting uvicorn $MODULE on $HOST:$PORT (reload=$RELOAD)"
exec uvicorn $MODULE --host $HOST --port $PORT $RELOAD_FLAG
