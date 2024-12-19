#!/bin/bash

# Redis 서버 시작 (백그라운드에서 실행)
nohup ./redis-stable/src/redis-server > redis-server.log 2>&1 &

# Celery 워커 시작 (백그라운드에서 실행, 로그 파일 저장)
cd app
nohup celery -A tasks worker --pool=solo --loglevel=info > celery-worker.log 2>&1 &

# FastAPI 서버 (Uvicorn) 시작 (백그라운드에서 실행, 로그 파일 저장)
nohup uvicorn main:app --host 0.0.0.0 --port 8000 > uvicorn-server.log 2>&1 &