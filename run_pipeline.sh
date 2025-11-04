#!/bin/bash

# Run Ollama in background
OLLAMA_HOST=127.0.0.1:11200 OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_NOHISTORY=true OLLAMA_NUM_PARALLEL=1 OLLAMA_GPU_OVERHEAD=10240 ollama serve &

# Run API in background
uvicorn api_stream:app --port 8081 &

# Run demo_chat (foreground)
python demo_chat.py
