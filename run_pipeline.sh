#!/bin/bash
set -e

trap 'kill $(jobs -p)' EXIT

echo "[Init] Activating conda environment 'beau'..."
eval "$(conda shell.bash hook)"
conda activate beau

echo "[Init] Starting MySQL..."
sudo service mysql start || echo "MySQL service start skipped (may be running or no sudo)"

echo "[Init] Starting Ollama server..."
OLLAMA_HOST=127.0.0.1:11200 \
OLLAMA_MAX_LOADED_MODELS=2 \
OLLAMA_NOHISTORY=true \
OLLAMA_NUM_PARALLEL=1 \
OLLAMA_GPU_OVERHEAD=10240 \
ollama serve > ollama.log 2>&1 &

echo "[Init] Waiting for Ollama to be ready..."
while ! curl -s http://127.0.0.1:11200/api/tags > /dev/null; do
    sleep 1
done
echo "Ollama is ready!"

echo "[Init] Pulling models (Qwen 2.5 7B & Qwen Vision)..."
ollama pull qwen2.5:7b

ollama pull qwen2.5vl:7b

echo "[Init] Starting API (api_stream)..."
if [ ! -f "api_stream.py" ]; then
    echo "Looking for api_stream.py..."    
    if [ -d "beau-pipeline" ]; then
        cd beau-pipeline
    elif [ -d "../beau-pipeline" ]; then
        cd ../beau-pipeline
    else
        echo "WARNING: api_stream.py not found in the current directory. Adjust the 'cd' in the script."
    fi
fi

uvicorn api_stream:app --host 0.0.0.0 --port 8081 > api.log 2>&1 &

sleep 5

echo "[Init] Launching demo_chat (Frontend)..."
python demo_chat_apiv3.py
