#!/usr/bin/env bash

sudo apt-get update
sudo apt-get install -y git ffmpeg libsndfile1 espeak-ng wget git tar curl build-essential \
    patchelf ninja-build cmake libssl-dev; # Maybe even libsndfile1-dev sox

# Based on https://github.com/dscripka/openWakeWord/blob/main/notebooks/automatic_model_training.ipynb

cd ~/workspace/oww

# Install UV if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Install Python 3.11 if not already installed
uv python install 3.11

# create virtual environment
if [ ! -d ".openwakeword_venv" ] || [ ! -f ".openwakeword_venv/bin/activate" ]; then
    echo "(Re)-Creating virtual environment."
    rm -rf .openwakeword_venv
    uv venv --python 3.11 .openwakeword_venv
fi

# Activate virtual environment and install dependencies
source .openwakeword_venv/bin/activate
python3 -m ensurepip --upgrade
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

if [ -d "piper-sample-generator" ]; then
    echo "Updating piper-sample-generator."
    cd piper-sample-generator
    git pull
else
    echo "Cloning piper-sample-generator."
    git clone https://github.com/rhasspy/piper-sample-generator piper-sample-generator
fi

cd ~/workspace/oww

PIPER_MODEL_PATH=piper-sample-generator/models/en_US-libritts_r-medium.pt
if [ ! -f "$PIPER_MODEL_PATH" ] || ! sha256sum -c "$PIPER_MODEL_PATH.sha256" 2>/dev/null; then
    echo "Downloading piper-sample-generator model."
    wget -O piper-sample-generator/models/en_US-libritts_r-medium.pt 'https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt'
    sha256sum piper-sample-generator/models/en_US-libritts_r-medium.pt > piper-sample-generator/models/en_US-libritts_r-medium.pt.sha256
else
    echo "Piper-sample-generator model already exists and is valid."
fi

# install openwakeword (full installation to support training)
if [ -d "openwakeword" ]; then
    echo "Updating openwakeword."
    cd openwakeword
    git pull
else
    echo "Cloning openwakeword."
    git clone https://github.com/dscripka/openwakeword openwakeword
fi

cd ~/workspace/oww

pip install -e ./openwakeword

# Download required models (workaround for Colab)
mkdir -p ./openwakeword/openwakeword/resources/models
wget https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx -O ./openwakeword/openwakeword/resources/models/embedding_model.onnx
wget https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.tflite -O ./openwakeword/openwakeword/resources/models/embedding_model.tflite
wget https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx -O ./openwakeword/openwakeword/resources/models/melspectrogram.onnx
wget https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.tflite -O ./openwakeword/openwakeword/resources/models/melspectrogram.tflite