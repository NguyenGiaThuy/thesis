#!/bin/bash

# ============================================================================================
# Declaration
# ============================================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
NOCOL='\033[0m'

# ============================================================================================
# PyTorch With Cuda
# ============================================================================================
echo ""
echo -e "${NOCOL}[1/5] Starting to install PyTorch-With-CUDA...${NOCOL}"

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
if ! [[ $? -eq 0 ]]; then 
  echo "Using pip (instead of pip3)..."
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
fi

if [[ $? -eq 0 ]]; then
    echo -e "${GREEN}[1/5] \xE2\x9C\x94 Successfully installed PyTorch-With-CUDA.${NOCOL}"
else
    echo -e "${RED}[1/5] \xE2\x9D\x8C Failed to install PyTorch-With-CUDA.${NOCOL}"
    exit 1
fi

# ============================================================================================
# Our Python Dependencies
# ============================================================================================
echo ""
echo -e "${NOCOL}[2/5] Starting to install other Python dependencies...${NOCOL}"

pip3 install -r requirements.txt
if ! [[ $? -eq 0 ]]; then 
  echo "Using pip (instead of pip3)..."
  pip install -r requirements.txt 
fi

if [[ $? -eq 0 ]]; then
    echo -e "${GREEN}[2/5] \xE2\x9C\x94 Successfully installed other Python dependencies.${NOCOL}"
else
    echo -e "${RED}[2/5] \xE2\x9D\x8C Failed to install other Python dependencies.${NOCOL}"
    exit 1
fi

# ============================================================================================
# Download dataset
# ============================================================================================



# ============================================================================================
# Download models
# ============================================================================================



# ============================================================================================
# Run web app
# ============================================================================================
echo ""
echo -e "${NOCOL}[5/5] Starting to run web app...${NOCOL}"

python3 ./web_app/app.py
if ! [[ $? -eq 0 ]]; then 
  echo "Using python (instead of python3)..."
  python ./web_app/app.py
fi

if [[ $? -eq 0 ]]; then
    echo -e "${GREEN}[5/5] \xE2\x9C\x94 Successfully installed other Python dependencies.${NOCOL}"
else
    echo -e "${RED}[5/5] \xE2\x9D\x8C Failed to install other Python dependencies.${NOCOL}"
    exit 1
fi
