#!/bin/bash

# ============================================================================================
# Declaration
# ============================================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
NOCOL='\033[0m'

# ============================================================================================
# PyTorch (With Cuda Or Not)
# ============================================================================================
echo ""

if [[ $1 = "cuda" ]]; then
    echo -e "${NOCOL}[1/3] Starting to install PyTorch-With-CUDA...${NOCOL}"

    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    if ! [[ $? -eq 0 ]]; then 
        echo "Using pip (instead of pip3)..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    fi

else
    echo -e "${NOCOL}[1/3] Starting to install PyTorch...${NOCOL}"
    pip3 install torch torchvision torchaudio
    if ! [[ $? -eq 0 ]]; then 
        echo "Using pip (instead of pip3)..."
        pip install torch torchvision torchaudio
    fi
fi

if [[ $? -eq 0 ]]; then
    echo -e "${GREEN}[1/3] \xE2\x9C\x94 Successfully installed PyTorch.${NOCOL}"
else
    echo -e "${RED}[1/3] \xE2\x9D\x8C Failed to install PyTorch.${NOCOL}"
    exit 1
fi

# ============================================================================================
# Our Python Dependencies
# ============================================================================================
echo ""
echo -e "${NOCOL}[2/3] Starting to install other Python dependencies...${NOCOL}"

pip3 install -r requirements.txt
if ! [[ $? -eq 0 ]]; then 
  echo "Using pip (instead of pip3)..."
  pip install -r requirements.txt 
fi

if [[ $? -eq 0 ]]; then
    echo -e "${GREEN}[2/3] \xE2\x9C\x94 Successfully installed other Python dependencies.${NOCOL}"
else
    echo -e "${RED}[2/3] \xE2\x9D\x8C Failed to install other Python dependencies.${NOCOL}"
    exit 1
fi

# ============================================================================================
# Download models
# ============================================================================================
echo ""
echo -e "${NOCOL}[3/3] Starting to download models...${NOCOL}"

if ! test -f results/resnet50/best_model_6.pth; then
    mkdir -p results
    curl --ssl-no-revoke https://thuy-thesis.s3.ap-southeast-1.amazonaws.com/resnet50.zip --output ./results/resnet50.zip && unzip ./results/resnet50.zip -d ./results
    echo "Successfully downloaded models."
else
    echo "Model file: results/resnet50/best_model_6.pth has already existed."
fi

if [[ $? -eq 0 ]]; then
    echo -e "${GREEN}[3/3] \xE2\x9C\x94 Successfully got models.${NOCOL}"
else
    echo -e "${RED}[3/3] \xE2\x9D\x8C Failed to get models.${NOCOL}"
    exit 1
fi
