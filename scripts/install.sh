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
    echo -e "${NOCOL}[1/5] Starting to install PyTorch-With-CUDA...${NOCOL}"

    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    if ! [[ $? -eq 0 ]]; then 
        echo "Using pip (instead of pip3)..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    fi

else
    echo -e "${NOCOL}[1/5] Starting to install PyTorch...${NOCOL}"
    pip3 install torch torchvision torchaudio
    if ! [[ $? -eq 0 ]]; then 
        echo "Using pip (instead of pip3)..."
        pip install torch torchvision torchaudio
    fi
fi

if [[ $? -eq 0 ]]; then
    echo -e "${GREEN}[1/5] \xE2\x9C\x94 Successfully installed PyTorch.${NOCOL}"
else
    echo -e "${RED}[1/5] \xE2\x9D\x8C Failed to install PyTorch.${NOCOL}"
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
# Download models
# ============================================================================================
echo ""
echo -e "${NOCOL}[3/5] Starting to download models...${NOCOL}"

if ! test -f results/resnet50/best_model_6.pth; then
    mkdir -p results
    curl --ssl-no-revoke https://thesis-files.thuy.binhql.com/resnet50.zip --output ./results/resnet50.zip \
        && unzip ./results/resnet50.zip -d ./results \
        && rm -rf ./results/resnet50.zip
    echo "Successfully downloaded models."
else
    echo "Model file: results/resnet50/best_model_6.pth has already existed."
fi

if [[ $? -eq 0 ]]; then
    echo -e "${GREEN}[3/5] \xE2\x9C\x94 Successfully got models.${NOCOL}"
else
    echo -e "${RED}[3/5] \xE2\x9D\x8C Failed to get models.${NOCOL}"
    exit 1
fi

# ============================================================================================
# Create web_app/static directory
# ============================================================================================
echo ""
echo -e "${NOCOL}[4/5] Starting to create web_app/static directory...${NOCOL}"

mkdir -p web_app/static

if [[ $? -eq 0 ]]; then
    echo -e "${GREEN}[4/5] \xE2\x9C\x94 Successfully created web_app/static directory.${NOCOL}"
else
    echo -e "${RED}[4/5] \xE2\x9D\x8C Failed to create web_app/static directory.${NOCOL}"
    exit 1
fi

# ============================================================================================
# Download demo dataset
# ============================================================================================
echo ""
echo -e "${NOCOL}[5/5] Starting to download demo dataset...${NOCOL}"

curl --ssl-no-revoke https://thesis-files.thuy.binhql.com/test-datasets.zip --output ./test-datasets.zip \
    && unzip ./test-datasets.zip \
    && rm -rf ./test-datasets.zip

if [[ $? -eq 0 ]]; then
    echo -e "${GREEN}[5/5] \xE2\x9C\x94 Successfully downloaded demo dataset.${NOCOL}"
else
    echo -e "${RED}[5/5] \xE2\x9D\x8C Failed to download demo dataset.${NOCOL}"
    exit 1
fi
