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
    echo -e "${NOCOL}[1/4] Starting to install PyTorch-With-CUDA...${NOCOL}"

    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    if ! [[ $? -eq 0 ]]; then 
        echo "Using pip (instead of pip3)..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    fi

else
    echo -e "${NOCOL}[1/4] Starting to install PyTorch...${NOCOL}"
    pip3 install torch torchvision torchaudio
    if ! [[ $? -eq 0 ]]; then 
        echo "Using pip (instead of pip3)..."
        pip install torch torchvision torchaudio
    fi
fi

if [[ $? -eq 0 ]]; then
    echo -e "${GREEN}[1/4] \xE2\x9C\x94 Successfully installed PyTorch.${NOCOL}"
else
    echo -e "${RED}[1/4] \xE2\x9D\x8C Failed to install PyTorch.${NOCOL}"
    exit 1
fi

# ============================================================================================
# Our Python Dependencies
# ============================================================================================
echo ""
echo -e "${NOCOL}[2/4] Starting to install other Python dependencies...${NOCOL}"

pip3 install -r requirements.txt
if ! [[ $? -eq 0 ]]; then 
  echo "Using pip (instead of pip3)..."
  pip install -r requirements.txt 
fi

if [[ $? -eq 0 ]]; then
    echo -e "${GREEN}[2/4] \xE2\x9C\x94 Successfully installed other Python dependencies.${NOCOL}"
else
    echo -e "${RED}[2/4] \xE2\x9D\x8C Failed to install other Python dependencies.${NOCOL}"
    exit 1
fi

# ============================================================================================
# Download dataset
# ============================================================================================



# ============================================================================================
# Download models
# ============================================================================================
echo ""
echo -e "${NOCOL}[4/4] Starting to download and install models...${NOCOL}"

mkdir -p results
curl --ssl-no-revoke https://thuy-thesis.s3.ap-southeast-1.amazonaws.com/resnet50.zip --output ./results/resnet50.zip && unzip ./results/resnet50.zip -d ./results

if [[ $? -eq 0 ]]; then
    echo -e "${GREEN}[4/4] \xE2\x9C\x94 Successfully downloaded and installed models.${NOCOL}"
else
    echo -e "${RED}[4/4] \xE2\x9D\x8C Failed to download and install models.${NOCOL}"
    exit 1
fi

# ============================================================================================
# Run web app
# ============================================================================================
echo ""
echo -e "${NOCOL}Starting to run web app...${NOCOL}"

python3 ./web_app/app.py
if ! [[ $? -eq 0 ]]; then 
echo "Using python (instead of python3)..."
python ./web_app/app.py
fi

if [[ $? -eq 0 ]]; then
echo -e "${GREEN}\xE2\x9C\x94 Successfully installed other Python dependencies.${NOCOL}"
else
echo -e "${RED}\xE2\x9D\x8C Failed to install other Python dependencies.${NOCOL}"
exit 1
fi
