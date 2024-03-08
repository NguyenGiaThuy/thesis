#!/bin/bash

# ============================================================================================
# Run web app
# ============================================================================================
echo ""
echo -e "${NOCOL}Starting to run web app...${NOCOL}"

python3 web_app/app.py
if ! [[ $? -eq 0 ]]; then 
echo "Using python (instead of python3)..."
python web_app/app.py
fi

if [[ $? -eq 0 ]]; then
echo -e "${GREEN}\xE2\x9C\x94 Successfully ran web app.${NOCOL}"
else
echo -e "${RED}\xE2\x9D\x8C Failed to run web app.${NOCOL}"
exit 1
fi
