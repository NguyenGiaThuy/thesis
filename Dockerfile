FROM pytorch/pytorch

SHELL ["/bin/bash", "-c"]

WORKDIR /app

COPY  configs/*         \
      models/*          \
      utils/*           \
      web_app/*         \
      main.py           \
      requirements.txt  \
      start.sh          \
      ./

RUN apt update -y && apt upgrade -y
RUN apt install -y curl vim
RUN cd /app && ./start.sh
