FROM pytorch/pytorch

SHELL ["/bin/bash", "-c"]

RUN apt update -y && apt upgrade -y
RUN apt install -y curl vim zip

WORKDIR /app

RUN mkdir -p configs models scripts utils web_app

COPY configs            /app/configs
COPY models             /app/models
COPY scripts            /app/scripts
COPY utils              /app/utils
COPY web_app            /app/web_app
COPY main.py            .
COPY requirements.txt   .

RUN scripts/install.sh

CMD ["scripts/start.sh"]
