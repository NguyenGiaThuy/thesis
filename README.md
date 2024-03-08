# A Transformer-Based Multi-Modal Fake News Detection

## I. REQUIREMENTS

- Python Version: `3.10`

## I. TRAINING

## II. DEMO APPLICATION

This is a proof of concept to demo our research. The demo application can be run by any of the following ways.

### 2.1. Local

At the root directory, run:

```bash
./scripts/install.sh
./scripts/start.sh
```

### 2.2 Docker

At the root directory, run:

```bash
docker build -t fakenewsdetection .
docker run -d --name fakenewsdetection fakenewsdetection
```

### 2.3. Docker Compose (Production)

The gateway logs are streamed to the files in `compose\nginx\logs` in the host machine.

```
cd compose
docker compose up
```
