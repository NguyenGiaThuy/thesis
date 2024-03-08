# A Transformer-Based Multi-Modal Fake News Detection

## I. TRAINING

## II. DEMO APPLICATION

### 2.1. Requirements

- Python Version: `3.10`
- CUDA Version: $\geq$ `12.1`

### 2.2. Quick Start (Local)

```bash
./scripts/install.sh
./scripts/start.sh
```

### 2.3. Quick Start (Docker)

```bash
docker build -t fakenewsdetection .
docker run -d --name fakenewsdetection fakenewsdetection
```

### 2.4. Docker Compose
```
cd compose
docker compose up
```
