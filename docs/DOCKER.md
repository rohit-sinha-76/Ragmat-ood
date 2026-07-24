# Docker Containerization Guide

This document describes how to build, run, and evaluate the RAGMat-OOD project within a GPU-accelerated Docker environment.

---

## 1. Prerequisites

To run with full GPU acceleration, the host machine must have:
1. **NVIDIA Driver** installed on the host.
2. **Docker Engine** installed on the host.
3. **NVIDIA Container Toolkit** installed to pass GPU devices to the container:
 - [NVIDIA Container Toolkit Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

If running in Windows Subsystem for Linux (WSL), the NVIDIA Container Toolkit is fully supported inside the WSL terminal out-of-the-box once installed on the Windows host.

---

## 2. Building the Image

Build the Docker image from the root workspace folder:
```bash
docker build -t ragmat-ood:latest .
```

Alternatively, build using Docker Compose:
```bash
docker-compose build
```

---

## 3. Running Container Tasks

### Run the Test Suite
Verify environment and model integrity:
```bash
# Using raw Docker
docker run --rm ragmat-ood:latest

# Using Docker Compose
docker-compose run --rm ragmat-ood
```

### Reproduce Key Results
Execute the unified reproduction scripts to recalculate CIs, run gating analyzers, and compile interpretability reports:
```bash
# Using raw Docker
docker run --rm \
 -v "$(pwd)/data:/app/data" \
 -v "$(pwd)/final_result:/app/final_result" \
 -v "$(pwd)/checkpoints:/app/checkpoints" \
 ragmat-ood:latest \
 bash scripts/reproduce_all.sh

# Using Docker Compose
docker-compose run --rm ragmat-ood bash scripts/reproduce_all.sh
```

### Train GNN (Tier 1) Models on GPU
Run the training module with full GPU mapping (`--gpus all` or `deploy.resources.reservations.devices` in Compose):
```bash
# Using raw Docker
docker run --rm --gpus all \
 -v "$(pwd)/data:/app/data" \
 -v "$(pwd)/final_result:/app/final_result" \
 -v "$(pwd)/checkpoints:/app/checkpoints" \
 ragmat-ood:latest \
 python -m ragmat.train configs/tier1_cgcnn.yaml

# Using Docker Compose
docker-compose run --rm ragmat-ood python -m ragmat.train configs/tier1_cgcnn.yaml
```

---

## 4. Volume Mounts Reference

To maintain caches and output predictions across container lifecycles, the container utilizes three key volume mounts:

| Host Directory | Container Directory | Purpose |
|---|---|---|
| `./data` | `/app/data` | Persists JARVIS-DFT raw JSON downloads, graph `.pt` caches, and FAISS retrieval indices. |
| `./final_result` | `/app/final_result` | Collects prediction CSV files, bootstrap results, gating final reports, and logs. |
| `./checkpoints` | `/app/checkpoints` | Persists trained GNN model weights (`.pt` files). |

---

## 5. Security & User Context

The container runs under a non-root user context (`researcher`, UID `1000`, GID `1000`).
When mounting host folders into the container, ensure that the files and directories on the host are readable and writable by your local host user (which usually maps to UID `1000` on Linux/WSL).
