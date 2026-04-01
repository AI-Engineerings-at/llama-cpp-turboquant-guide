---
title: TurboQuant on RTX 3090 — 100K Context, 4.3x KV-Cache Compression
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: static
pinned: true
license: cc-by-4.0
tags:
  - llm
  - quantization
  - kv-cache
  - turboquant
  - benchmark
  - rtx3090
  - consumer-hardware
  - mistral
  - llama-cpp
---

# llama-cpp-turboquant-guide

<div align="center">

![TurboQuant](https://img.shields.io/badge/TurboQuant-turbo3%20KV--Cache-blueviolet)
![Context](https://img.shields.io/badge/Context-100%2C000%20tokens-brightgreen)
![GPU](https://img.shields.io/badge/GPU-RTX%203090%2024GB-76b900?logo=nvidia)
![VRAM Overhead](https://img.shields.io/badge/VRAM%20overhead-%2B1.8%20GB%20only-blue)
![Speed Loss](https://img.shields.io/badge/Speed%20loss--8.5%25%20only-orange)
![License](https://img.shields.io/badge/license-CC%20BY%204.0-green)

**Practical guide: TurboQuant KV-cache quantization on consumer hardware.**
**100,000 token context on a single RTX 3090 — verified, reproducible, step-by-step.**

*Based on [TurboQuant (ICLR 2026, arXiv:2504.19874)](https://arxiv.org/abs/2504.19874)*

[Results](#-results) · [Quick Start](#-quick-start) · [How It Works](#-how-it-works) · [Errors & Fixes](#-errors--fixes) · [Deutsch](#-deutsch)

</div>

---

## 📊 Results

Tested on **NVIDIA RTX 3090 (24 GB VRAM)** with **Mistral-Small-3.2-24B Q4_K_M**.
Results are the average of **two independent benchmark runs** (April 1, 2026).

| | Baseline (f16) | TurboQuant turbo3 | Delta |
|--|:--------------:|:-----------------:|:-----:|
| **Context** | 8,192 tokens | **100,000 tokens** | **+12.2×** |
| **VRAM** | 15.5 GB | 17.4 GB | +1.9 GB only |
| **Tokens/s** | 50.2 | 46.0 | **−8.3%** |
| **KV-Cache size** | ~1 GB (f16) | ~2.8 GB (3-bit) | **4.3× compression** |

> **12× more context. +12% VRAM. −8% speed. Same model weights.**

**Run 1 (cold):** Baseline 49.2 TPS / 15,408 MB → Turbo3 45.0 TPS / 17,224 MB
**Run 2 (warm, idle GPU):** Baseline 51.2 TPS / 15,695 MB → Turbo3 47.1 TPS / 17,581 MB

Raw data: [`results/turboquant-rtx3090-2026-04-01.json`](results/turboquant-rtx3090-2026-04-01.json) · [`results/turboquant-rtx3090-2026-04-01-v2.json`](results/turboquant-rtx3090-2026-04-01-v2.json)

---

## 🚀 Quick Start

### 1. Build the Docker Image (~20 minutes)

```bash
docker build -t turboquant:feature .

# Verify TurboQuant is compiled in:
docker run --rm turboquant:feature llama-server -h 2>&1 | grep -A3 "cache-type-k"
# Must show: turbo2, turbo3, turbo4
```

### 2. Download a Model

```bash
# Set your HuggingFace token
export HF_TOKEN=hf_your_token_here

bash scripts/download-model.sh
```

### 3. Run Baseline (f16, 8K context)

```bash
bash scripts/run-baseline.sh
# Server starts on port 8180
```

### 4. Run TurboQuant (turbo3, 100K context)

```bash
bash scripts/run-turbo.sh
# Server starts on port 8182
```

### 5. Test It

```bash
# Check available context
curl -s http://localhost:8180/v1/models | jq '.data[0].context_length'
# Baseline: 8192

curl -s http://localhost:8182/v1/models | jq '.data[0].context_length'
# TurboQuant: 131072 (model max, allocated to 100000)

# Generate tokens (measures TPS in response)
curl http://localhost:8182/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"local","messages":[{"role":"user","content":"Count from 1 to 200"}],"max_tokens":500}'
```

---

## ⚙️ How It Works

### The KV-Cache Problem

When an LLM runs, it caches Key-Value pairs for every token in the context window.
This cache grows **linearly** with context length:

```
Mistral-Small-3.2 24B on RTX 3090 (24 GB total, ~14.4 GB for model weights):

Context    KV-Cache (f16)   Available after model   Fits?
  8,192        ~1 GB             9.6 GB               ✅
 32,000        ~4 GB             9.6 GB               ✅
100,000       ~12 GB             9.6 GB               ❌ OOM without TurboQuant
100,000     ~2.8 GB (turbo3)     9.6 GB               ✅
```

### What TurboQuant Does

TurboQuant compresses the KV-cache from 16-bit floats to 2–4-bit integers.
**It does NOT compress the model weights** — only the runtime cache.

```
f16 KV-Cache  →  turbo3 KV-Cache
  16 bits     →    3 bits  =  4.3× compression
```

The model reads the quantized cache and generates text normally.
Quality loss: <1% perplexity increase at turbo3 (per paper).

### Two Repos — Critical Distinction

There are two TurboQuant repositories with confusing names:

| Repo | What it is | When to use |
|------|-----------|-------------|
| `TheTom/turboquant_plus` | Python library for research | HuggingFace models, Python API |
| `TheTom/llama-cpp-turboquant` | llama.cpp fork | **This guide — llama-server** |

**This guide uses `TheTom/llama-cpp-turboquant`, branch `feature/turboquant-kv-cache`.**

---

## 🐛 Errors & Fixes

Every error we hit during setup, documented so you don't repeat them:

### E1: Wrong Repository

**Symptom:** No `turbo2`/`turbo3`/`turbo4` options after building.
**Cause:** Built from `TheTom/turboquant_plus` (Python library) instead of `TheTom/llama-cpp-turboquant`.
**Fix:** Use the correct repo. See Dockerfile.

### E2: Wrong cmake Flag

**Symptom:** CUDA not used during inference, slow CPU fallback.
**Cause:** Old flag `-DLLAMA_CUBLAS=ON` was renamed in llama.cpp post-GGML-refactor.
**Fix:**
```dockerfile
# WRONG (old, silently ignored):
cmake -DLLAMA_CUBLAS=ON -DLLAMA_CUDA=ON

# CORRECT:
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
```

### E3: libcuda.so.1 Not Found at Build Time

**Symptom:** Build fails with `cannot find -lcuda` or linker error for `libcuda.so.1`.
**Cause:** CUDA devel images have a stub `libcuda.so` but not `libcuda.so.1` (the runtime driver is injected at container start, not build time).
**Fix:** Add symlink before cmake:
```dockerfile
RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so \
           /usr/local/cuda/lib64/stubs/libcuda.so.1 \
    && echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/cuda-stubs.conf \
    && ldconfig
```

### E4: Wrong Branch

**Symptom:** `Unsupported cache type: turbo3` at runtime despite clean build.
**Cause:** Cloning the default `master` branch of `llama-cpp-turboquant` — which is a standard llama.cpp fork **without** TurboQuant. The implementation is on `feature/turboquant-kv-cache`.
**Fix:**
```bash
git clone https://github.com/TheTom/llama-cpp-turboquant.git \
  --branch feature/turboquant-kv-cache --depth=1
```
Always verify before building:
```bash
curl -s "https://api.github.com/repos/TheTom/llama-cpp-turboquant/branches" \
  | python3 -c "import sys,json; [print(b['name']) for b in json.load(sys.stdin)]"
```

### E5: Wrong HuggingFace Repo Name

**Symptom:** 404 or 401 when downloading model.
**Cause:** Model repo names change. Don't rely on memory or cached context.
**Fix:** Always query HF Search API before downloading:
```bash
curl -s -H "Authorization: Bearer $HF_TOKEN" \
  "https://huggingface.co/api/models?search=bartowski+mistral+small+3.2&limit=5" \
  | python3 -c "import sys,json; [print(m['modelId']) for m in json.load(sys.stdin)]"
```

---

## 🔬 Reproduce Our Results

```bash
# 1. Build
docker build -t turboquant:feature .

# 2. Download model (~14 GB)
export HF_TOKEN=hf_your_token
bash scripts/download-model.sh

# 3. Baseline measurement
bash scripts/run-baseline.sh &
sleep 45  # wait for server startup
curl -s http://localhost:8180/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"local","messages":[{"role":"user","content":"Count from 1 to 200, one per line."}],"max_tokens":500}' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); u=d['usage']; print(f'TPS: {u[\"completion_tokens\"] / (d[\"usage\"].get(\"total_time_ms\",10000)/1000):.1f}')"
nvidia-smi --query-gpu=memory.used --format=csv,noheader

# 4. Turbo3 measurement
docker stop turboquant-baseline
bash scripts/run-turbo.sh &
sleep 90  # 100K context allocation takes longer
# repeat curl + nvidia-smi on port 8182
```

Expected results matching our run: see [`results/turboquant-rtx3090-2026-04-01.json`](results/turboquant-rtx3090-2026-04-01.json)

---

## Hardware Requirements

| | Minimum | Our Setup |
|--|---------|----------|
| GPU VRAM | 16 GB | RTX 3090 24 GB |
| System RAM | 16 GB | 32 GB |
| Disk | 30 GB | SSD |
| CUDA | 12.x | 12.6.3 |
| OS | Linux / Windows + Docker | Windows + Docker Desktop |

> **Note on Windows:** Docker Desktop works fine. Avoid `/tmp/` paths — use named Docker volumes for model storage.

---

## Model Compatibility

Tested with **Mistral-Small-3.2-24B Q4_K_M** (14 GB).
Should work with any GGUF model that fits the VRAM budget after KV-cache allocation.

| Model | Size | VRAM (model) | Max ctx (turbo3) |
|-------|------|-------------|-----------------|
| Mistral-Small-3.2 24B Q4_K_M | 14 GB | 14.4 GB | ~100K on 24 GB GPU |
| Llama-3.1 8B Q4_K_M | 4.7 GB | 5.1 GB | ~200K on 16 GB GPU |
| Qwen2.5 14B Q4_K_M | 8.5 GB | 8.8 GB | ~150K on 16 GB GPU |

*Estimates. Actual values depend on architecture and batch size.*

---

## 📄 License

Content and scripts: [CC BY 4.0](LICENSE)
Based on [TurboQuant (arXiv:2504.19874)](https://arxiv.org/abs/2504.19874) by Thomas et al. (ICLR 2026)
llama.cpp fork: [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant)

---

---

## 🇩🇪 Deutsch

### TurboQuant auf Consumer-Hardware — Praktischer Guide

Dieses Repository dokumentiert unsere Erfahrungen beim Einsatz von TurboQuant (ICLR 2026)
auf einer RTX 3090 im Homelab-Betrieb. Wir sind das erste europäische Team,
das diese Methode praktisch auf Consumer-Hardware veröffentlicht dokumentiert hat.

### Das Ergebnis

Mit TurboQuant turbo3 (3-bit KV-Cache) haben wir auf einer RTX 3090 (24 GB):

- **12× mehr Context** (8.192 → 100.000 Tokens)
- nur **+1.8 GB VRAM** Mehrverbrauch
- nur **−8.5% Geschwindigkeitsverlust**
- **gleiche Modellgewichte** — nur der Laufzeit-Cache wird komprimiert

### Warum das wichtig ist

Größerer Context bedeutet: Längere Dokumente, mehr Gesprächshistorie, besseres RAG,
Code-Analyse ganzer Codebasen — alles auf einer einzigen Consumer-GPU.

### Fehler-Protokoll (5 Fehler die wir gemacht haben)

Alle 5 Fehler aus unserem Setup sind unter [Errors & Fixes](#-errors--fixes) dokumentiert.
Der häufigste: falscher Branch (`master` statt `feature/turboquant-kv-cache`).

### Schnellstart (Deutsch)

```bash
# Image bauen (~20 Minuten)
docker build -t turboquant:feature .

# Modell herunterladen (14 GB)
export HF_TOKEN=dein_token
bash scripts/download-model.sh

# Baseline starten (f16, 8K Context)
bash scripts/run-baseline.sh

# TurboQuant starten (turbo3, 100K Context)
bash scripts/run-turbo.sh
```

Vollständige deutsche Dokumentation: [`WHITEPAPER.de.md`](WHITEPAPER.de.md)

---

*AI Engineering Lab · April 2026 · [ai-engineering.at](https://ai-engineering.at)*
