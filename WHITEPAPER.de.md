# TurboQuant auf Consumer-Hardware
## 100.000 Token Context auf einer RTX 3090 — Schritt für Schritt

> AI Engineering Lab | April 2026
> Getestet auf: NVIDIA RTX 3090 (24 GB VRAM), Windows + Docker Desktop
> Modell: Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M

---

## Executive Summary

TurboQuant ist eine KV-Cache-Quantisierungsmethode aus dem Paper
"TurboQuant: Ultra-Low-Bit KV-Cache Quantization for LLMs" (ICLR 2026, arXiv:2504.19874).

**Das Ergebnis in einer Zeile:**
Mit TurboQuant turbo3 (3-bit KV-Cache) erreichen wir auf einer RTX 3090
einen Context von **100.000 Tokens** — bei nur 8,5% Geschwindigkeitsverlust
und ohne Änderung der Modellgewichte.

| | Baseline (f16) | TurboQuant turbo3 | Delta |
|--|:---:|:---:|:---:|
| **Context** | 8.192 | **100.000** | **+12,2×** |
| **VRAM** | 15,4 GB | 17,2 GB | +1,8 GB |
| **Tokens/s** | 49,2 | 45,0 | −8,5% |
| **KV-Cache** | ~1 GB (f16) | ~2,8 GB (3-bit) | 4,3× Kompression |

---

## 1. Das Problem: Der KV-Cache frisst VRAM

Wenn ein LLM läuft, berechnet es für jeden Token Key-Value-Paare (KV).
Diese werden im VRAM gecacht, damit spätere Tokens darauf zugreifen können.

Das Problem: Der KV-Cache **wächst linear mit der Kontextlänge**.

Für Mistral-Small-3.2 24B auf einer RTX 3090 (24 GB, davon ~14,4 GB für Modellgewichte):

```
Context       KV-Cache (f16)    Verfügbar nach Modell    Passt?
  8.192           ~1 GB              9,6 GB                 ✅
 32.000           ~4 GB              9,6 GB                 ✅
100.000          ~12 GB              9,6 GB                 ❌ OOM
100.000        ~2,8 GB (turbo3)      9,6 GB                 ✅
```

100K Context ist ohne Optimierung auf einer 24-GB-GPU schlicht nicht möglich.

---

## 2. Was TurboQuant macht

TurboQuant komprimiert den KV-Cache von 16-bit auf 2–4-bit.
**NICHT die Modellgewichte** — nur den Laufzeit-Cache.

```
f16 KV-Cache (16 bit) → turbo3 KV-Cache (3 bit) = 4,3× weniger Speicher
```

Das Modell liest den quantisierten Cache und generiert Text ganz normal.
Qualitätsverlust: laut Paper <1% Perplexity-Anstieg bei turbo3.

---

## 3. Das Ecosystem — Zwei Repos, ein häufiger Fehler

Es gibt zwei TurboQuant-Repositories mit verwirrenden Namen:

| Repo | Was es ist | Wann benutzen |
|------|-----------|--------------|
| `TheTom/turboquant_plus` | Python-Bibliothek | HuggingFace-Modelle, Forschung |
| `TheTom/llama-cpp-turboquant` | llama.cpp-Fork | **Dieser Guide — llama-server** |

**Dieser Guide verwendet `TheTom/llama-cpp-turboquant`, Branch `feature/turboquant-kv-cache`.**

Kritisch: Der Default-Branch `master` ist ein normales llama.cpp **ohne TurboQuant**.
Die Implementierung liegt auf `feature/turboquant-kv-cache`.

---

## 4. Setup — Schritt für Schritt

### 4.1 Branch verifizieren (vor dem Build!)

```bash
curl -s "https://api.github.com/repos/TheTom/llama-cpp-turboquant/branches" \
  | python3 -c "import sys,json; [print(b['name']) for b in json.load(sys.stdin)]"
# Erwartet: feature/turboquant-kv-cache, master
```

### 4.2 Docker Image bauen (~20 Minuten)

```bash
docker build -t turboquant:feature .

# Verifizieren: turbo2, turbo3, turbo4 müssen erscheinen
docker run --rm turboquant:feature llama-server -h 2>&1 | grep -A3 "cache-type-k"
```

### 4.3 Modell herunterladen (~14 GB)

```bash
export HF_TOKEN=hf_dein_token
bash scripts/download-model.sh
```

### 4.4 Baseline starten (Referenzwert)

```bash
bash scripts/run-baseline.sh
# → Port 8180, f16 KV-Cache, 8192 Context
```

### 4.5 TurboQuant starten

```bash
bash scripts/run-turbo.sh
# → Port 8182, turbo3 KV-Cache, 100.000 Context
```

---

## 5. Fehler-Protokoll

Alle 5 Fehler aus unserem Setup — damit du sie nicht wiederholst:

### Fehler 1: Falsches Repository
**Symptom:** Kein `turbo2`/`turbo3`/`turbo4` nach dem Build.
**Ursache:** `TheTom/turboquant_plus` (Python-Bibliothek) statt `TheTom/llama-cpp-turboquant` gebaut.
**Fix:** Richtiges Repo verwenden — siehe Dockerfile.

### Fehler 2: Falsches cmake-Flag
**Symptom:** Kein CUDA, CPU-Fallback.
**Ursache:** `-DLLAMA_CUBLAS=ON` wurde umbenannt.
**Fix:** `-DGGML_CUDA=ON` (modernes llama.cpp, post-GGML-Refactor).

### Fehler 3: libcuda.so.1 fehlt beim Build
**Symptom:** Linker-Fehler beim cmake-Build.
**Ursache:** Docker-Build-Time hat keinen NVIDIA-Treiber — nur ein Stub ohne `.1`-Suffix.
**Fix:** Symlink VOR cmake setzen (siehe Dockerfile).

### Fehler 4: Falscher Branch
**Symptom:** `Unsupported cache type: turbo3` zur Laufzeit.
**Ursache:** Master-Branch geklont (standard llama.cpp, kein TurboQuant).
**Fix:** `git clone --branch feature/turboquant-kv-cache`

### Fehler 5: Falscher HuggingFace-Repo-Name
**Symptom:** 404 beim Modell-Download.
**Ursache:** Repo-Namen aus dem Gedächtnis rekonstruiert — falsch.
**Fix:** Immer live via HF Search API verifizieren (nie aus dem Kontext nehmen).

---

## 6. Benchmark-Methodik

Jede Messung:
1. VRAM messen nach Server-Start (nvidia-smi)
2. 3× curl an `/v1/chat/completions` mit "Count from 1 to 200"
3. Durchschnitt aus 3 Läufen
4. Container stoppen, 30s warten, nächste Messung

TPS-Berechnung: `completion_tokens / (total_duration_ms / 1000)`

---

## 7. Produktions-Checkliste

Bevor du TurboQuant in Produktion einsetzt:

- [ ] Image gebaut von `feature/turboquant-kv-cache` (NICHT master)
- [ ] Verifiziert: `llama-server -h | grep turbo` zeigt turbo2, turbo3, turbo4
- [ ] VRAM-Budget berechnet: Modell + KV-Cache + Overhead ≤ GPU-VRAM
- [ ] Port-Konflikte geprüft (kein anderer Service auf dem Port)
- [ ] Startup-Zeit eingeplant: 100K Context braucht ~90s Startzeit
- [ ] Qualität getestet: Stichproben-Outputs mit turbo3 vs f16 verglichen
- [ ] Modell-Download via HF Search API verifiziert (nicht aus Erinnerung)

---

## 8. Rohdaten

Alle Benchmark-Rohdaten: [`results/turboquant-rtx3090-2026-04-01.json`](results/turboquant-rtx3090-2026-04-01.json)

---

*AI Engineering Lab · April 2026 · [ai-engineering.at](https://ai-engineering.at)*
*Basierend auf TurboQuant (arXiv:2504.19874) von Thomas et al. (ICLR 2026)*
