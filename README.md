# 🚗 Car Assistant LLM

A fine-tuned **Phi-2 + QLoRA** car diagnostic assistant that runs entirely locally on **Apple Silicon macOS**. Enter your car symptoms and optional OBD-II codes to receive a structured diagnosis with risk assessment and safety guidance.

---

## Project Overview

| Item | Details |
|---|---|
| **Base model** | `microsoft/phi-2` (2.7B parameters) |
| **Fine-tuning method** | QLoRA (LoRA r=8, α=16) via TRL SFTTrainer |
| **Training data** | `data/train.jsonl` — car symptom → structured diagnosis pairs |
| **Adapter** | `models/car-assistant-qlora/` |
| **UI** | Gradio dashboard with login, OBD lookup, warning light gallery |
| **Target platform** | Apple Silicon macOS (MPS) |

---

## Architecture

```
Car_Assistant_LLM/
├── app.py                          # Entrypoint — loads once, launches Gradio
├── requirements.txt                # Runtime dependencies (no bitsandbytes, no trl)
├── MIGRATION_REPORT.md             # Colab → local migration documentation
├── scripts/
│   └── test_load.py                # Phase 2 model verification script
├── src/
│   ├── config.py                   # Paths, constants, generation parameters
│   ├── model_loader.py             # Tokenizer + Phi-2 + LoRA adapter loading (MPS-aware)
│   ├── obd_utils.py                # OBD regex, CSV loader, lookup, search
│   ├── diagnosis.py                # Prompt, generate_once, fallback, diagnose()
│   └── ui.py                       # Full Gradio dashboard (build_app)
├── data/
│   ├── train.jsonl                 # Fine-tuning dataset
│   └── obd-trouble-codes.csv       # ~4000 OBD-II DTC codes
├── models/
│   └── car-assistant-qlora/
│       ├── adapter_config.json     # LoRA configuration
│       ├── adapter_model.safetensors  # LoRA weights (~7.5 MB)
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       └── checkpoint-14/          # Training checkpoint (fallback adapter)
├── assets/
│   └── icons/                      # Warning light PNGs (optional)
└── notebooks/
    └── finetuning.ipynb            # Original Colab notebook (source of truth)
```

### Data Flow

```
app.py
  │
  ├─ load_obd_db()      → obd_db dict
  ├─ load_model_and_tokenizer() → (model, tokenizer, device)
  │    ├─ get_device()           → "mps" | "cpu"
  │    ├─ load_tokenizer()       → AutoTokenizer (phi-2)
  │    ├─ load_base_model()      → AutoModelForCausalLM (float32)
  │    └─ load_peft_model()      → PeftModel (LoRA adapter attached)
  │
  └─ build_app(model, tokenizer, obd_db)
       └─ Gradio Blocks
            ├─ Login page
            └─ Main app
                 ├─ Assistant tab → diagnose(symptoms, obd_text) → LLM output
                 ├─ OBD DB tab   → obd_search / obd_lookup
                 ├─ Warning Lights tab → gallery + SVG placeholders
                 └─ Common Problems tab → static reference
```

---

## Installation

### Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.10 or 3.11
- ~12 GB free disk space (Phi-2 model weights download ~5 GB)
- ~12 GB RAM recommended (Phi-2 in float32 uses ~10–11 GB)

### 1. Clone / open the project

```bash
cd /path/to/Car_Assistant_LLM
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** `bitsandbytes` is intentionally excluded. 4-bit quantization is CUDA-only and not supported on Apple Silicon. The model loads in float32 instead.

---

## Setup

No additional setup is required. The LoRA adapter is already in `models/car-assistant-qlora/`.

On first launch, `microsoft/phi-2` (~5 GB) will be downloaded automatically to `~/.cache/huggingface/hub/`. Subsequent launches use the cached weights.

---

## Running Locally

```bash
python app.py
```

Then open **http://127.0.0.1:7860** in your browser.

**Login credentials:** `admin` / `car123`

Expected startup output:
```
============================================================
  Car Assistant LLM — Starting Up
============================================================

[app] Loading OBD database...
[app] OBD database ready: XXXX codes loaded.

[app] Loading model and tokenizer...
[model_loader] Loading tokenizer from 'microsoft/phi-2'...
[model_loader] Loading base model 'microsoft/phi-2' (float32, mps)...
[model_loader] Attaching LoRA adapter from '.../models/car-assistant-qlora'...
[model_loader] LoRA adapter attached successfully (primary).

[app] Model ready on device: mps

[app] Building Gradio interface...
[app] Launching Gradio...
[app] Open http://127.0.0.1:7860 in your browser.
```

---

## Model Verification

Run this **before** launching the full app to confirm the model loads and generates correctly:

```bash
python scripts/test_load.py
```

This script verifies:
1. torch, transformers, peft imports
2. MPS/CPU device detection
3. Adapter directory and required files
4. Tokenizer loading
5. Base model loading (float32)
6. LoRA adapter attachment
7. Model configuration summary
8. Test generation with a car diagnostic prompt

A green `✅ ALL VERIFICATION STEPS PASSED` at the end means the full app is ready.

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'bitsandbytes'`
This is expected and not a problem. `bitsandbytes` is removed from requirements because it requires CUDA. The model runs in float32 on MPS/CPU.

### Model loads but generation is very slow
- Phi-2 in float32 on MPS achieves approximately 5–15 tokens/second on M-series chips.
- First generation may be slower due to MPS kernel compilation.
- Avoid running other memory-intensive apps simultaneously.

### `RuntimeError: MPS backend out of memory`
- Close other applications.
- If it persists, the model will automatically fall back to CPU (add `device = "cpu"` override in `src/config.py`).

### Phi-2 download fails
- Ensure you have a stable internet connection for the first run.
- The model is cached at `~/.cache/huggingface/hub/models--microsoft--phi-2/`.
- To pre-download: `python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('microsoft/phi-2')"`

### `adapter_config.json` not found
- Confirm `models/car-assistant-qlora/` contains `adapter_config.json` and `adapter_model.safetensors`.
- Run `python scripts/test_load.py` for detailed diagnostics.

### OBD CSV not loading
- The file at `data/obd-trouble-codes.csv` is required for the full code database.
- If missing, the app falls back to a built-in set of 14 common OBD codes.

### Login not working
- Default credentials: username `admin`, password `car123`.
- Credentials are defined in `src/config.py` → `VALID_USERS`.

---

## Adapter Loading Explanation

The project uses a **LoRA adapter** (not a merged model). The inference flow is:

1. **Base model** (`microsoft/phi-2`) is loaded from HuggingFace Hub in float32.
2. **LoRA adapter** (`models/car-assistant-qlora/adapter_model.safetensors`, ~7.5 MB) is attached using `PeftModel.from_pretrained()`.
3. The combined model behaves as the fine-tuned Car Assistant.

**Why not merge?** The adapter can be swapped or updated without re-downloading the 5 GB base model. The `adapter_config.json` records the exact LoRA configuration used during training (r=8, α=16, target modules: q_proj, k_proj, v_proj, o_proj).

**Adapter priority:**
- Primary: `models/car-assistant-qlora/` (final saved adapter)
- Fallback: `models/car-assistant-qlora/checkpoint-14/` (mid-training checkpoint, also valid)

---

## Future Retraining Workflow

To retrain or fine-tune the adapter:

1. Prepare training data in `data/train.jsonl` using the `{"text": "<s>[INST] ... [/INST] ... </s>"}` format.
2. Use the original `notebooks/finetuning.ipynb` as the reference training script (or adapt the training cells into a `scripts/train.py`).
3. Install training dependencies separately:
   ```bash
   pip install trl datasets bitsandbytes  # bitsandbytes only for CUDA training
   ```
4. Set `OUT_DIR` to `models/car-assistant-qlora/` in the training script.
5. After training, verify the new adapter with `python scripts/test_load.py`.

The inference pipeline in `src/` does not need modification — it will automatically use the new adapter weights.

---

## Tech Stack

| Component | Library |
|---|---|
| Base LLM | `microsoft/phi-2` via HuggingFace Transformers |
| LoRA fine-tuning | PEFT 0.18.1 |
| Training framework | TRL SFTTrainer (training only) |
| Inference device | Apple Silicon MPS via PyTorch |
| UI | Gradio 4.x |
| OBD data | pandas CSV loader |
| Serialization | safetensors |
