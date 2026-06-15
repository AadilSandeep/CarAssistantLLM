# Car Assistant LLM — Migration Report

**Status:** Phase 0 Complete (Analysis)
**Date:** 2026-06-15
**Target:** Apple Silicon macOS (M-series) local Python project

---

## 1. Current Architecture

The project originated as a **Google Colab notebook** (`notebooks/finetuning.ipynb`) that was never migrated to a proper Python package. The notebook contains all logic inline across sequential cells. The root `app.py` is **empty** (0 bytes). The `src/` directory is **empty**.

### Execution flow (as-found in notebook)

| Cell group | Purpose |
|---|---|
| Cell 1 | GPU check (`nvidia-smi`) — Colab/CUDA specific |
| Cell 2 | `pip install nbformat` |
| Cell 3 | Drive mount (`google.colab`) + dataset loading from `/content/drive/MyDrive/mini/content/train.jsonl` |
| Cell 4 | Fine-tuning config + training loop (SFTTrainer / TRL) |
| Cell 5 | Model verification test generation — uses `/content/car-assistant-qlora` |
| Cell 6 | OBD helper functions (regex + dict lookup) |
| Cell 7 | Model loading (base + LoRA adapter from Drive) |
| Cell 8 | `generate_once()`, `baseline_fallback()`, `diagnose()` — core inference |
| Cell 9 | First Gradio UI (simple version, `demo.launch(share=True)`) |
| Cell 10 | Full Gradio dashboard (login, sidebar, OBD DB, warning lights gallery, common problems) |

---

## 2. Discovered Dependencies

From notebook pip cells and `requirements.txt`:

| Package | Role | Status |
|---|---|---|
| `torch` | Core ML framework | Keep — replace CUDA with MPS |
| `transformers` | Phi-2 model + tokenizer | Keep |
| `peft` | LoRA adapter loading (`PeftModel`) | Keep |
| `accelerate` | Device dispatch helpers | Keep |
| `bitsandbytes` | 4-bit quantization (NF4) | **RISK** — not supported on Apple Silicon MPS |
| `trl` | SFTTrainer (training only) | Drop from runtime requirements |
| `gradio` | UI framework | Keep |
| `pandas` | OBD CSV loading | Keep |
| `datasets` | Training data loading (training only) | Drop from runtime requirements |
| `safetensors` | Loading adapter weights | Keep |
| `nbformat` | Notebook introspection (colab cell) | Drop |
| `google-colab` | Drive mount | **REMOVE** |

### Critical bitsandbytes issue
`bitsandbytes` 4-bit quantization (`BitsAndBytesConfig` / `load_in_4bit=True`) is a **CUDA-only feature**. It does **not** work on Apple Silicon MPS. The model must be loaded in **float32 or float16** without quantization on macOS, which increases RAM requirements (~5–6 GB for Phi-2 in float32).

---

## 3. Existing Model Configuration

- **Base model:** `microsoft/phi-2` (from HuggingFace Hub)
- **Adapter directory:** `models/car-assistant-qlora/`
- **Adapter file:** `adapter_model.safetensors` (~7.5 MB — LoRA weights only)
- **Checkpoint:** `models/car-assistant-qlora/checkpoint-14/` (contains optimizer state + adapter)
- **Inference mode:** `true` in adapter config

---

## 4. Existing Tokenizer Configuration

From `models/car-assistant-qlora/tokenizer_config.json`:

```json
{
  "add_prefix_space": false,
  "bos_token": "<|endoftext|>",
  "eos_token": "<|endoftext|>",
  "pad_token": "<|endoftext|>",
  "model_max_length": 384,
  "tokenizer_class": "TokenizersBackend"
}
```

**Notebook overrides used at training/inference time:**
- `tokenizer.pad_token = tokenizer.eos_token` (if pad_token is None)
- `tokenizer.padding_side = "right"`
- `tokenizer.model_max_length = 384`

---

## 5. Existing PEFT Configuration

From `models/car-assistant-qlora/adapter_config.json`:

```json
{
  "base_model_name_or_path": "microsoft/phi-2",
  "peft_type": "LORA",
  "task_type": "CAUSAL_LM",
  "r": 8,
  "lora_alpha": 16,
  "lora_dropout": 0.05,
  "bias": "none",
  "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
  "inference_mode": true,
  "peft_version": "0.18.1"
}
```

---

## 6. Existing Generation Parameters

From notebook inference cells:

| Parameter | Value |
|---|---|
| `max_new_tokens` | 200 (main), 120 (quick test) |
| `do_sample` | True |
| `temperature` | 0.5 (first attempt), 0.2 (retry) |
| `top_p` | 0.9 |
| `repetition_penalty` | 1.25 |
| `bad_words_ids` | Computed from bad_words list (code/HTML prevention) |
| `pad_token_id` | `tokenizer.eos_token_id` |
| `eos_token_id` | `tokenizer.eos_token_id` |

**Generation retry logic:**
1. Call `generate_once(symptoms, obd_text, temperature=0.5)`
2. If `looks_valid()` → return
3. Call `generate_once(symptoms, obd_text, temperature=0.2)` (retry)
4. If `looks_valid()` → return
5. Fall through to `baseline_fallback()` (rule-based)

**Validity check:** At least 4 of 7 required headings must appear in the output.

---

## 7. Existing Gradio Structure

Two UI implementations exist in the notebook:

### Simple UI (Cell 9 — earlier version)
- `gr.Blocks` with title "Car Assistant LLM — Diagnosis + OBD-II"
- Two textboxes: Symptoms (4 lines), OBD-II Code (1 line)
- One button: "Diagnose ✅"
- Output: Textbox (16 lines)
- Examples list (4 hardcoded examples)
- Clear button
- `demo.launch(share=True, debug=False)`

### Full Dashboard UI (Cell 10 — final version, used for migration)
- `gr.Blocks` with title "Car Assistant LLM — Dashboard"
- Custom CSS (topbar, sidebar-btn, gallery)
- Soft theme (`gr.themes.Soft()`)
- **Login page:** username/password (admin/car123)
- **Main app (post-login):**
  - Sidebar: Car Assistant | OBD Code Database | Warning Light Icons | Common Problems | Logout
  - Page: Assistant — symptoms + OBD input → diagnosis output
  - Page: OBD DB — search + lookup from loaded CSV
  - Page: Warning Lights — gallery with SVG placeholders, click for details
  - Page: Common Problems — static text reference
- `app.launch(share=True, theme=theme, css=CSS)`

---

## 8. Existing OBD Workflow

### OBD inline dict (notebook Cell 6)
Seven hardcoded codes used as quick lookup fallback:
`P0300, P0301, P0171, P0420, P0128, P0455, P0700`

### OBD CSV database (Cell 10)
- File: `data/obd-trouble-codes.csv`
- Columns: `"P0100","Mass or Volume Air Flow Circuit Malfunction"` (no header row — first column is code, second is description)
- Loader: `load_obd_db()` — auto-detects column names with candidates list, falls back to positional (col[0]=code, col[1]=desc)
- Functions: `obd_lookup()`, `obd_search()`, `obd_list_text()`

### OBD regex
```python
OBD_RE = re.compile(r"\bP[0-3][0-9A-F]{3}\b", re.IGNORECASE)
```
Used in `normalize_obd_codes()` to extract codes from free text.

---

## 9. Identified Risks

| Risk | Severity | Mitigation |
|---|---|---|
| `bitsandbytes` 4-bit quant not supported on MPS | **HIGH** | Load model in `torch_dtype=torch.float32` without quantization on Apple Silicon |
| Phi-2 (~2.7B params) requires ~5–6 GB RAM in float32 | Medium | Acceptable on MacBook Pro with 16+ GB unified memory |
| `tokenizer_class: TokenizersBackend` — may require `use_fast=True` | Medium | Use `AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)` |
| `bad_words_ids` computed at startup requires tokenizer to be ready | Low | Compute once at init time, store globally |
| OBD CSV has no header row (codes are quoted) | Low | `pd.read_csv` with `header=None` and positional column access |
| Gradio `share=True` requires internet for tunneling | Low | Change to `share=False` for local-only operation |
| Model download from HuggingFace Hub on first run (~5 GB) | Medium | Document clearly in README; use `~/.cache/huggingface` |
| `checkpoint-14` contains optimizer state (not needed for inference) | Low | Use top-level `models/car-assistant-qlora/` adapter directory |

---

## 10. Migration Plan — STATUS

### Phase 0 ✅ — Migration Report (complete)
Full project analysis documented. MIGRATION_REPORT.md created before any code changes.

### Phase 1 ✅ — Project Analysis (complete)
Notebook fully read (1423 lines). All cells analyzed:
- Base model: `microsoft/phi-2` confirmed
- LoRA config: r=8, α=16, target=[q_proj, k_proj, v_proj, o_proj]
- Generation params: max_new_tokens=200, temp=0.5/0.2, top_p=0.9, rep_penalty=1.25
- Prompt format: `<s>[INST] ... [/INST]\n` (matches train.jsonl format)
- OBD CSV: no-header, positional columns (col0=code, col1=desc)
- UI: full dashboard with login, sidebar, 4 pages

### Phase 2 ✅ — Model Verification Script (complete)
Created `scripts/test_load.py` with:
- 8 verification steps with explicit pass/fail output
- MPS device detection (no CUDA assumptions)
- float32 loading (no bitsandbytes)
- Primary + checkpoint-14 fallback adapter loading
- Full test generation with notebook-identical prompt

### Phase 3 ✅ — Remove Colab Dependencies (complete)
All Colab-specific code removed:
- `from google.colab import drive` → not present in any src/ file
- `drive.mount(...)` → not present
- `/content/drive/MyDrive/mini/content/...` → replaced with `pathlib` paths in `src/config.py`
- `/content/car-assistant-qlora` → `PROJECT_ROOT / "models" / "car-assistant-qlora"`
- `nvidia-smi` shell command → not present
- CUDA/BitsAndBytes config → removed

### Phase 4 ✅ — Clean Project Structure (complete)
Created:
- `src/__init__.py` — package init
- `src/config.py` — all paths (pathlib), all constants, generation params, OBD hints, UI config
- `src/model_loader.py` — MPS-aware float32 loading, primary+fallback adapter, `load_model_and_tokenizer()`
- `src/obd_utils.py` — OBD_RE, normalize_obd_codes, explain_obd, load_obd_db (header=None), obd_lookup, obd_search
- `src/diagnosis.py` — SYSTEM_FORMAT, bad_words, generate_once, looks_valid, baseline_fallback, diagnose
- `src/ui.py` — full Gradio dashboard via `build_app(model, tokenizer, obd_db)`

### Phase 5 ✅ — Application Entrypoint (complete)
`app.py` (was empty, now a clean entrypoint):
- Loads OBD DB once
- Loads model + tokenizer once
- Calls `build_app()` and launches Gradio on http://127.0.0.1:7860

### Phase 6 ✅ — Requirements Cleanup (complete)
`requirements.txt` updated:
- Removed: `bitsandbytes` (CUDA-only), `trl` (training-only), `datasets` (training-only), `nbformat` (Colab utility)
- Kept: `torch`, `transformers`, `peft`, `accelerate`, `safetensors`, `gradio`, `pandas`
- Added version lower bounds appropriate for Apple Silicon

### Phase 7 ✅ — Reliability Review (complete)
Fixes applied:
- OBD CSV: `pd.read_csv(path, header=None)` — correctly handles no-header format
- Prompt format: matches `train.jsonl` exactly (`<s>[INST] ... [/INST]\n response </s>`)
- bad_words_ids: computed once via `init_bad_words(tokenizer)` at startup, not per-request
- MPS: no quantization, no CUDA-only ops; device determined dynamically
- Adapter loading: primary path first, graceful fallback to checkpoint-14
- Model injection: model/tokenizer passed as arguments (no mutable globals in inference path)

### Phase 8 ✅ — Documentation (complete)
`README.md` generated with:
1. Project overview and tech stack table
2. Architecture diagram (text + data flow)
3. Folder structure
4. Prerequisites and installation steps
5. Setup notes (first-run download, no extra setup)
6. Running locally + expected startup output
7. Model verification command
8. Troubleshooting (6 common issues)
9. Adapter loading explanation
10. Future retraining workflow

---

**All phases complete. Migration ready for model verification.**
