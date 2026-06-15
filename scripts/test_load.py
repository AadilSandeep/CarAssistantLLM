"""
scripts/test_load.py
====================
Phase 2 — Model Verification Script

Verifies that:
  1. Tokenizer loads from microsoft/phi-2
  2. Base Phi-2 model loads (float32, MPS/CPU — no bitsandbytes)
  3. LoRA adapter attaches from models/car-assistant-qlora/
  4. A test generation runs successfully

Usage:
    python scripts/test_load.py

STOP CONDITION: All further implementation is gated on this passing.
"""

import sys
import time
from pathlib import Path

# ── Add project root to path so src/ is importable ───────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

ADAPTER_DIR = PROJECT_ROOT / "models" / "car-assistant-qlora"
BASE_MODEL  = "microsoft/phi-2"

# ── Step 0: Imports ───────────────────────────────────────────────────────────
print("=" * 60)
print("Car Assistant LLM — Model Verification")
print("=" * 60)
print(f"\nProject root : {PROJECT_ROOT}")
print(f"Adapter dir  : {ADAPTER_DIR}")
print(f"Base model   : {BASE_MODEL}\n")

try:
    import torch
    print(f"[OK] torch {torch.__version__}")
except ImportError as e:
    print(f"[FAIL] torch not found: {e}")
    sys.exit(1)

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import transformers
    print(f"[OK] transformers {transformers.__version__}")
except ImportError as e:
    print(f"[FAIL] transformers not found: {e}")
    sys.exit(1)

try:
    from peft import PeftModel
    import peft
    print(f"[OK] peft {peft.__version__}")
except ImportError as e:
    print(f"[FAIL] peft not found: {e}")
    sys.exit(1)

# ── Step 1: Device detection ──────────────────────────────────────────────────
print("\n--- Step 1: Device Detection ---")
if torch.backends.mps.is_available():
    device = "mps"
    print("[OK] Apple Silicon MPS detected — using MPS")
else:
    device = "cpu"
    print("[WARN] MPS not available — falling back to CPU")
print(f"Device: {device}")

# ── Step 2: Check adapter directory ──────────────────────────────────────────
print("\n--- Step 2: Adapter Directory Check ---")
if not ADAPTER_DIR.exists():
    print(f"[FAIL] Adapter directory not found: {ADAPTER_DIR}")
    sys.exit(1)

required_files = ["adapter_config.json", "adapter_model.safetensors"]
for f in required_files:
    p = ADAPTER_DIR / f
    if p.exists():
        print(f"[OK] {f} ({p.stat().st_size / 1024:.1f} KB)")
    else:
        print(f"[FAIL] Missing: {f}")
        sys.exit(1)

# ── Step 3: Load Tokenizer ────────────────────────────────────────────────────
print(f"\n--- Step 3: Loading Tokenizer from {BASE_MODEL} ---")
print("(Downloads to ~/.cache/huggingface on first run)\n")
t0 = time.time()
try:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = 384
    elapsed = time.time() - t0
    print(f"[OK] Tokenizer loaded in {elapsed:.1f}s")
    print(f"     vocab size    : {tokenizer.vocab_size}")
    print(f"     pad token     : {tokenizer.pad_token!r}")
    print(f"     eos token     : {tokenizer.eos_token!r}")
    print(f"     model_max_len : {tokenizer.model_max_length}")
except Exception as e:
    print(f"[FAIL] Tokenizer load failed: {e}")
    sys.exit(1)

# ── Step 4: Load Base Model ───────────────────────────────────────────────────
print(f"\n--- Step 4: Loading Base Model ({BASE_MODEL}) ---")
print("NOTE: bitsandbytes 4-bit quantization is CUDA-only.")
print("Loading in float32 without quantization for Apple Silicon.\n")
t0 = time.time()
try:
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True,       # Required for Phi-2
    )
    base_model.eval()
    elapsed = time.time() - t0
    param_count = sum(p.numel() for p in base_model.parameters())
    print(f"[OK] Base model loaded in {elapsed:.1f}s")
    print(f"     parameters  : {param_count / 1e9:.2f}B")
    print(f"     dtype       : {next(base_model.parameters()).dtype}")
except Exception as e:
    print(f"[FAIL] Base model load failed: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ── Step 5: Move to Device ────────────────────────────────────────────────────
print(f"\n--- Step 5: Moving Model to {device} ---")
t0 = time.time()
try:
    base_model = base_model.to(device)
    elapsed = time.time() - t0
    print(f"[OK] Model moved to {device} in {elapsed:.1f}s")
except Exception as e:
    print(f"[WARN] Could not move to {device}, staying on CPU: {e}")
    device = "cpu"

# ── Step 6: Attach LoRA Adapter ───────────────────────────────────────────────
print(f"\n--- Step 6: Attaching LoRA Adapter from {ADAPTER_DIR} ---")
t0 = time.time()
try:
    model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))
    model.eval()
    elapsed = time.time() - t0
    print(f"[OK] LoRA adapter attached in {elapsed:.1f}s")
    print(f"     peft type   : LORA")
    print(f"     adapter dir : {ADAPTER_DIR}")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"     trainable   : {trainable:,} / {total:,} params")
except Exception as e:
    print(f"[FAIL] LoRA adapter load failed: {e}")
    import traceback; traceback.print_exc()
    print("\nAttempting fallback: checkpoint-14...")
    checkpoint_dir = ADAPTER_DIR / "checkpoint-14"
    if checkpoint_dir.exists():
        try:
            model = PeftModel.from_pretrained(base_model, str(checkpoint_dir))
            model.eval()
            print(f"[OK] Adapter loaded from checkpoint-14")
        except Exception as e2:
            print(f"[FAIL] checkpoint-14 also failed: {e2}")
            sys.exit(1)
    else:
        sys.exit(1)

# ── Step 7: Print Model Info ──────────────────────────────────────────────────
print("\n--- Step 7: Model Configuration Summary ---")
cfg = model.config if hasattr(model, "config") else base_model.config
print(f"     model type        : {getattr(cfg, 'model_type', 'unknown')}")
print(f"     hidden size       : {getattr(cfg, 'hidden_size', 'N/A')}")
print(f"     num layers        : {getattr(cfg, 'num_hidden_layers', 'N/A')}")
print(f"     num attn heads    : {getattr(cfg, 'num_attention_heads', 'N/A')}")
print(f"     max position emb  : {getattr(cfg, 'max_position_embeddings', 'N/A')}")

# ── Step 8: Test Generation ───────────────────────────────────────────────────
print("\n--- Step 8: Test Generation ---")
TEST_PROMPT = (
    "<s>[INST] You are a car diagnostic assistant. "
    "IMPORTANT: Do NOT write code. Do NOT write HTML. "
    "Only write the sections exactly as specified.\n\n"
    "Return output in this exact format (no code, no HTML):\n\n"
    "Symptom Summary:\n"
    "OBD-II Interpretation:\n"
    "Likely Causes (ranked):\n"
    "Risk Level (Low/Medium/High):\n"
    "Safe Checks (user can do):\n"
    "Do NOT Do:\n"
    "Next Action:\n\n"
    "Symptoms: My car makes a clicking sound while starting. No warning lights.\n"
    "OBD Codes: None\n"
    "OBD Notes:\nNone provided.\n"
    "[/INST]\n"
)

print(f"Prompt (first 120 chars): {TEST_PROMPT[:120]}...\n")
t0 = time.time()
try:
    inputs = tokenizer(TEST_PROMPT, return_tensors="pt").to(device)
    print(f"Input token count: {inputs['input_ids'].shape[1]}")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.5,
            top_p=0.9,
            repetition_penalty=1.25,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    elapsed = time.time() - t0
    generated_tokens = output_ids.shape[1] - inputs["input_ids"].shape[1]
    tok_per_sec = generated_tokens / elapsed

    raw_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    if "[/INST]" in raw_text:
        response = raw_text.split("[/INST]", 1)[-1]
    else:
        response = raw_text
    response = response.replace("</s>", "").replace("<s>", "").strip()

    print(f"[OK] Generation completed in {elapsed:.1f}s")
    print(f"     generated tokens : {generated_tokens}")
    print(f"     throughput       : {tok_per_sec:.1f} tok/s")
    print(f"\n--- Generated Response ---")
    print(response)
    print("--- End Response ---\n")

except Exception as e:
    print(f"[FAIL] Generation failed: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ── Final Result ──────────────────────────────────────────────────────────────
print("=" * 60)
print("✅  ALL VERIFICATION STEPS PASSED")
print("=" * 60)
print("\nReady to proceed with UI migration.")
print(f"Launch command: python app.py")
