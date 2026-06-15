"""
src/model_loader.py
===================
Tokenizer, base model, and PEFT adapter loading for Apple Silicon macOS.

Key decisions:
  - bitsandbytes 4-bit quantization is CUDA-only → NOT used here.
  - Model is loaded in torch.float32 (full precision) for MPS/CPU.
  - MPS is preferred over CPU when available (Apple Silicon).
  - Model and tokenizer are loaded once and reused across inference calls.
"""

import sys
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from src.config import (
    BASE_MODEL_NAME,
    ADAPTER_DIR,
    CHECKPOINT_DIR,
    MODEL_MAX_LENGTH,
    PADDING_SIDE,
)


# ── Device selection (Apple Silicon aware) ─────────────────────────────────────
def get_device() -> str:
    """Return 'mps', or 'cpu' as fallback. Never assumes CUDA."""
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ── Tokenizer loading ──────────────────────────────────────────────────────────
def load_tokenizer(model_name: str = BASE_MODEL_NAME):
    """
    Load the Phi-2 tokenizer with the same settings used during fine-tuning.

    The adapter directory contains tokenizer files (tokenizer.json,
    tokenizer_config.json) saved by the trainer, but we load from the
    base model name to ensure the canonical vocabulary is used, then
    apply the same overrides applied in the notebook.
    """
    print(f"[model_loader] Loading tokenizer from '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # Preserve notebook behaviour
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = PADDING_SIDE
    tokenizer.model_max_length = MODEL_MAX_LENGTH

    print(f"[model_loader] Tokenizer ready (vocab={tokenizer.vocab_size}, "
          f"max_len={tokenizer.model_max_length})")
    return tokenizer


# ── Model loading (Apple Silicon, float32, no bitsandbytes) ───────────────────
def load_base_model(model_name: str = BASE_MODEL_NAME, device: str = "cpu"):
    """
    Load Phi-2 base model in float32 without 4-bit quantization.

    bitsandbytes NF4 quantization requires CUDA and is not available on MPS.
    Phi-2 (~2.7B params) requires ~10-11 GB RAM in float32. On a MacBook Pro
    with 16+ GB unified memory this is acceptable.

    trust_remote_code=True is required by microsoft/phi-2.
    """
    print(f"[model_loader] Loading base model '{model_name}' (float32, {device})...")
    print("[model_loader] NOTE: First run downloads ~5 GB to ~/.cache/huggingface")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True,   # Required for Phi-2
    )
    model.eval()
    model.config.use_cache = False  # Consistent with training config

    print(f"[model_loader] Moving base model to {device}...")
    model = model.to(device)
    print(f"[model_loader] Base model ready on {device}")
    return model


# ── LoRA adapter loading ───────────────────────────────────────────────────────
def load_peft_model(base_model, adapter_dir: Path = ADAPTER_DIR, device: str = "cpu"):
    """
    Attach the LoRA adapter to the base model using PeftModel.

    Primary path  : models/car-assistant-qlora/
    Fallback path : models/car-assistant-qlora/checkpoint-14/

    The checkpoint-14 directory includes optimizer state which is not
    needed for inference, but the adapter weights inside it are valid.
    """
    primary = adapter_dir
    fallback = CHECKPOINT_DIR

    def _try_load(path: Path):
        print(f"[model_loader] Attaching LoRA adapter from '{path}'...")
        model = PeftModel.from_pretrained(base_model, str(path))
        model.eval()
        return model

    # Primary adapter directory
    if primary.exists() and (primary / "adapter_model.safetensors").exists():
        try:
            model = _try_load(primary)
            print("[model_loader] LoRA adapter attached successfully (primary).")
            return model
        except Exception as e:
            print(f"[model_loader] Primary adapter failed: {e}")
    else:
        print(f"[model_loader] Primary adapter not found at {primary}")

    # Fallback to checkpoint-14
    if fallback.exists() and (fallback / "adapter_model.safetensors").exists():
        print("[model_loader] Trying checkpoint-14 fallback...")
        try:
            model = _try_load(fallback)
            print("[model_loader] LoRA adapter attached (from checkpoint-14).")
            return model
        except Exception as e:
            print(f"[model_loader] Checkpoint-14 also failed: {e}")

    raise RuntimeError(
        "Failed to load LoRA adapter from either primary or checkpoint-14 directory.\n"
        f"  Primary  : {primary}\n"
        f"  Fallback : {fallback}\n"
        "Run: python scripts/test_load.py for detailed diagnostics."
    )


# ── Convenience: load everything in one call ───────────────────────────────────
def load_model_and_tokenizer():
    """
    Full initialisation sequence. Returns (model, tokenizer, device).

    Call this once at application startup (e.g. in app.py).
    The model and tokenizer should be passed into inference functions
    rather than re-loaded per request.
    """
    device    = get_device()
    tokenizer = load_tokenizer()
    base      = load_base_model(device=device)
    model     = load_peft_model(base, device=device)
    return model, tokenizer, device
