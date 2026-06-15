import sys
import time
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import ADAPTER_DIR, BASE_MODEL_NAME

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "mps" else torch.float32

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"Loading base model ({dtype})...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    base_model.eval()

    print("Attaching adapter...")
    model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))
    model.eval()

    symptoms = "clicking sound while starting"

    # Prompt A: Current notebook inference prompt
    SYSTEM_FORMAT = (
        "Return output in this exact format (no code, no HTML):\n\n"
        "Symptom Summary:\n"
        "OBD-II Interpretation:\n"
        "Likely Causes (ranked):\n"
        "Risk Level (Low/Medium/High):\n"
        "Safe Checks (user can do):\n"
        "Do NOT Do:\n"
        "Next Action:\n"
    )
    prompt_a = (
        f"<s>[INST] You are a car diagnostic assistant. "
        f"IMPORTANT: Do NOT write code. Do NOT write HTML. "
        f"Only write the sections exactly as specified.\n\n"
        f"{SYSTEM_FORMAT}\n"
        f"Symptoms: {symptoms}\n"
        f"OBD Codes: None\n"
        f"OBD Notes:\nNone provided.\n"
        f"[/INST]\n"
    )

    # Prompt B: Training-style prompt (minimal)
    prompt_b = f"<s>[INST] {symptoms} [/INST]\nSymptom:"

    # Prompt C: Training-style prompt (exact match with vehicle info boilerplate)
    prompt_c = f"<s>[INST] {symptoms} (100000 km, 5 years old, Petrol fuel, Automatic transmission, 0 issues reported, history: Good)\nExtra Info: No OBD code [/INST]\nSymptom:"

    def generate_response(prompt_text, name):
        print(f"\n{'='*40}")
        print(f"Testing Prompt {name}")
        print(f"{'='*40}")
        print(f"Prompt content:\n{prompt_text}\n")
        
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        t0 = time.time()
        
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.5,
                top_p=0.9,
                repetition_penalty=1.25,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
        elapsed = time.time() - t0
        raw_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
        
        if "[/INST]" in raw_text:
            response = raw_text.split("[/INST]", 1)[-1]
        else:
            response = raw_text
            
        # Clean up tags for readability
        response = response.replace("</s>", "").replace("<s>", "")
        
        print(f"Response ({elapsed:.1f}s):\n{response.strip()}")
        print(f"\n--- End Prompt {name} ---\n")

    generate_response(prompt_a, "A (Notebook Inference)")
    generate_response(prompt_b, "B (Training Style Minimal)")
    generate_response(prompt_c, "C (Training Style Exact)")

if __name__ == "__main__":
    main()
