import sys
import time
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ADAPTER_DIR = PROJECT_ROOT / "models" / "car-assistant-qlora"
BASE_MODEL  = "microsoft/phi-2"

def test_dtype(dtype, device="mps"):
    print(f"\n=========================================")
    print(f"Testing dtype: {dtype} on {device}")
    print(f"=========================================")
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = 384
    
    t0 = time.time()
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        base_model.eval()
        base_model = base_model.to(device)
        model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))
        model.eval()
        elapsed_load = time.time() - t0
        print(f"Load time: {elapsed_load:.1f}s")
        
        prompt = "<s>[INST] Car cranks slowly and I hear rapid clicking during start [/INST]\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        t1 = time.time()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.5,
                top_p=0.9,
                repetition_penalty=1.25,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        elapsed_gen = time.time() - t1
        
        generated_tokens = output_ids.shape[1] - inputs["input_ids"].shape[1]
        tok_per_sec = generated_tokens / elapsed_gen
        
        raw_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
        print(f"Generation throughput: {tok_per_sec:.1f} tok/s")
        print(f"Response:\n{raw_text}")
        
        del model
        del base_model
        import gc
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
            
        return True
        
    except Exception as e:
        print(f"Failed: {e}")
        return False

if __name__ == "__main__":
    if torch.backends.mps.is_available():
        test_dtype(torch.float16)
        test_dtype(torch.bfloat16)
    else:
        print("MPS not available")
