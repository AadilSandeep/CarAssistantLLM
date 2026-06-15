import os
import sys
import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from pathlib import Path

# Add src to path to use current project modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.diagnosis import diagnose as current_diagnose
from src.obd_utils import load_obd_db, normalize_obd_codes, explain_obd
from src.config import OBD_CSV_PATH, ADAPTER_DIR, BASE_MODEL_NAME

# ======================================================================
# 1. ORIGINAL NOTEBOOK LOGIC (Copied directly from notebook cell 8)
# ======================================================================

REQUIRED_HEADINGS = [
    "Symptom Summary:",
    "OBD-II Interpretation:",
    "Likely Causes",
    "Risk Level",
    "Safe Checks",
    "Do NOT Do",
    "Next Action"
]

def looks_valid(text: str) -> bool:
    t = text.strip()
    return sum(h.lower() in t.lower() for h in REQUIRED_HEADINGS) >= 4

SYSTEM_FORMAT = """Return output in this exact format (no code, no HTML):

Symptom Summary:
OBD-II Interpretation:
Likely Causes (ranked):
Risk Level (Low/Medium/High):
Safe Checks (user can do):
Do NOT Do:
Next Action:
"""

OBD_HINTS = {
    "P0300": "Random/Multiple Cylinder Misfire Detected",
    "P0301": "Cylinder 1 Misfire Detected",
    "P0171": "System Too Lean (Bank 1)",
    "P0420": "Catalyst System Efficiency Below Threshold (Bank 1)",
    "P0128": "Coolant Thermostat (Coolant Temp Below Regulating Temp)",
    "P0455": "EVAP System Large Leak Detected",
    "P0700": "Transmission Control System Malfunction",
}

def notebook_baseline_fallback(symptoms: str, obd_text: str):
    codes = normalize_obd_codes(obd_text or "")
    codes_str = ", ".join(codes) if codes else "None"
    
    local_obd = dict(OBD_HINTS)
    local_obd.update({
        "P0302": "Cylinder 2 Misfire Detected",
        "P0303": "Cylinder 3 Misfire Detected",
        "P0304": "Cylinder 4 Misfire Detected",
        "P0351": "Ignition Coil A Primary/Secondary Circuit",
        "P0352": "Ignition Coil B Primary/Secondary Circuit",
        "P0117": "Engine Coolant Temp Circuit Low Input",
        "P0118": "Engine Coolant Temp Circuit High Input",
    })
    def explain_obd2(codes):
        if not codes:
            return "None provided."
        return "\n".join([f"{c}: {local_obd.get(c, 'Unknown/General OBD-II code (lookup needed)')}" for c in codes])
    
    obd_info = explain_obd2(codes)
    s = (symptoms or "").lower()
    
    high_risk_terms = ["brake", "overheat", "steam", "smoke", "fuel smell", "petrol smell", "raw gas", "burning smell"]
    risk = "High" if any(t in s for t in high_risk_terms) else "Medium"
    
    causes = []
    safe_checks = []
    do_not = []
    next_action = []
    
    if ("fuel smell" in s or "petrol smell" in s or "raw gas" in s) and ("shake" in s or "shaking" in s or "rough" in s):
        causes = [
            "Misfire causing unburnt fuel (spark plug/ignition coil/injector)",
            "Fuel leak in engine bay or EVAP system leak",
            "Air-fuel imbalance (MAF/MAP sensor, vacuum leak)"
        ]
        safe_checks = [
            "If smell is strong, turn off engine and check for visible leaks under hood/near fuel lines (no flames).",
            "Check if Check Engine Light is flashing (flashing = stop driving).",
            "Note whether shaking happens at idle only or also while driving."
        ]
        do_not = [
            "Do not continue driving if fuel smell is strong or if the engine is shaking heavily.",
            "Do not smoke or use open flame near the vehicle."
        ]
        next_action = [
            "Tow to a service center if fuel smell is strong.",
            "Ask for misfire diagnosis (spark plugs, coils, injectors) and fuel leak inspection."
        ]
        risk = "High"
    elif ("click" in s or "clicking" in s) and ("start" in s or "starting" in s or "crank" in s):
        causes = [
            "Weak battery or loose/corroded terminals",
            "Starter relay/solenoid issue",
            "Starter motor issue"
        ]
        safe_checks = [
            "Check battery terminals for looseness/corrosion (engine OFF).",
            "Observe if headlights dim a lot during crank attempt.",
            "Try jump start if available (follow safe procedure)."
        ]
        do_not = [
            "Do not crank repeatedly for long durations.",
            "Do not ignore burning smell or smoke."
        ]
        next_action = [
            "Get battery + starter + charging system tested.",
            "If it starts with jump, battery/charging is likely."
        ]
    elif ("overheat" in s or "temperature" in s or "temp" in s or "steam" in s):
        causes = [
            "Low coolant / coolant leak",
            "Thermostat stuck",
            "Radiator fan not working",
            "Water pump issue"
        ]
        safe_checks = [
            "Stop driving if gauge goes to red; let engine cool.",
            "Check coolant level only when engine is cool.",
            "Look for coolant leaks under the car."
        ]
        do_not = [
            "Do not open radiator cap when hot.",
            "Do not continue driving while overheating."
        ]
        next_action = [
            "Cooling system inspection recommended immediately.",
            "Tow if overheating persists."
        ]
        risk = "High"
    elif "brake" in s or "braking" in s or "grinding" in s or "soft pedal" in s:
        causes = [
            "Worn brake pads/rotors",
            "Brake fluid leak / air in brake lines",
            "Stuck caliper"
        ]
        safe_checks = [
            "Check brake fluid level if accessible.",
            "Test braking gently at low speed in a safe area."
        ]
        do_not = [
            "Do not drive at high speed.",
            "Do not ignore grinding noises."
        ]
        next_action = [
            "Brake inspection urgently recommended."
        ]
        risk = "High"
    elif any(c.startswith("P03") for c in codes) or ("misfire" in s or "rough idle" in s or "shaking" in s):
        causes = [
            "Spark plug worn/fouled",
            "Ignition coil issue",
            "Fuel injector issue",
            "Vacuum leak / MAF sensor issue"
        ]
        safe_checks = [
            "If Check Engine Light is flashing, avoid driving.",
            "Note if issue worsens under acceleration.",
            "Check for loose intake hose if visible."
        ]
        do_not = [
            "Do not keep driving long distances with severe misfire."
        ]
        next_action = [
            "Request misfire diagnosis (plugs/coils/injectors) and read freeze-frame data."
        ]
        risk = "High" if "flashing" in s else risk
    else:
        causes = [
            "General engine/electrical issue",
            "Sensor or maintenance-related fault"
        ]
        safe_checks = [
            "Note when the symptom occurs (startup/idle/acceleration).",
            "Provide vehicle model/year and any warning lights.",
            "Scan OBD-II codes if possible."
        ]
        do_not = [
            "Do not ignore persistent warning lights."
        ]
        next_action = [
            "If persistent, visit a technician with the symptom notes."
        ]
    
    causes_fmt = "\n".join([f"{i+1}) {c}" for i, c in enumerate(causes)])
    checks_fmt = "\n".join([f"- {c}" for c in safe_checks])
    donot_fmt = "\n".join([f"- {d}" for d in do_not])
    next_fmt = "\n".join([f"- {n}" for n in next_action])
    
    return f"""Symptom Summary:
- {symptoms.strip()}

OBD-II Interpretation:
- Codes: {codes_str}
{obd_info}

Likely Causes (ranked):
{causes_fmt}

Risk Level (Low/Medium/High):
- {risk}

Safe Checks (user can do):
{checks_fmt}

Do NOT Do:
{donot_fmt}

Next Action:
{next_fmt}
"""

def notebook_generate_once(symptoms: str, obd_text: str, temperature: float, model, tokenizer, bad_words_ids):
    codes = normalize_obd_codes(obd_text or "")
    codes_str = ", ".join(codes) if codes else "None"
    # Note: notebook explain_obd relies on global OBD_HINTS which we replicated above
    # Wait, the notebook's explain_obd uses a global dict. I will use the one above.
    def nb_explain_obd(codes):
        if not codes:
            return "None provided."
        lines = []
        for c in codes:
            lines.append(f"{c}: {OBD_HINTS.get(c, 'Unknown/General OBD-II code (lookup needed)')}")
        return "\n".join(lines)
    obd_info = nb_explain_obd(codes)

    prompt = (
        f"<s>[INST] You are a car diagnostic assistant. "
        f"IMPORTANT: Do NOT write code. Do NOT write HTML. "
        f"Only write the sections exactly as specified.\n\n"
        f"{SYSTEM_FORMAT}\n"
        f"Symptoms: {symptoms.strip()}\n"
        f"OBD Codes: {codes_str}\n"
        f"OBD Notes:\n{obd_info}\n"
        f"[/INST]\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            repetition_penalty=1.25,
            bad_words_ids=bad_words_ids,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    text = tokenizer.decode(out_ids[0], skip_special_tokens=False)
    if "[/INST]" in text:
        text = text.split("[/INST]", 1)[-1]
    return text.replace("</s>", "").replace("<s>", "").strip()

def notebook_diagnose(symptoms: str, obd_text: str, model, tokenizer, bad_words_ids):
    symptoms = (symptoms or "").strip()
    if not symptoms:
        return "Please enter symptoms (e.g., clicking sound when starting, warning light, vibration, smell)."

    out1 = notebook_generate_once(symptoms, obd_text, 0.5, model, tokenizer, bad_words_ids)
    if looks_valid(out1):
        return out1

    out2 = notebook_generate_once(symptoms, obd_text, 0.2, model, tokenizer, bad_words_ids)
    if looks_valid(out2):
        return out2

    return notebook_baseline_fallback(symptoms, obd_text)

# ======================================================================
# 2. RUN COMPARISON
# ======================================================================

def main():
    print("Loading models...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Use torch.float16 as an approximation of the notebook's bnb_4bit_compute_dtype
    # The current codebase uses torch.bfloat16. We will see if there is a difference.
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    # Notebook didn't explicitly use model_max_length during generation, but did during training
    
    # Compute notebook bad words
    bad_words = ["def ", "class ", "import ", "argparse", "parser =", "raw_input", "print(", "super().__init__", "pydantic"]
    bad_words_ids = [tokenizer(bw, add_special_tokens=False).input_ids for bw in bad_words if len(tokenizer(bw, add_special_tokens=False).input_ids) > 0]
    
    # Load model
    print(f"Loading base model to {device} in float16...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    base_model.eval()
    base_model.config.use_cache = False
    base_model = base_model.to(device)
    
    print("Loading PEFT adapter...")
    model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))
    model.eval()
    
    # Load obd_db for the current app
    obd_db = load_obd_db(OBD_CSV_PATH)
    from src.diagnosis import init_bad_words
    init_bad_words(tokenizer)
    
    test_cases = [
        ("Clicking sound when starting, no warning lights", "P0300"),
        ("Strong raw gas smell and heavy engine shaking", "P0302")
    ]
    
    for symptoms, obd in test_cases:
        print("\n" + "="*80)
        print(f"TESTING: {symptoms} | {obd}")
        
        # We need a static seed for deterministic comparison
        torch.manual_seed(42)
        print("\n--- ORIGINAL NOTEBOOK OUTPUT ---")
        nb_out = notebook_diagnose(symptoms, obd, model, tokenizer, bad_words_ids)
        print(nb_out)
        
        torch.manual_seed(42)
        print("\n--- CURRENT APP OUTPUT ---")
        # For the current app, we pass the same model and tokenizer
        # In practice, current app runs on bfloat16 and we're passing float16 here, 
        # but this is just testing the LOGIC equality.
        app_out = current_diagnose(symptoms, obd, model, tokenizer, obd_db)
        print(app_out)
        
        if nb_out == app_out:
            print("\n✅ PERFECT MATCH")
        else:
            print("\n❌ DIFFERENCE DETECTED")

if __name__ == "__main__":
    main()
