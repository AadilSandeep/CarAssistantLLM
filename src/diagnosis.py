"""
src/diagnosis.py
================
Core inference engine: prompt construction, generation, validation, fallback.

All logic is extracted directly from notebook Cell 8:
  - SYSTEM_FORMAT prompt template (preserved verbatim)
  - generate_once() with exact generation parameters
  - looks_valid() heading-based validation
  - baseline_fallback() rule-based fallback
  - diagnose() orchestrator with two-attempt retry

The model and tokenizer are injected (not imported as globals) so this
module has no startup side-effects.
"""

import torch
from src.config import (
    REQUIRED_HEADINGS,
    MIN_VALID_HEADINGS,
    GEN_MAX_NEW_TOKENS,
    GEN_TEMPERATURE_FIRST,
    GEN_TEMPERATURE_RETRY,
    GEN_TOP_P,
    GEN_REPETITION_PENALTY,
    OBD_HINTS,
)
from src.obd_utils import normalize_obd_codes, explain_obd

# ── Prompt template (from notebook Cell 8, preserved verbatim) ─────────────────
SYSTEM_FORMAT = """Return output in this exact format (no code, no HTML):

Symptom Summary:
OBD-II Interpretation:
Likely Causes (ranked):
Risk Level (Low/Medium/High):
Safe Checks (user can do):
Do NOT Do:
Next Action:
"""

# ── bad_words that suppress code/HTML output ──────────────────────────────────
# Computed once using the tokenizer at app startup; stored here.
_bad_words_ids: list[list[int]] = []

BAD_WORD_STRINGS = [
    "def ", "class ", "import ", "argparse", "parser =",
    "raw_input", "print(", "super().__init__", "pydantic",
]


def init_bad_words(tokenizer) -> list[list[int]]:
    """
    Compute bad_words_ids from the tokenizer. Call once at startup.
    Stores result in module-level _bad_words_ids for subsequent calls.
    """
    global _bad_words_ids
    result = []
    for bw in BAD_WORD_STRINGS:
        ids = tokenizer(bw, add_special_tokens=False).input_ids
        if len(ids) > 0:
            result.append(ids)
    _bad_words_ids = result
    return _bad_words_ids


# ── Validity check (from notebook Cell 8) ─────────────────────────────────────
def looks_valid(text: str) -> bool:
    """
    Return True if the generated text contains at least MIN_VALID_HEADINGS
    of the required section headings. Matches are case-insensitive.
    """
    t = text.strip()
    return sum(h.lower() in t.lower() for h in REQUIRED_HEADINGS) >= MIN_VALID_HEADINGS


# ── Single generation attempt (from notebook Cell 8) ──────────────────────────
def generate_once(
    symptoms: str,
    obd_text: str,
    temperature: float,
    model,
    tokenizer,
    obd_db: dict | None = None,
) -> str:
    """
    Build the prompt, tokenize, generate, decode, and return the assistant
    response text. Uses the exact prompt format from the notebook.
    """
    codes = normalize_obd_codes(obd_text or "")
    codes_str = ", ".join(codes) if codes else "None"
    obd_info = explain_obd(codes, db=obd_db)

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

    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=GEN_MAX_NEW_TOKENS,
            do_sample=True,
            temperature=temperature,
            top_p=GEN_TOP_P,
            repetition_penalty=GEN_REPETITION_PENALTY,
            bad_words_ids=_bad_words_ids if _bad_words_ids else None,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(out_ids[0], skip_special_tokens=False)
    if "[/INST]" in text:
        text = text.split("[/INST]", 1)[-1]
    return text.replace("</s>", "").replace("<s>", "").strip()


# ── Rule-based fallback (from notebook Cell 8, preserved verbatim) ─────────────
def baseline_fallback(symptoms: str, obd_text: str, obd_db: dict | None = None) -> str:
    """
    Deterministic rule-based response when LLM output does not pass validation.
    Covers the most common symptom categories with pre-written structured output.
    """
    codes = normalize_obd_codes(obd_text or "")
    codes_str = ", ".join(codes) if codes else "None"

    # Expand OBD hints with extras
    local_obd = dict(OBD_HINTS)
    if obd_db:
        local_obd.update(obd_db)

    def explain_obd2(codes):
        if not codes:
            return "None provided."
        return "\n".join(
            [f"{c}: {local_obd.get(c, 'Unknown/General OBD-II code (lookup needed)')}" for c in codes]
        )

    obd_info = explain_obd2(codes)
    s = (symptoms or "").lower()

    # Risk heuristic
    high_risk_terms = ["brake", "overheat", "steam", "smoke", "fuel smell", "petrol smell", "raw gas", "burning smell"]
    risk = "High" if any(t in s for t in high_risk_terms) else "Medium"

    causes = []
    safe_checks = []
    do_not = []
    next_action = []

    # Fuel smell + shaking → misfire + fuel leak
    if (("fuel smell" in s or "petrol smell" in s or "raw gas" in s)
            and ("shake" in s or "shaking" in s or "rough" in s)):
        causes = [
            "Misfire causing unburnt fuel (spark plug/ignition coil/injector)",
            "Fuel leak in engine bay or EVAP system leak",
            "Air-fuel imbalance (MAF/MAP sensor, vacuum leak)",
        ]
        safe_checks = [
            "If smell is strong, turn off engine and check for visible leaks under hood/near fuel lines (no flames).",
            "Check if Check Engine Light is flashing (flashing = stop driving).",
            "Note whether shaking happens at idle only or also while driving.",
        ]
        do_not = [
            "Do not continue driving if fuel smell is strong or if the engine is shaking heavily.",
            "Do not smoke or use open flame near the vehicle.",
        ]
        next_action = [
            "Tow to a service center if fuel smell is strong.",
            "Ask for misfire diagnosis (spark plugs, coils, injectors) and fuel leak inspection.",
        ]
        risk = "High"

    # Clicking on start → battery/starter
    elif ("click" in s or "clicking" in s) and ("start" in s or "starting" in s or "crank" in s):
        causes = [
            "Weak battery or loose/corroded terminals",
            "Starter relay/solenoid issue",
            "Starter motor issue",
        ]
        safe_checks = [
            "Check battery terminals for looseness/corrosion (engine OFF).",
            "Observe if headlights dim a lot during crank attempt.",
            "Try jump start if available (follow safe procedure).",
        ]
        do_not = [
            "Do not crank repeatedly for long durations.",
            "Do not ignore burning smell or smoke.",
        ]
        next_action = [
            "Get battery + starter + charging system tested.",
            "If it starts with jump, battery/charging is likely.",
        ]

    # Overheating → cooling system
    elif ("overheat" in s or "temperature" in s or "temp" in s or "steam" in s):
        causes = [
            "Low coolant / coolant leak",
            "Thermostat stuck",
            "Radiator fan not working",
            "Water pump issue",
        ]
        safe_checks = [
            "Stop driving if gauge goes to red; let engine cool.",
            "Check coolant level only when engine is cool.",
            "Look for coolant leaks under the car.",
        ]
        do_not = [
            "Do not open radiator cap when hot.",
            "Do not continue driving while overheating.",
        ]
        next_action = [
            "Cooling system inspection recommended immediately.",
            "Tow if overheating persists.",
        ]
        risk = "High"

    # Brakes → braking system
    elif "brake" in s or "braking" in s or "grinding" in s or "soft pedal" in s:
        causes = [
            "Worn brake pads/rotors",
            "Brake fluid leak / air in brake lines",
            "Stuck caliper",
        ]
        safe_checks = [
            "Check brake fluid level if accessible.",
            "Test braking gently at low speed in a safe area.",
        ]
        do_not = [
            "Do not drive at high speed.",
            "Do not ignore grinding noises.",
        ]
        next_action = [
            "Brake inspection urgently recommended.",
        ]
        risk = "High"

    # Misfire codes → engine misfire
    elif (any(c.startswith("P03") for c in codes)
          or ("misfire" in s or "rough idle" in s or "shaking" in s)):
        causes = [
            "Spark plug worn/fouled",
            "Ignition coil issue",
            "Fuel injector issue",
            "Vacuum leak / MAF sensor issue",
        ]
        safe_checks = [
            "If Check Engine Light is flashing, avoid driving.",
            "Note if issue worsens under acceleration.",
            "Check for loose intake hose if visible.",
        ]
        do_not = [
            "Do not keep driving long distances with severe misfire.",
        ]
        next_action = [
            "Request misfire diagnosis (plugs/coils/injectors) and read freeze-frame data.",
        ]
        risk = "High" if "flashing" in s else risk

    # Default generic
    else:
        causes = [
            "General engine/electrical issue",
            "Sensor or maintenance-related fault",
        ]
        safe_checks = [
            "Note when the symptom occurs (startup/idle/acceleration).",
            "Provide vehicle model/year and any warning lights.",
            "Scan OBD-II codes if possible.",
        ]
        do_not = [
            "Do not ignore persistent warning lights.",
        ]
        next_action = [
            "If persistent, visit a technician with the symptom notes.",
        ]

    # Format the structured output
    causes_fmt = "\n".join([f"{i+1}) {c}" for i, c in enumerate(causes)])
    checks_fmt = "\n".join([f"- {c}" for c in safe_checks])
    donot_fmt  = "\n".join([f"- {d}" for d in do_not])
    next_fmt   = "\n".join([f"- {n}" for n in next_action])

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


# ── Main diagnose() orchestrator (from notebook Cell 8) ────────────────────────
def diagnose(
    symptoms: str,
    obd_text: str,
    model,
    tokenizer,
    obd_db: dict | None = None,
) -> str:
    """
    Orchestrate LLM diagnosis with two-attempt retry, then rule-based fallback.

    Attempt 1: temperature=0.5 (more creative)
    Attempt 2: temperature=0.2 (more deterministic, if attempt 1 invalid)
    Fallback:  baseline_fallback() (rule-based, always valid)
    """
    symptoms = (symptoms or "").strip()
    if not symptoms:
        return "Please enter symptoms (e.g., clicking sound when starting, warning light, vibration, smell)."

    # Attempt 1
    out1 = generate_once(symptoms, obd_text, GEN_TEMPERATURE_FIRST, model, tokenizer, obd_db)
    if looks_valid(out1):
        return out1

    # Attempt 2 (lower temperature)
    out2 = generate_once(symptoms, obd_text, GEN_TEMPERATURE_RETRY, model, tokenizer, obd_db)
    if looks_valid(out2):
        return out2

    # Fallback
    return baseline_fallback(symptoms, obd_text, obd_db)
