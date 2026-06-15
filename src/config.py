"""
src/config.py
=============
Central configuration: paths, constants, and generation parameters.

All path references use pathlib so the project works from any working directory.
No Google Colab or /content/ paths exist here.
"""

from pathlib import Path

# ── Project layout ─────────────────────────────────────────────────────────────
# Resolves to the project root regardless of where python app.py is called from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data
DATA_DIR         = PROJECT_ROOT / "data"
OBD_CSV_PATH     = DATA_DIR / "obd-trouble-codes.csv"
TRAIN_JSONL_PATH = DATA_DIR / "train.jsonl"

# Model artifacts
MODELS_DIR   = PROJECT_ROOT / "models"
ADAPTER_DIR  = MODELS_DIR / "car-assistant-qlora"
CHECKPOINT_DIR = ADAPTER_DIR / "checkpoint-14"   # fallback only

# Assets
ASSETS_DIR   = PROJECT_ROOT / "assets"
ICONS_DIR    = ASSETS_DIR / "icons"

# ── Model configuration ────────────────────────────────────────────────────────
BASE_MODEL_NAME = "microsoft/phi-2"

# Tokenizer settings (match training configuration from notebook)
MODEL_MAX_LENGTH = 384
PADDING_SIDE     = "right"

# ── Generation parameters (preserved exactly from notebook inference cells) ────
# First attempt — slightly higher temperature for diversity
GEN_MAX_NEW_TOKENS    = 200
GEN_TEMPERATURE_FIRST = 0.5
GEN_TEMPERATURE_RETRY = 0.2
GEN_TOP_P             = 0.9
GEN_REPETITION_PENALTY = 1.25

# Headings that must appear in a valid response (4+ required)
REQUIRED_HEADINGS = [
    "Symptom Summary:",
    "OBD-II Interpretation:",
    "Likely Causes",
    "Risk Level",
    "Safe Checks",
    "Do NOT Do",
    "Next Action",
]
MIN_VALID_HEADINGS = 4  # at least this many must be present

# ── OBD inline fallback hints (from notebook Cell 6) ──────────────────────────
# Used when the CSV is unavailable or a code is not in the CSV.
OBD_HINTS: dict[str, str] = {
    "P0300": "Random/Multiple Cylinder Misfire Detected",
    "P0301": "Cylinder 1 Misfire Detected",
    "P0302": "Cylinder 2 Misfire Detected",
    "P0303": "Cylinder 3 Misfire Detected",
    "P0304": "Cylinder 4 Misfire Detected",
    "P0171": "System Too Lean (Bank 1)",
    "P0420": "Catalyst System Efficiency Below Threshold (Bank 1)",
    "P0128": "Coolant Thermostat (Coolant Temp Below Regulating Temp)",
    "P0455": "EVAP System Large Leak Detected",
    "P0700": "Transmission Control System Malfunction",
    "P0351": "Ignition Coil A Primary/Secondary Circuit",
    "P0352": "Ignition Coil B Primary/Secondary Circuit",
    "P0117": "Engine Coolant Temp Circuit Low Input",
    "P0118": "Engine Coolant Temp Circuit High Input",
}

# ── Authentication (from notebook Cell 10) ─────────────────────────────────────
VALID_USERS: dict[str, str] = {
    "admin": "car123",
}

# ── Gradio UI settings ─────────────────────────────────────────────────────────
APP_TITLE  = "Car Assistant LLM — Dashboard"
APP_SHARE  = False   # Set True only if you need a public Gradio share link

# ── Warning light definitions (from notebook Cell 10) ─────────────────────────
# Tuple: (icon_filename, display_title, explanation, emoji, accent_color_hex)
WARNING_LIGHTS = [
    (
        "engine_light.png",
        "Check Engine (MIL)",
        "Engine/emissions fault. If flashing: avoid driving; scan OBD codes immediately.",
        "🔧",
        "#f59e0b",
    ),
    (
        "oil_pressure.png",
        "Oil Pressure",
        "STOP engine ASAP. Low oil pressure can cause severe engine damage.",
        "🛢",
        "#ef4444",
    ),
    (
        "battery_changing.png",
        "Battery / Charging",
        "Charging system fault. Vehicle may stall. Check alternator and battery.",
        "🔋",
        "#3b82f6",
    ),
    (
        "coolant_temp.png",
        "Coolant Temperature",
        "Overheating risk. Stop safely, let engine cool, and check coolant when safe.",
        "🌡",
        "#ef4444",
    ),
    (
        "abs.png",
        "ABS",
        "ABS may be disabled. Normal brakes work, but reduced control; service soon.",
        "🛑",
        "#f97316",
    ),
    (
        "airbag.png",
        "Airbag / SRS",
        "Airbag system fault. Safety risk; service recommended.",
        "💥",
        "#8b5cf6",
    ),
]

# ── Common problems quick reference (from notebook Cell 10) ───────────────────
COMMON_PROBLEMS = [
    ("Clicking sound on start",  "Weak battery / loose terminals / starter relay issue"),
    ("Engine overheating",       "Low coolant / radiator fan / thermostat / water pump issue"),
    ("Grinding brakes",          "Worn brake pads/rotors — inspect urgently"),
    ("Vibration at high speed",  "Wheel balancing / alignment / worn tires"),
    ("Poor mileage",             "Low tire pressure / dirty air filter / O2 sensor / driving style"),
]

# ── Gradio example inputs (from notebook Cell 9) ───────────────────────────────
GRADIO_EXAMPLES = [
    ["Clicking sound when starting, no warning lights",                         "P0300"],
    ["Strong raw gas smell and heavy engine shaking",                           "P0302"],
    ["Engine overheating, temperature gauge goes high, steam near hood",        "P0128"],
    ["Grinding noise while braking, brake pedal feels soft",                    ""],
]
