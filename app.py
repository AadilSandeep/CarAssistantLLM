"""
app.py
======
Car Assistant LLM — Application Entrypoint

Usage:
    python app.py

Loads the model, tokenizer, and OBD database exactly once,
then launches the Gradio dashboard.

Login credentials: admin / car123
"""

import sys
from pathlib import Path

# Ensure src/ is importable when running from project root
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import APP_SHARE, OBD_CSV_PATH
from src.model_loader import load_model_and_tokenizer
from src.obd_utils import load_obd_db
from src.ui import build_app


def main():
    print("=" * 60)
    print("  Car Assistant LLM — Starting Up")
    print("=" * 60)

    # ── 1. Load OBD database ──────────────────────────────────────────────────
    print("\n[app] Loading OBD database...")
    obd_db = load_obd_db(OBD_CSV_PATH)
    print(f"[app] OBD database ready: {len(obd_db)} codes loaded.\n")

    # ── 2. Load model + tokenizer (once) ─────────────────────────────────────
    print("[app] Loading model and tokenizer (this may take a few minutes on first run)...")
    model, tokenizer, device = load_model_and_tokenizer()
    print(f"\n[app] Model ready on device: {device}\n")

    # ── 3. Build and launch Gradio app ────────────────────────────────────────
    print("[app] Building Gradio interface...")
    app = build_app(model, tokenizer, obd_db)

    print("[app] Launching Gradio...")
    print("[app] Open http://127.0.0.1:7860 in your browser.")
    print("[app] Login: admin / car123\n")

    app.launch(
        share=APP_SHARE,
        debug=False,
        server_name="127.0.0.1",
        server_port=7860,
    )


if __name__ == "__main__":
    main()
