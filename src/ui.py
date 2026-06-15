"""
src/ui.py
=========
Gradio dashboard UI — extracted and migrated from notebook Cell 10.

Changes from the original:
  - Google Drive paths replaced with pathlib local paths (via config.py)
  - share=True changed to share=False (local-only by default)
  - diagnose() receives model/tokenizer/obd_db as closures instead of globals
  - ICONS_DIR, OBD_CSV_PATH come from config.py
  - All app logic is wrapped in build_app() so app.py can call it cleanly

Everything else is preserved verbatim from the notebook.
"""

import os
import textwrap
import tempfile
import gradio as gr

from src.config import (
    APP_TITLE,
    APP_SHARE,
    ICONS_DIR,
    WARNING_LIGHTS,
    COMMON_PROBLEMS,
    GRADIO_EXAMPLES,
    VALID_USERS,
)
from src.obd_utils import obd_lookup, obd_search, obd_list_text


# ── Authentication (from notebook Cell 10, preserved verbatim) ─────────────────
def do_login(username, password):
    u = (username or "").strip()
    p = (password or "").strip()
    if u in VALID_USERS and VALID_USERS[u] == p:
        return (
            gr.update(visible=False),   # hide login page
            gr.update(visible=True),    # show main app
            "",                         # clear login message
            u,                          # save logged-in user
        )
    return (
        gr.update(visible=True),
        gr.update(visible=False),
        "❌ Invalid username or password.",
        "",
    )


def do_logout():
    return (
        gr.update(visible=True),    # show login page
        gr.update(visible=False),   # hide main app
        "",                         # clear header
        "",                         # clear user state
    )


def render_header(u):
    if not u:
        return ""
    return f"""
    <div class="topbar">
        <div class="brand">🚗 Car Assistant LLM — Dashboard</div>
        <div class="userchip">✅ Logged in as: <b>{u}</b></div>
    </div>
    """


# ── Warning light gallery helpers (from notebook Cell 10) ─────────────────────
_SVG_CACHE: dict = {}


def _make_svg(emoji: str, color: str, title: str) -> str:
    """Generate a coloured SVG placeholder and cache the temp file path."""
    key = (emoji, color, title)
    if key in _SVG_CACHE:
        return _SVG_CACHE[key]
    svg = textwrap.dedent(f"""\
        <svg xmlns="http://www.w3.org/2000/svg" width="160" height="160" viewBox="0 0 160 160">
          <rect width="160" height="160" rx="20" fill="{color}" opacity="0.12"/>
          <rect x="3" y="3" width="154" height="154" rx="18"
                fill="none" stroke="{color}" stroke-width="3"/>
          <text x="80" y="82" font-size="68" text-anchor="middle"
                dominant-baseline="middle">{emoji}</text>
          <text x="80" y="140" font-size="12" text-anchor="middle"
                fill="{color}" font-family="sans-serif"
                font-weight="bold">{title[:16]}</text>
        </svg>""")
    tmp = tempfile.NamedTemporaryFile(suffix=".svg", delete=False)
    tmp.write(svg.encode())
    tmp.close()
    _SVG_CACHE[key] = tmp.name
    return tmp.name


def build_warning_items() -> list[tuple[str, str]]:
    """Always returns all warning light items — real PNG or SVG placeholder."""
    items = []
    for fname, title, _desc, emoji, color in WARNING_LIGHTS:
        real = str(ICONS_DIR / fname)
        path = real if os.path.exists(real) else _make_svg(emoji, color, title)
        items.append((path, title))
    return items


def warning_explain(evt: gr.SelectData) -> str:
    """Index maps directly to WARNING_LIGHTS — no filtering."""
    idx = evt.index
    if idx is None or idx >= len(WARNING_LIGHTS):
        return "Select an icon to see details."
    _, title, desc, _emoji, _color = WARNING_LIGHTS[idx]
    return f"### {title}\n\n{desc}"


# ── Common problems text (from notebook Cell 10) ───────────────────────────────
def common_text() -> str:
    return "\n\n".join([f"• {title}\n  - {desc}" for title, desc in COMMON_PROBLEMS])


# ── Navigation (from notebook Cell 10) ────────────────────────────────────────
def show_page(page: str):
    return (
        gr.update(visible=(page == "assistant")),
        gr.update(visible=(page == "obd")),
        gr.update(visible=(page == "lights")),
        gr.update(visible=(page == "common")),
    )


# ── Custom CSS (from notebook Cell 10, preserved verbatim) ────────────────────
CSS = """
.topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 14px 16px;
    border-bottom: 1px solid #e6e6e6;
    background: #ffffff;
}
.brand {
    font-size: 22px;
    font-weight: 800;
}
.userchip {
    font-size: 13px;
    padding: 8px 12px;
    border: 1px solid #e6e6e6;
    border-radius: 999px;
    background: #fafafa;
}
.sidebar-btn button {
    width: 100% !important;
    justify-content: flex-start !important;
    font-weight: 700 !important;
}
#gallery_full {
    max-height: 620px !important;
    overflow-y: auto !important;
}
#gallery_full .grid-wrap {
    gap: 12px !important;
}
#gallery_full img {
    border-radius: 12px !important;
    object-fit: contain !important;
    max-height: 180px !important;
}
"""


# ── App builder ────────────────────────────────────────────────────────────────
def build_app(model, tokenizer, obd_db: dict) -> gr.Blocks:
    """
    Construct and return the full Gradio Blocks app.

    model, tokenizer, obd_db are passed in from app.py so they are loaded
    only once and shared across all inference calls.
    """
    from src.diagnosis import diagnose, init_bad_words

    # Pre-compute bad_words_ids once
    init_bad_words(tokenizer)

    # Thin wrapper so Gradio can call diagnose without passing model/tokenizer
    def safe_diagnose(symptoms: str, obd_text: str) -> str:
        return diagnose(symptoms, obd_text, model, tokenizer, obd_db)

    # OBD search/lookup wrappers (db captured in closure)
    def _obd_search(query, db):
        return obd_search(query, db)

    def _obd_lookup(code, db):
        return obd_lookup(code, db)

    theme = gr.themes.Soft()

    with gr.Blocks(title=APP_TITLE, theme=theme, css=CSS) as app:
        state_user   = gr.State("")
        state_obd_db = gr.State(obd_db)

        # ── LOGIN PAGE ──────────────────────────────────────────────────────
        login_page = gr.Group(visible=True)
        with login_page:
            gr.Markdown("## 🔐 Login")
            in_user  = gr.Textbox(label="Username", placeholder="admin")
            in_pass  = gr.Textbox(label="Password", type="password", placeholder="car123")
            btn_login = gr.Button("Login")
            login_msg = gr.Markdown("")

        # ── MAIN APP (post-login) ───────────────────────────────────────────
        main_app = gr.Group(visible=False)
        with main_app:
            header = gr.HTML(render_header(""))

            with gr.Row():
                # ── SIDEBAR ─────────────────────────────────────────────────
                with gr.Column(scale=1, min_width=240):
                    btn_assistant = gr.Button("🧰  Car Assistant",      elem_classes=["sidebar-btn"])
                    btn_obd       = gr.Button("🔎  OBD Code Database",  elem_classes=["sidebar-btn"])
                    btn_lights    = gr.Button("🚨  Warning Light Icons", elem_classes=["sidebar-btn"])
                    btn_common    = gr.Button("🛠  Common Problems",     elem_classes=["sidebar-btn"])
                    gr.Markdown("---")
                    btn_logout    = gr.Button("🚪 Logout")

                # ── MAIN CONTENT ─────────────────────────────────────────────
                with gr.Column(scale=4):

                    # ── Assistant page ───────────────────────────────────────
                    page_assistant = gr.Group(visible=True)
                    with page_assistant:
                        gr.Markdown("### 🧰 Car Assistant (Symptoms → Guidance)")
                        symptoms_in = gr.Textbox(
                            lines=5,
                            label="Symptoms",
                            placeholder="e.g., Clicking sound when starting, no warning lights",
                        )
                        obd_in = gr.Textbox(
                            lines=1,
                            label="OBD Codes (optional)",
                            placeholder="e.g., P0300, P0171",
                        )
                        btn_diag = gr.Button("Diagnose ✅")
                        out_diag = gr.Textbox(lines=16, label="Diagnosis & Guidance")
                        btn_diag.click(
                            fn=safe_diagnose,
                            inputs=[symptoms_in, obd_in],
                            outputs=out_diag,
                        )
                        gr.Markdown("## Quick Examples (for demo)")
                        gr.Examples(
                            examples=GRADIO_EXAMPLES,
                            inputs=[symptoms_in, obd_in],
                        )
                        with gr.Row():
                            clear_btn = gr.Button("Clear")
                            clear_btn.click(lambda: ("", ""), outputs=[symptoms_in, obd_in])

                    # ── OBD page ─────────────────────────────────────────────
                    page_obd = gr.Group(visible=False)
                    with page_obd:
                        gr.Markdown("### 🔎 OBD Code Database")
                        gr.Markdown(
                            f"Loaded from `data/obd-trouble-codes.csv`  \n"
                            f"Total codes: **{len(obd_db)}**"
                        )
                        with gr.Row():
                            search_q  = gr.Textbox(label="Search (code/keyword)", placeholder="misfire / P03")
                            btn_search = gr.Button("Search")
                        codes_list = gr.Textbox(
                            lines=14,
                            value=obd_list_text(obd_db),
                            label="Available Codes (filtered)",
                        )
                        with gr.Row():
                            lookup_code = gr.Textbox(label="Lookup code", placeholder="P0302")
                            btn_lookup  = gr.Button("Lookup")
                        meaning = gr.Textbox(lines=6, label="Meaning")

                        btn_search.click(
                            fn=lambda q, db: obd_search(q, db),
                            inputs=[search_q, state_obd_db],
                            outputs=codes_list,
                        )
                        btn_lookup.click(
                            fn=lambda c, db: obd_lookup(c, db),
                            inputs=[lookup_code, state_obd_db],
                            outputs=meaning,
                        )

                    # ── Warning lights page ───────────────────────────────────
                    page_lights = gr.Group(visible=False)
                    with page_lights:
                        gr.Markdown("### 🚨 Warning Light Icons")
                        gr.Markdown(
                            f"Icons loaded from `assets/icons/`.  \n"
                            f"Missing icons show coloured placeholders automatically."
                        )
                        gallery = gr.Gallery(
                            label="Warning Lights (click for details)",
                            value=build_warning_items(),
                            columns=3,
                            rows=2,
                            height=440,
                            object_fit="contain",
                            elem_id="gallery_full",
                        )
                        explain_box = gr.Markdown("👆 Select an icon above to see details.")
                        gallery.select(fn=warning_explain, inputs=None, outputs=explain_box)

                    # ── Common problems page ──────────────────────────────────
                    page_common = gr.Group(visible=False)
                    with page_common:
                        gr.Markdown("### 🛠 Common Car Problems")
                        gr.Textbox(lines=18, value=common_text(), label="Quick Reference")

            # ── Sidebar navigation ───────────────────────────────────────────
            all_pages = [page_assistant, page_obd, page_lights, page_common]
            btn_assistant.click(fn=lambda: show_page("assistant"), inputs=None, outputs=all_pages)
            btn_obd.click(      fn=lambda: show_page("obd"),       inputs=None, outputs=all_pages)
            btn_lights.click(   fn=lambda: show_page("lights"),    inputs=None, outputs=all_pages)
            btn_common.click(   fn=lambda: show_page("common"),    inputs=None, outputs=all_pages)

            # ── Logout ────────────────────────────────────────────────────────
            btn_logout.click(
                fn=do_logout,
                inputs=None,
                outputs=[login_page, main_app, header, state_user],
            )

        # ── Login wiring ─────────────────────────────────────────────────────
        btn_login.click(
            fn=do_login,
            inputs=[in_user, in_pass],
            outputs=[login_page, main_app, login_msg, state_user],
        ).then(
            fn=render_header,
            inputs=[state_user],
            outputs=[header],
        )

    return app
