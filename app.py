"""
WHO Hand Hygiene Monitor
Developed by OMAC Developer — S M Baqir
"""

import streamlit as st
import cv2
import numpy as np
import datetime
import io
import time

from fpdf import FPDF

# ═══════════════════════════════════════════════════════════════
#  WHO STEP DEFINITIONS
# ═══════════════════════════════════════════════════════════════

WHO_HANDRUB_STEPS = [
    {"id":1,"name":"Apply product",          "caption":"Apply a palmful of alcohol-based product into a cupped hand.",              "icon":"💧"},
    {"id":2,"name":"Palm to palm",           "caption":"Rub hands palm to palm in a circular motion.",                             "icon":"🤲"},
    {"id":3,"name":"Palm over dorsum",       "caption":"Right palm over left dorsum with interlaced fingers — then switch.",        "icon":"✋"},
    {"id":4,"name":"Fingers interlaced",     "caption":"Palm to palm with fingers interlaced — rub back and forth.",               "icon":"🙌"},
    {"id":5,"name":"Backs of fingers",       "caption":"Backs of fingers to opposing palms with fingers interlocked.",             "icon":"👐"},
    {"id":6,"name":"Rotational — thumb",     "caption":"Rotational rubbing of left thumb clasped in right palm — then switch.",    "icon":"👍"},
    {"id":7,"name":"Rotational — fingertips","caption":"Rotational rubbing of fingertips of right hand in left palm — switch.",    "icon":"☝️"},
]

WHO_HANDWASH_STEPS = [
    {"id":1, "name":"Wet hands",              "caption":"Wet your hands with clean, running water.",                                "icon":"💧"},
    {"id":2, "name":"Apply soap",             "caption":"Apply enough soap to cover all hand surfaces.",                           "icon":"🧴"},
    {"id":3, "name":"Palm to palm",           "caption":"Rub hands palm to palm.",                                                 "icon":"🤲"},
    {"id":4, "name":"Palm over dorsum",       "caption":"Right palm over left dorsum with interlaced fingers — then switch.",       "icon":"✋"},
    {"id":5, "name":"Fingers interlaced",     "caption":"Palm to palm with fingers interlaced.",                                   "icon":"🙌"},
    {"id":6, "name":"Backs of fingers",       "caption":"Backs of fingers to opposing palms with fingers interlocked.",            "icon":"👐"},
    {"id":7, "name":"Rotational — thumb",     "caption":"Rotational rubbing of left thumb in right palm — then switch.",           "icon":"👍"},
    {"id":8, "name":"Rotational — fingertips","caption":"Rotational rubbing of fingertips of right hand in left palm — switch.",   "icon":"☝️"},
    {"id":9, "name":"Rinse hands",            "caption":"Rinse hands well under running water.",                                   "icon":"🚿"},
    {"id":10,"name":"Dry with towel",         "caption":"Dry hands thoroughly using a single-use towel.",                          "icon":"🧻"},
    {"id":11,"name":"Turn off tap",           "caption":"Use the towel to turn off the faucet.",                                   "icon":"🚰"},
]

# ═══════════════════════════════════════════════════════════════
#  HAND DETECTION  (skin-color segmentation via OpenCV only)
# ═══════════════════════════════════════════════════════════════

def detect_hands(frame_bgr):
    """
    Returns dict:
      num_hands      : 0 | 1 | 2   (estimated)
      both_present   : bool
      annotated      : frame with overlay drawn
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # Skin colour range in HSV
    lower = np.array([0,  20, 70],  dtype=np.uint8)
    upper = np.array([20, 255, 255], dtype=np.uint8)
    mask  = cv2.inRange(hsv, lower, upper)

    # Clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Keep only blobs big enough to be a hand
    hand_contours = [c for c in contours if cv2.contourArea(c) > 8000]
    num_hands = min(len(hand_contours), 2)

    annotated = frame_bgr.copy()
    cv2.drawContours(annotated, hand_contours, -1, (0, 200, 80), 3)

    # Bounding boxes
    for c in hand_contours[:2]:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(annotated, (x, y), (x+w, y+h), (79, 142, 247), 2)

    # Skin pixel ratio in centre strip — used as "hands together" proxy
    h_frame = frame_bgr.shape[0]
    w_frame = frame_bgr.shape[1]
    centre_mask = mask[h_frame//4 : 3*h_frame//4, w_frame//4 : 3*w_frame//4]
    skin_ratio  = np.count_nonzero(centre_mask) / centre_mask.size

    return {
        "num_hands"    : num_hands,
        "both_present" : num_hands >= 2,
        "skin_ratio"   : skin_ratio,
        "annotated"    : annotated,
    }


def validate_step(step_id, det):
    """Heuristic step validation from detection dict."""
    n       = det["num_hands"]
    both    = det["both_present"]
    skin    = det["skin_ratio"]
    rubbing = skin > 0.18   # large skin area in centre = hands together

    rules = {
        1 : n >= 1,
        2 : both and rubbing,
        3 : both and rubbing,
        4 : both and rubbing,
        5 : both and rubbing,
        6 : both and rubbing,
        7 : both and rubbing,
        8 : both and rubbing,
        9 : n >= 1,
        10: n >= 1,
        11: n >= 1,
    }
    return rules.get(step_id, n >= 1)


# ═══════════════════════════════════════════════════════════════
#  PDF REPORT
# ═══════════════════════════════════════════════════════════════

def make_pdf(full_name, emp_code, mode, step_statuses, steps):
    done      = sum(1 for s in step_statuses.values() if s == "done")
    score_pct = int(done / len(steps) * 100) if steps else 0
    errors    = sum(1 for s in step_statuses.values() if s == "error")

    pdf = FPDF()
    pdf.add_page()

    # Header banner
    pdf.set_fill_color(15, 82, 186)
    pdf.rect(0, 0, 210, 28, "F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 15)
    pdf.set_xy(10, 7)
    pdf.cell(190, 8, "WHO Hand Hygiene Compliance Report", align="C")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_xy(10, 17)
    pdf.cell(190, 6, "OMAC Developer  |  Developed by S M Baqir", align="C")

    # Info row
    pdf.set_text_color(30, 30, 30)
    pdf.set_xy(10, 34)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(65, 7, f"Name:  {full_name}")
    pdf.cell(55, 7, f"Code:  {emp_code}")
    pdf.cell(70, 7, f"Date:  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    pdf.set_xy(10, 43)
    pdf.cell(65, 7, f"Protocol:  {mode}")

    # Score badge
    sc_color = (22,163,74) if score_pct >= 80 else (217,119,6) if score_pct >= 50 else (220,38,38)
    pdf.set_fill_color(*sc_color)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_xy(10, 54)
    pdf.cell(190, 16, f"Score: {score_pct}%   |   {'PASS' if score_pct >= 80 else 'FAIL'}", align="C", fill=True)

    # Table header
    pdf.set_text_color(30, 30, 30)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_xy(10, 76)
    pdf.set_fill_color(210, 225, 255)
    pdf.cell(10, 7, "#",       border=1, fill=True, align="C")
    pdf.cell(55, 7, "Step",    border=1, fill=True)
    pdf.cell(105,7, "Instruction", border=1, fill=True)
    pdf.cell(20, 7, "Result",  border=1, fill=True, align="C")
    pdf.ln()

    pdf.set_font("Helvetica", "", 9)
    for i, step in enumerate(steps):
        status = step_statuses.get(step["id"], "pending")
        res    = "✓ OK" if status == "done" else ("✗ ERR" if status == "error" else "—")
        if status == "done":
            pdf.set_fill_color(220, 252, 231)
        elif status == "error":
            pdf.set_fill_color(254, 226, 226)
        else:
            pdf.set_fill_color(249, 250, 251)

        caption = step["caption"][:72] + "…" if len(step["caption"]) > 72 else step["caption"]
        pdf.cell(10, 7, str(i+1), border=1, fill=True, align="C")
        pdf.cell(55, 7, step["name"], border=1, fill=True)
        pdf.cell(105,7, caption, border=1, fill=True)

        if status == "done":
            pdf.set_text_color(22, 163, 74)
        elif status == "error":
            pdf.set_text_color(220, 38, 38)
        else:
            pdf.set_text_color(120, 120, 120)
        pdf.cell(20, 7, res, border=1, fill=True, align="C")
        pdf.set_text_color(30, 30, 30)
        pdf.ln()

    # Summary line
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(65, 7, f"Correct: {done}/{len(steps)}")
    pdf.cell(65, 7, f"Errors: {errors}")
    pdf.cell(60, 7, f"Compliance: {'PASS ✓' if score_pct >= 80 else 'FAIL ✗'}")

    # Footer
    pdf.set_y(-14)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(140, 140, 140)
    pdf.cell(190, 6, "Developed by OMAC Developer — S M Baqir  |  WHO: who.int/gpsc/hand-hygiene", align="C")

    buf = io.BytesIO()
    pdf.output(buf)
    buf.seek(0)
    return buf.read(), score_pct


# ═══════════════════════════════════════════════════════════════
#  SESSION STATE
# ═══════════════════════════════════════════════════════════════

def init():
    for k, v in {
        "full_name": "", "emp_code": "", "mode": "Hand Rub (WHO)",
        "started": False, "current_step": 0, "step_statuses": {},
        "session_done": False, "theme": "dark",
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ═══════════════════════════════════════════════════════════════
#  CSS
# ═══════════════════════════════════════════════════════════════

def css(dark):
    bg, surf, bdr = ("#0f1117","#1a1d27","rgba(255,255,255,0.08)") if dark else ("#f5f6fa","#ffffff","rgba(0,0,0,0.08)")
    txt, txt2     = ("#e6e9f0","#9ca3b8") if dark else ("#1a1d27","#6b7280")
    acc           = "#4f8ef7" if dark else "#2563eb"
    ok            = "#22c97a" if dark else "#16a34a"
    err           = "#f05454" if dark else "#dc2626"

    st.markdown(f"""<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    html,body,[data-testid="stAppViewContainer"]{{background:{bg}!important;color:{txt}!important;font-family:'Inter',sans-serif!important}}
    [data-testid="stSidebar"]{{background:{surf}!important;border-right:1px solid {bdr}!important}}
    .scard{{background:{surf};border:1px solid {bdr};border-radius:10px;padding:10px 13px;margin-bottom:7px}}
    .scard.active{{border-color:{acc};background:{"rgba(79,142,247,0.07)" if dark else "rgba(37,99,235,0.05)"};box-shadow:0 0 0 2px {"rgba(79,142,247,0.14)" if dark else "rgba(37,99,235,0.1)"}}}
    .scard.done{{border-color:{ok};background:{"rgba(34,201,122,0.06)" if dark else "rgba(22,163,74,0.05)"}}}
    .scard.error{{border-color:{err};background:{"rgba(240,84,84,0.06)" if dark else "rgba(220,38,38,0.05)"}}}
    .caption{{background:{"rgba(79,142,247,0.07)" if dark else "rgba(37,99,235,0.05)"};border-left:3px solid {acc};border-radius:8px;padding:11px 15px;margin:10px 0}}
    .scorebox{{background:{surf};border:1px solid {bdr};border-radius:12px;padding:18px;text-align:center;margin-top:12px}}
    .footer{{position:fixed;bottom:0;left:0;right:0;background:{surf};border-top:1px solid {bdr};padding:7px 0;text-align:center;font-size:12px;color:{txt2};z-index:999}}
    .badge{{display:inline-block;font-size:11px;padding:2px 8px;border-radius:20px;font-weight:500}}
    .ba{{background:{"rgba(79,142,247,0.15)" if dark else "rgba(37,99,235,0.1)"};color:{acc}}}
    .bd{{background:{"rgba(34,201,122,0.15)" if dark else "rgba(22,163,74,0.1)"};color:{ok}}}
    .be{{background:{"rgba(240,84,84,0.15)" if dark else "rgba(220,38,38,0.1)"};color:{err}}}
    div[data-testid="stVerticalBlock"]{{padding-bottom:46px}}
    </style>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  STEP CARD
# ═══════════════════════════════════════════════════════════════

def step_card(step, status, is_cur):
    cls   = "scard " + ("active" if is_cur else status if status in ("done","error") else "")
    badge = ('<span class="badge ba">Detecting…</span>' if is_cur else
             '<span class="badge bd">✓ Done</span>'     if status == "done" else
             '<span class="badge be">✗ Redo</span>'     if status == "error" else "")
    st.markdown(f"""<div class="{cls}">
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <span style="font-size:13px;font-weight:500">{step['icon']} {step['name']}</span>{badge}
      </div>
      <div style="font-size:11px;opacity:.6;margin-top:3px">{step['caption']}</div>
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    init()
    st.set_page_config(page_title="WHO Hand Hygiene Monitor", page_icon="🤲", layout="wide")
    dark = st.session_state.theme == "dark"
    css(dark)

    steps = WHO_HANDRUB_STEPS if "Rub" in st.session_state.mode else WHO_HANDWASH_STEPS
    total = len(steps)

    # ── SIDEBAR ──────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🤲 Hand Hygiene Monitor")
        st.caption("WHO Protocol · AI-Powered")
        st.divider()

        if st.button("☀️ Light mode" if dark else "🌙 Dark mode", use_container_width=True):
            st.session_state.theme = "light" if dark else "dark"
            st.rerun()

        st.divider()
        st.session_state.full_name = st.text_input("Full Name",      value=st.session_state.full_name, placeholder="e.g. Ahmed Ali Khan")
        st.session_state.emp_code  = st.text_input("Employee Code",  value=st.session_state.emp_code,  placeholder="e.g. EMP-0042")
        st.session_state.mode      = st.radio("Protocol", ["Hand Rub (WHO)", "Hand Wash (WHO)"],
                                               index=0 if "Rub" in st.session_state.mode else 1)
        steps = WHO_HANDRUB_STEPS if "Rub" in st.session_state.mode else WHO_HANDWASH_STEPS
        total = len(steps)

        st.divider()
        if not st.session_state.started:
            if st.button("▶  Start Session", use_container_width=True, type="primary"):
                if not st.session_state.full_name.strip():
                    st.error("Enter your full name.")
                elif not st.session_state.emp_code.strip():
                    st.error("Enter employee code.")
                else:
                    st.session_state.started      = True
                    st.session_state.current_step = 0
                    st.session_state.step_statuses= {s["id"]: "pending" for s in steps}
                    st.session_state.session_done = False
                    st.rerun()
        else:
            if st.button("⏹  Reset", use_container_width=True):
                st.session_state.started       = False
                st.session_state.current_step  = 0
                st.session_state.step_statuses = {}
                st.session_state.session_done  = False
                st.rerun()

        st.divider()
        st.caption("**WHO Reference**\nHand rub: 20–30 sec\nHand wash: 40–60 sec")

    # ── MAIN COLUMNS ─────────────────────────────────────────────
    col_cam, col_steps = st.columns([3, 2], gap="large")

    with col_cam:
        if st.session_state.full_name:
            st.markdown(f"#### Welcome, {st.session_state.full_name}")
            if st.session_state.emp_code:
                st.caption(f"Employee Code: **{st.session_state.emp_code}**")
        else:
            st.markdown("#### WHO Hand Hygiene Monitor")

        if not st.session_state.started:
            st.info("Fill in your details in the sidebar, then press **Start Session**.")

        else:
            cur_idx = st.session_state.current_step
            cur     = steps[cur_idx] if cur_idx < total else None

            if cur and not st.session_state.session_done:
                acc = "#4f8ef7" if dark else "#2563eb"
                st.markdown(f"""<div class="caption">
                  <div style="font-size:11px;font-weight:600;color:{acc};margin-bottom:4px">
                    STEP {cur['id']} of {total} — {cur['name'].upper()}
                  </div>
                  <div style="font-size:14px">{cur['icon']} {cur['caption']}</div>
                </div>""", unsafe_allow_html=True)

                # Camera snapshot
                snap = st.camera_input("📷 Take snapshot to verify this step", key=f"cam_{cur['id']}")

                if snap:
                    arr    = np.frombuffer(snap.read(), np.uint8)
                    frame  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    det    = detect_hands(frame)
                    ok     = validate_step(cur["id"], det)

                    # Coloured border overlay
                    color  = (0, 200, 80) if ok else (60, 60, 240)
                    h, w   = frame.shape[:2]
                    cv2.rectangle(frame, (0,0), (w,h), color, 10)
                    label  = "✅  Correct posture!" if ok else "❌  Adjust hand position"
                    cv2.rectangle(frame, (0, h-44), (w, h), (0,0,0), -1)
                    cv2.putText(frame, "Correct posture!" if ok else "Adjust hand position",
                                (12, h-14), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0,200,80) if ok else (60,80,240), 2)

                    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                             caption=label, use_container_width=True)

                    b1, b2 = st.columns(2)
                    with b1:
                        if st.button("✅  Accept & next", use_container_width=True, type="primary"):
                            _advance(steps, cur["id"], True)
                            st.rerun()
                    with b2:
                        if st.button("❌  Mark error & next", use_container_width=True):
                            _advance(steps, cur["id"], False)
                            st.rerun()

                    if ok:
                        st.success(f"Hands detected: {det['num_hands']} — posture looks correct!")
                    else:
                        st.error(f"Hands detected: {det['num_hands']} — posture needs adjustment.")

                else:
                    st.caption("Take a snapshot to analyse your hand position, then accept or mark error.")

    # ── STEPS COLUMN ─────────────────────────────────────────────
    with col_steps:
        st.markdown("#### Steps Checklist")

        for i, step in enumerate(steps):
            status = st.session_state.step_statuses.get(step["id"], "pending")
            is_cur = (st.session_state.started
                      and i == st.session_state.current_step
                      and not st.session_state.session_done)
            step_card(step, status, is_cur)

        if st.session_state.started:
            done   = sum(1 for s in st.session_state.step_statuses.values() if s == "done")
            errors = sum(1 for s in st.session_state.step_statuses.values() if s == "error")
            pct    = int(done / total * 100) if total else 0
            col    = "#22c97a" if pct >= 80 else ("#f5a623" if pct >= 50 else "#f05454")
            st.markdown(f"""<div class="scorebox">
              <div style="font-size:11px;opacity:.6;margin-bottom:6px">SESSION SCORE</div>
              <div style="font-size:44px;font-weight:600;color:{col};line-height:1">{pct}%</div>
              <div style="font-size:12px;opacity:.6;margin-top:6px">{done} correct · {errors} errors · {total-done-errors} remaining</div>
            </div>""", unsafe_allow_html=True)

        if st.session_state.session_done:
            done  = sum(1 for s in st.session_state.step_statuses.values() if s == "done")
            pct   = int(done / total * 100)
            grade = "Excellent 🏆" if pct == 100 else "Good 👍" if pct >= 80 else "Needs Practice ⚠️"
            st.success(f"**Session complete!** {grade} — {pct}%")
            st.balloons()

            pdf_bytes, _ = make_pdf(
                st.session_state.full_name,
                st.session_state.emp_code,
                st.session_state.mode,
                st.session_state.step_statuses,
                steps,
            )
            fname = f"HH_{st.session_state.emp_code}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
            st.download_button("📥  Download PDF Report", pdf_bytes, fname,
                               mime="application/pdf", use_container_width=True, type="primary")

    # ── FOOTER ───────────────────────────────────────────────────
    st.markdown(
        '<div class="footer">OMAC Developer &nbsp;|&nbsp; Developed by <b>S M Baqir</b> &nbsp;|&nbsp; WHO Hand Hygiene Monitoring Tool</div>',
        unsafe_allow_html=True,
    )


def _advance(steps, step_id, success):
    st.session_state.step_statuses[step_id] = "done" if success else "error"
    nxt = st.session_state.current_step + 1
    if nxt >= len(steps):
        st.session_state.session_done = True
    else:
        st.session_state.current_step = nxt


if __name__ == "__main__":
    main()
