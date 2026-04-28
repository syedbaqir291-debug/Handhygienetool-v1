"""
WHO Hand Hygiene Monitor
Developed by OMAC Developer — S M Baqir
"""

import streamlit as st
import numpy as np
import time
import io
import datetime

# ── lazy-import opencv so Streamlit Cloud can finish loading even if wheel
#    is still being built; real usage is only inside the video processor
try:
    import cv2
    CV2_OK = True
except ModuleNotFoundError:
    CV2_OK = False

try:
    import mediapipe as mp
    MP_OK = True
except ModuleNotFoundError:
    MP_OK = False

try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
    import av
    WEBRTC_OK = True
except ModuleNotFoundError:
    WEBRTC_OK = False

try:
    from fpdf import FPDF
    FPDF_OK = True
except ModuleNotFoundError:
    FPDF_OK = False


# ═══════════════════════ WHO STEP DEFINITIONS ═══════════════════════════════

WHO_HANDRUB_STEPS = [
    {"id":1,"name":"Apply product",         "caption":"Apply a palmful of alcohol-based product into a cupped hand.",                     "duration":3,"icon":"💧"},
    {"id":2,"name":"Palm to palm",          "caption":"Rub hands palm to palm in a circular motion.",                                    "duration":4,"icon":"🤲"},
    {"id":3,"name":"Palm over dorsum",      "caption":"Right palm over left dorsum with interlaced fingers — then switch.",               "duration":4,"icon":"✋"},
    {"id":4,"name":"Fingers interlaced",    "caption":"Palm to palm with fingers interlaced — rub back and forth.",                      "duration":4,"icon":"🙌"},
    {"id":5,"name":"Backs of fingers",      "caption":"Backs of fingers to opposing palms with fingers interlocked.",                    "duration":4,"icon":"👐"},
    {"id":6,"name":"Rotational — thumb",    "caption":"Rotational rubbing of left thumb clasped in right palm — then switch.",           "duration":4,"icon":"👍"},
    {"id":7,"name":"Rotational — fingertips","caption":"Rotational rubbing of fingertips of right hand in left palm — then switch.",    "duration":4,"icon":"☝️"},
]

WHO_HANDWASH_STEPS = [
    {"id":1, "name":"Wet hands",             "caption":"Wet your hands with clean, running water.",                                       "duration":3,"icon":"💧"},
    {"id":2, "name":"Apply soap",            "caption":"Apply enough soap to cover all hand surfaces.",                                   "duration":3,"icon":"🧴"},
    {"id":3, "name":"Palm to palm",          "caption":"Rub hands palm to palm.",                                                        "duration":5,"icon":"🤲"},
    {"id":4, "name":"Palm over dorsum",      "caption":"Right palm over left dorsum with interlaced fingers — then switch.",               "duration":5,"icon":"✋"},
    {"id":5, "name":"Fingers interlaced",    "caption":"Palm to palm with fingers interlaced.",                                           "duration":5,"icon":"🙌"},
    {"id":6, "name":"Backs of fingers",      "caption":"Backs of fingers to opposing palms with fingers interlocked.",                    "duration":5,"icon":"👐"},
    {"id":7, "name":"Rotational — thumb",    "caption":"Rotational rubbing of left thumb in right palm — then switch.",                   "duration":5,"icon":"👍"},
    {"id":8, "name":"Rotational — fingertips","caption":"Rotational rubbing of fingertips of right hand in left palm — then switch.",    "duration":5,"icon":"☝️"},
    {"id":9, "name":"Rinse hands",           "caption":"Rinse hands well under running water.",                                          "duration":5,"icon":"🚿"},
    {"id":10,"name":"Dry with towel",        "caption":"Dry hands thoroughly using a single-use towel.",                                  "duration":5,"icon":"🧻"},
    {"id":11,"name":"Turn off tap",          "caption":"Use the towel to turn off the faucet.",                                          "duration":3,"icon":"🚰"},
]


# ═══════════════════════ MEDIAPIPE HAND ANALYSER ════════════════════════════

class HandAnalyser:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
        )

    def analyse(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        annotated = frame_bgr.copy()
        num_hands = 0
        hands_touching = False
        fingers_interlaced = False

        if results.multi_hand_landmarks:
            num_hands = len(results.multi_hand_landmarks)
            for lm in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    annotated, lm,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style(),
                )
            if num_hands == 2:
                lm0 = results.multi_hand_landmarks[0].landmark
                lm1 = results.multi_hand_landmarks[1].landmark
                w0 = np.array([lm0[0].x, lm0[0].y])
                w1 = np.array([lm1[0].x, lm1[0].y])
                hands_touching = np.linalg.norm(w0 - w1) < 0.35
                tips_0 = [lm0[i].x for i in [8, 12, 16, 20]]
                tips_1 = [lm1[i].x for i in [8, 12, 16, 20]]
                all_tips = sorted(zip(tips_0 + tips_1, [0]*4 + [1]*4))
                hand_seq = [h for _, h in all_tips]
                alternating = sum(1 for i in range(len(hand_seq)-1) if hand_seq[i] != hand_seq[i+1])
                fingers_interlaced = alternating >= 4

        return {
            "annotated_frame": annotated,
            "num_hands": num_hands,
            "hands_touching": hands_touching,
            "fingers_interlaced": fingers_interlaced,
        }

    def validate_step(self, step_id, analysis):
        n = analysis["num_hands"]
        touching = analysis["hands_touching"]
        interlaced = analysis["fingers_interlaced"]
        rules = {
            1: n >= 1,
            2: touching,
            3: touching,
            4: interlaced,
            5: touching and not interlaced,
            6: touching,
            7: touching,
            8: touching and not interlaced,
            9: n >= 1,
            10: n >= 1,
            11: n >= 1,
        }
        return rules.get(step_id, n >= 1)


# ═══════════════════════ WEBRTC VIDEO PROCESSOR ═════════════════════════════

class HygieneVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.analyser = HandAnalyser()
        self.current_step_id = 1
        self.step_result = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        analysis = self.analyser.analyse(img)
        annotated = analysis["annotated_frame"]
        valid = self.analyser.validate_step(self.current_step_id, analysis)
        self.step_result = valid

        color = (0, 200, 80) if valid else (60, 60, 240)
        h, w = annotated.shape[:2]
        cv2.rectangle(annotated, (0, 0), (w, h), color, 8)
        label = "Correct posture" if valid else "Adjust hand position"
        cv2.rectangle(annotated, (0, h - 44), (w, h), (0, 0, 0), -1)
        cv2.putText(annotated, label, (12, h - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 200, 80) if valid else (60, 80, 240), 2)
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


# ═══════════════════════ PDF REPORT GENERATOR ═══════════════════════════════

def generate_pdf_report(full_name, emp_code, mode, step_statuses, steps, score_pct):
    pdf = FPDF()
    pdf.add_page()

    # Header
    pdf.set_fill_color(15, 82, 186)
    pdf.rect(0, 0, 210, 30, "F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_xy(10, 8)
    pdf.cell(190, 10, "WHO Hand Hygiene Compliance Report", align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_xy(10, 20)
    pdf.cell(190, 6, "OMAC Developer — S M Baqir", align="C")

    # Employee info
    pdf.set_text_color(30, 30, 30)
    pdf.set_xy(10, 38)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(95, 8, f"Full Name:  {full_name}")
    pdf.cell(95, 8, f"Employee Code:  {emp_code}")
    pdf.ln(8)
    pdf.set_xy(10, 48)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(95, 7, f"Protocol:  {mode}")
    pdf.cell(95, 7, f"Date:  {datetime.datetime.now().strftime('%Y-%m-%d  %H:%M')}")

    # Score box
    score_color = (22, 163, 74) if score_pct >= 80 else (217, 119, 6) if score_pct >= 50 else (220, 38, 38)
    pdf.set_xy(10, 60)
    pdf.set_fill_color(*score_color)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 22)
    pdf.cell(190, 18, f"Score: {score_pct}%", align="C", fill=True)

    # Step results table
    pdf.set_text_color(30, 30, 30)
    pdf.set_xy(10, 85)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(190, 8, "Step-by-Step Results", ln=True)

    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(220, 230, 255)
    pdf.cell(12, 7, "#", border=1, fill=True, align="C")
    pdf.cell(70, 7, "Step", border=1, fill=True)
    pdf.cell(95, 7, "Instruction", border=1, fill=True)
    pdf.cell(13, 7, "Result", border=1, fill=True, align="C")
    pdf.ln()

    pdf.set_font("Helvetica", "", 9)
    for i, step in enumerate(steps):
        status = step_statuses.get(step["id"], "pending")
        result_text = "OK" if status == "done" else ("ERR" if status == "error" else "-")
        if status == "done":
            pdf.set_fill_color(220, 252, 231)
        elif status == "error":
            pdf.set_fill_color(254, 226, 226)
        else:
            pdf.set_fill_color(249, 250, 251)

        row_h = 8
        # Wrap long caption
        caption = step["caption"]
        if len(caption) > 70:
            caption = caption[:68] + "…"

        pdf.cell(12, row_h, str(i+1), border=1, fill=True, align="C")
        pdf.cell(70, row_h, step["name"], border=1, fill=True)
        pdf.cell(95, row_h, caption, border=1, fill=True)
        c = (22,163,74) if status=="done" else (220,38,38) if status=="error" else (100,100,100)
        pdf.set_text_color(*c)
        pdf.cell(13, row_h, result_text, border=1, fill=True, align="C")
        pdf.set_text_color(30, 30, 30)
        pdf.ln()

    # Summary
    done = sum(1 for s in step_statuses.values() if s == "done")
    errors = sum(1 for s in step_statuses.values() if s == "error")
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(60, 7, f"Correct steps: {done}/{len(steps)}")
    pdf.cell(60, 7, f"Errors: {errors}")
    pdf.cell(60, 7, f"Compliance: {'PASS' if score_pct >= 80 else 'FAIL'}")

    # Footer
    pdf.set_y(-18)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(190, 6, "Developed by OMAC Developer — S M Baqir  |  WHO Hand Hygiene Reference: who.int", align="C")

    buf = io.BytesIO()
    pdf.output(buf)
    buf.seek(0)
    return buf.read()


# ═══════════════════════ SESSION STATE INIT ═════════════════════════════════

def init_state():
    defaults = {
        "full_name": "",
        "emp_code": "",
        "mode": "Hand Rub (WHO)",
        "started": False,
        "current_step": 0,
        "step_statuses": {},
        "step_start_time": None,
        "session_done": False,
        "theme": "dark",
        "pdf_ready": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ═══════════════════════ CSS ════════════════════════════════════════════════

def inject_css(dark: bool):
    bg       = "#0f1117" if dark else "#f5f6fa"
    surface  = "#1a1d27" if dark else "#ffffff"
    surface2 = "#222536" if dark else "#f0f2f8"
    border   = "rgba(255,255,255,0.08)" if dark else "rgba(0,0,0,0.08)"
    text     = "#e6e9f0" if dark else "#1a1d27"
    text2    = "#9ca3b8" if dark else "#6b7280"
    accent   = "#4f8ef7" if dark else "#2563eb"
    success  = "#22c97a" if dark else "#16a34a"
    danger   = "#f05454" if dark else "#dc2626"

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    html, body, [data-testid="stAppViewContainer"] {{
        background: {bg} !important;
        color: {text} !important;
        font-family: 'Inter', sans-serif !important;
    }}
    [data-testid="stSidebar"] {{
        background: {surface} !important;
        border-right: 1px solid {border} !important;
    }}
    .step-card {{
        background: {surface};
        border: 1px solid {border};
        border-radius: 10px;
        padding: 11px 13px;
        margin-bottom: 7px;
    }}
    .step-card.active {{
        border-color: {accent};
        background: {"rgba(79,142,247,0.07)" if dark else "rgba(37,99,235,0.05)"};
        box-shadow: 0 0 0 2px {"rgba(79,142,247,0.15)" if dark else "rgba(37,99,235,0.12)"};
    }}
    .step-card.done  {{ border-color:{success}; background:{"rgba(34,201,122,0.05)" if dark else "rgba(22,163,74,0.05)"}; }}
    .step-card.error {{ border-color:{danger};  background:{"rgba(240,84,84,0.05)"   if dark else "rgba(220,38,38,0.05)"}; }}
    .caption-box {{
        background: {"rgba(79,142,247,0.07)" if dark else "rgba(37,99,235,0.05)"};
        border-left: 3px solid {accent};
        border-radius: 8px;
        padding: 11px 15px;
        margin: 10px 0;
    }}
    .score-box {{
        background: {surface};
        border: 1px solid {border};
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        margin-top: 14px;
    }}
    .footer-bar {{
        position: fixed;
        bottom: 0; left: 0; right: 0;
        background: {surface};
        border-top: 1px solid {border};
        padding: 8px 0;
        text-align: center;
        font-size: 12px;
        color: {text2};
        z-index: 999;
    }}
    .badge {{
        display: inline-block; font-size: 11px;
        padding: 2px 8px; border-radius: 20px; font-weight: 500;
    }}
    .badge-active {{ background:{"rgba(79,142,247,0.15)" if dark else "rgba(37,99,235,0.1)"}; color:{accent}; }}
    .badge-done   {{ background:{"rgba(34,201,122,0.15)" if dark else "rgba(22,163,74,0.1)"}; color:{success}; }}
    .badge-error  {{ background:{"rgba(240,84,84,0.15)"  if dark else "rgba(220,38,38,0.1)"}; color:{danger}; }}
    div[data-testid="stVerticalBlock"] {{ padding-bottom: 48px; }}
    </style>
    """, unsafe_allow_html=True)


# ═══════════════════════ STEP CARD ══════════════════════════════════════════

def render_step_card(step, status, is_current):
    css_class = "step-card " + ("active" if is_current else status if status in ("done","error") else "")
    badge = (
        '<span class="badge badge-active">Detecting…</span>' if is_current else
        '<span class="badge badge-done">✓ Done</span>'      if status == "done" else
        '<span class="badge badge-error">✗ Redo</span>'     if status == "error" else ""
    )
    st.markdown(f"""
    <div class="{css_class}">
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <span style="font-size:13px;font-weight:500;">{step['icon']} {step['name']}</span>
        {badge}
      </div>
      <div style="font-size:11px;opacity:0.6;margin-top:3px;">{step['caption']}</div>
    </div>""", unsafe_allow_html=True)


# ═══════════════════════ STEP ADVANCE HELPER ════════════════════════════════

def advance_step(steps, step_id, success):
    st.session_state.step_statuses[step_id] = "done" if success else "error"
    nxt = st.session_state.current_step + 1
    if nxt >= len(steps):
        st.session_state.session_done = True
        st.session_state.pdf_ready = True
    else:
        st.session_state.current_step = nxt
        st.session_state.step_start_time = time.time()


# ═══════════════════════ MAIN ═══════════════════════════════════════════════

def main():
    init_state()
    st.set_page_config(
        page_title="WHO Hand Hygiene Monitor",
        page_icon="🤲",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    dark = st.session_state.theme == "dark"
    inject_css(dark)

    # ── Sidebar ─────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🤲 Hand Hygiene Monitor")
        st.caption("WHO Protocol · AI-Powered")
        st.divider()

        if st.button("☀️ Light" if dark else "🌙 Dark", use_container_width=True):
            st.session_state.theme = "light" if dark else "dark"
            st.rerun()

        st.divider()
        st.session_state.full_name = st.text_input(
            "Full Name", value=st.session_state.full_name, placeholder="e.g. Ahmed Ali Khan")
        st.session_state.emp_code = st.text_input(
            "Employee Code", value=st.session_state.emp_code, placeholder="e.g. EMP-0042")

        st.session_state.mode = st.radio(
            "Protocol",
            ["Hand Rub (WHO)", "Hand Wash (WHO)"],
            index=0 if "Rub" in st.session_state.mode else 1,
        )

        steps = WHO_HANDRUB_STEPS if "Rub" in st.session_state.mode else WHO_HANDWASH_STEPS

        st.divider()
        if not st.session_state.started:
            if st.button("▶  Start Session", use_container_width=True, type="primary"):
                if not st.session_state.full_name.strip():
                    st.error("Please enter your full name.")
                elif not st.session_state.emp_code.strip():
                    st.error("Please enter your employee code.")
                else:
                    st.session_state.started = True
                    st.session_state.current_step = 0
                    st.session_state.step_statuses = {s["id"]: "pending" for s in steps}
                    st.session_state.step_start_time = time.time()
                    st.session_state.session_done = False
                    st.session_state.pdf_ready = False
                    st.rerun()
        else:
            if st.button("⏹  Reset", use_container_width=True):
                for k in ("started","current_step","step_statuses","session_done","pdf_ready"):
                    st.session_state[k] = False if k != "step_statuses" and k != "current_step" else ({} if k=="step_statuses" else 0)
                st.session_state.started = False
                st.rerun()

        st.divider()
        st.caption("**WHO reference**\nHand rub: 20–30 sec\nHand wash: 40–60 sec")

        # ── Library status warnings ──────────────────────────────────────
        if not CV2_OK:
            st.warning("OpenCV not loaded yet — restart may be needed.", icon="⚠️")
        if not MP_OK:
            st.warning("MediaPipe not loaded.", icon="⚠️")
        if not WEBRTC_OK:
            st.warning("streamlit-webrtc not loaded.", icon="⚠️")

    # ── Main layout ─────────────────────────────────────────────────────────
    steps = WHO_HANDRUB_STEPS if "Rub" in st.session_state.mode else WHO_HANDWASH_STEPS
    total_steps = len(steps)

    col_cam, col_steps = st.columns([3, 2], gap="large")

    # ── Camera / detection column ────────────────────────────────────────
    with col_cam:
        if st.session_state.full_name:
            st.markdown(f"#### Welcome, {st.session_state.full_name}")
            if st.session_state.emp_code:
                st.caption(f"Employee Code: **{st.session_state.emp_code}**")
        else:
            st.markdown("#### WHO Hand Hygiene Monitor")

        if not st.session_state.started:
            st.info("Fill in your **Full Name** and **Employee Code**, then press **Start Session**.")

            # Snapshot camera test (always works, no webrtc needed)
            st.caption("Camera test — take a snapshot:")
            snapshot = st.camera_input("Snapshot test")
            if snapshot and CV2_OK and MP_OK:
                img_array = np.frombuffer(snapshot.read(), np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                analyser = HandAnalyser()
                result = analyser.analyse(frame)
                st.image(
                    cv2.cvtColor(result["annotated_frame"], cv2.COLOR_BGR2RGB),
                    caption=f"Hands detected: {result['num_hands']}",
                    use_container_width=True,
                )
            elif snapshot and not CV2_OK:
                st.warning("OpenCV not available for preview — start session to use live mode.")

        else:
            current_step_obj = steps[st.session_state.current_step] if st.session_state.current_step < total_steps else None

            if WEBRTC_OK and CV2_OK and MP_OK:
                processor = HygieneVideoProcessor()
                if current_step_obj:
                    processor.current_step_id = current_step_obj["id"]

                ctx = webrtc_streamer(
                    key="hygiene-monitor",
                    video_processor_factory=lambda: processor,
                    rtc_configuration=RTCConfiguration(
                        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
                    ),
                    media_stream_constraints={"video": True, "audio": False},
                    async_processing=True,
                )
            else:
                st.warning("Live camera modules not available — use manual buttons below.")
                ctx = None

            # Caption
            if current_step_obj and not st.session_state.session_done:
                accent = "#4f8ef7" if dark else "#2563eb"
                st.markdown(f"""
                <div class="caption-box">
                  <div style="font-size:11px;font-weight:600;color:{accent};margin-bottom:4px;">
                    STEP {current_step_obj['id']} of {total_steps} — {current_step_obj['name'].upper()}
                  </div>
                  <div style="font-size:14px;">{current_step_obj['icon']} {current_step_obj['caption']}</div>
                </div>""", unsafe_allow_html=True)

                b1, b2 = st.columns(2)
                with b1:
                    if st.button("✅ Mark correct", use_container_width=True):
                        advance_step(steps, current_step_obj["id"], True)
                        st.rerun()
                with b2:
                    if st.button("❌ Mark error", use_container_width=True):
                        advance_step(steps, current_step_obj["id"], False)
                        st.rerun()

                # Auto-advance from WebRTC processor
                if ctx and ctx.video_processor and ctx.video_processor.step_result is not None:
                    elapsed = time.time() - (st.session_state.step_start_time or time.time())
                    if elapsed >= current_step_obj["duration"]:
                        advance_step(steps, current_step_obj["id"], ctx.video_processor.step_result)
                        st.rerun()

    # ── Steps checklist column ───────────────────────────────────────────
    with col_steps:
        st.markdown("#### Steps checklist")

        for i, step in enumerate(steps):
            status = st.session_state.step_statuses.get(step["id"], "pending")
            is_current = (
                st.session_state.started
                and i == st.session_state.current_step
                and not st.session_state.session_done
            )
            render_step_card(step, status, is_current)

        # Score
        if st.session_state.started:
            done   = sum(1 for s in st.session_state.step_statuses.values() if s == "done")
            errors = sum(1 for s in st.session_state.step_statuses.values() if s == "error")
            score_pct = int((done / total_steps) * 100) if total_steps else 0
            color = "#22c97a" if score_pct >= 80 else ("#f5a623" if score_pct >= 50 else "#f05454")

            st.markdown(f"""
            <div class="score-box">
              <div style="font-size:11px;opacity:0.6;margin-bottom:6px;">SESSION SCORE</div>
              <div style="font-size:46px;font-weight:600;color:{color};line-height:1;">{score_pct}%</div>
              <div style="font-size:12px;opacity:0.6;margin-top:6px;">
                {done} correct · {errors} errors · {total_steps - done - errors} remaining
              </div>
            </div>""", unsafe_allow_html=True)

        # Session complete + download
        if st.session_state.session_done:
            done = sum(1 for s in st.session_state.step_statuses.values() if s == "done")
            score_pct = int((done / total_steps) * 100)
            grade = "Excellent 🏆" if score_pct == 100 else "Good 👍" if score_pct >= 80 else "Needs Practice ⚠️"
            st.success(f"**Session complete!** {grade} — {score_pct}%")
            st.balloons()

            if FPDF_OK:
                pdf_bytes = generate_pdf_report(
                    st.session_state.full_name,
                    st.session_state.emp_code,
                    st.session_state.mode,
                    st.session_state.step_statuses,
                    steps,
                    score_pct,
                )
                fname = f"HH_{st.session_state.emp_code}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
                st.download_button(
                    label="📥  Download PDF Report",
                    data=pdf_bytes,
                    file_name=fname,
                    mime="application/pdf",
                    use_container_width=True,
                    type="primary",
                )
            else:
                st.info("Install `fpdf2` to enable PDF download.")

    # ── Footer ───────────────────────────────────────────────────────────
    st.markdown(
        '<div class="footer-bar">OMAC Developer &nbsp;|&nbsp; Developed by <b>S M Baqir</b> &nbsp;|&nbsp; WHO Hand Hygiene Monitoring Tool</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
