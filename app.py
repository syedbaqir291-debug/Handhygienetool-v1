"""
WHO Hand Hygiene Monitor — Streamlit App
Real-time hand rub / hand wash step detection via webcam + MediaPipe
"""

import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import time
from dataclasses import dataclass, field
from typing import Optional
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

# ─────────────────────────── WHO STEP DEFINITIONS ──────────────────────────

WHO_HANDRUB_STEPS = [
    {
        "id": 1,
        "name": "Apply product",
        "caption": "Apply a palmful of the alcohol-based product into a cupped hand.",
        "duration": 3,
        "icon": "💧",
    },
    {
        "id": 2,
        "name": "Palm to palm",
        "caption": "Rub hands palm to palm in a circular motion.",
        "duration": 4,
        "icon": "🤲",
    },
    {
        "id": 3,
        "name": "Palm over dorsum",
        "caption": "Right palm over left dorsum with interlaced fingers — then switch.",
        "duration": 4,
        "icon": "✋",
    },
    {
        "id": 4,
        "name": "Fingers interlaced",
        "caption": "Palm to palm with fingers interlaced — rub back and forth.",
        "duration": 4,
        "icon": "🙌",
    },
    {
        "id": 5,
        "name": "Backs of fingers",
        "caption": "Backs of fingers to opposing palms with fingers interlocked.",
        "duration": 4,
        "icon": "👐",
    },
    {
        "id": 6,
        "name": "Rotational — thumb",
        "caption": "Rotational rubbing of left thumb clasped in right palm — then switch.",
        "duration": 4,
        "icon": "👍",
    },
    {
        "id": 7,
        "name": "Rotational — fingertips",
        "caption": "Rotational rubbing of fingertips of right hand in left palm — then switch.",
        "duration": 4,
        "icon": "☝️",
    },
]

WHO_HANDWASH_STEPS = [
    {
        "id": 1,
        "name": "Wet hands",
        "caption": "Wet your hands with clean, running water.",
        "duration": 3,
        "icon": "💧",
    },
    {
        "id": 2,
        "name": "Apply soap",
        "caption": "Apply enough soap to cover all hand surfaces.",
        "duration": 3,
        "icon": "🧴",
    },
    {
        "id": 3,
        "name": "Palm to palm",
        "caption": "Rub hands palm to palm in a circular motion.",
        "duration": 5,
        "icon": "🤲",
    },
    {
        "id": 4,
        "name": "Palm over dorsum",
        "caption": "Right palm over left dorsum with interlaced fingers — then switch.",
        "duration": 5,
        "icon": "✋",
    },
    {
        "id": 5,
        "name": "Fingers interlaced",
        "caption": "Palm to palm with fingers interlaced.",
        "duration": 5,
        "icon": "🙌",
    },
    {
        "id": 6,
        "name": "Backs of fingers",
        "caption": "Backs of fingers to opposing palms with fingers interlocked.",
        "duration": 5,
        "icon": "👐",
    },
    {
        "id": 7,
        "name": "Rotational — thumb",
        "caption": "Rotational rubbing of left thumb in right palm — then switch.",
        "duration": 5,
        "icon": "👍",
    },
    {
        "id": 8,
        "name": "Rotational — fingertips",
        "caption": "Rotational rubbing of fingertips of right hand in left palm — then switch.",
        "duration": 5,
        "icon": "☝️",
    },
    {
        "id": 9,
        "name": "Rinse hands",
        "caption": "Rinse hands well under running water.",
        "duration": 5,
        "icon": "🚿",
    },
    {
        "id": 10,
        "name": "Dry with single-use towel",
        "caption": "Dry hands thoroughly using a single-use towel.",
        "duration": 5,
        "icon": "🧻",
    },
    {
        "id": 11,
        "name": "Turn off tap",
        "caption": "Use the towel to turn off the faucet.",
        "duration": 3,
        "icon": "🚰",
    },
]


# ───────────────────────── MEDIAPIPE HAND ANALYSER ─────────────────────────

class HandAnalyser:
    """Detects hand presence and basic gesture features using MediaPipe Hands."""

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

    def analyse(self, frame_bgr: np.ndarray) -> dict:
        """
        Returns dict with keys:
          annotated_frame, num_hands, hands_touching,
          fingers_interlaced, motion_detected
        """
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
                    annotated,
                    lm,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style(),
                )

            if num_hands == 2:
                lm0 = results.multi_hand_landmarks[0].landmark
                lm1 = results.multi_hand_landmarks[1].landmark

                # Distance between wrist landmarks as a proximity proxy
                w0 = np.array([lm0[0].x, lm0[0].y])
                w1 = np.array([lm1[0].x, lm1[0].y])
                dist = np.linalg.norm(w0 - w1)
                hands_touching = dist < 0.35

                # Interlaced fingers: fingertip x-coords alternate between hands
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

    def validate_step(self, step_id: int, analysis: dict) -> bool:
        """Returns True if the gesture broadly matches the expected step."""
        n = analysis["num_hands"]
        touching = analysis["hands_touching"]
        interlaced = analysis["fingers_interlaced"]

        rules = {
            1: n >= 1,                    # any hand visible
            2: touching,                  # both hands touching
            3: touching,                  # dorsum rubbing — both touching
            4: interlaced,                # fingers interlaced
            5: touching and not interlaced,
            6: touching,                  # thumb rotational
            7: touching,                  # fingertip rotational
            # hand wash extras
            8: touching and not interlaced,
            9: n >= 1,
            10: n >= 1,
            11: n >= 1,
        }
        return rules.get(step_id, n >= 1)


# ─────────────────────── WEBRTC VIDEO PROCESSOR ────────────────────────────

class HygieneVideoProcessor(VideoProcessorBase):
    """Processes each webcam frame, draws overlay, validates current step."""

    def __init__(self):
        self.analyser = HandAnalyser()
        self.current_step_id: int = 1
        self.step_result: Optional[bool] = None   # True=OK, False=error, None=detecting

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        analysis = self.analyser.analyse(img)
        annotated = analysis["annotated_frame"]

        valid = self.analyser.validate_step(self.current_step_id, analysis)
        self.step_result = valid

        # Overlay border — green if correct, red if not
        color = (0, 200, 80) if valid else (60, 60, 240)
        h, w = annotated.shape[:2]
        cv2.rectangle(annotated, (0, 0), (w, h), color, 8)

        # Status text
        label = "Correct posture" if valid else "Adjust hand position"
        cv2.rectangle(annotated, (0, h - 44), (w, h), (0, 0, 0), -1)
        cv2.putText(
            annotated, label,
            (12, h - 16),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65,
            (0, 200, 80) if valid else (60, 80, 240),
            2,
        )

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


# ─────────────────────────── SESSION STATE INIT ────────────────────────────

def init_state():
    defaults = {
        "username": "",
        "mode": "Hand Rub (WHO)",
        "started": False,
        "current_step": 0,
        "step_statuses": {},   # step_id → "pending"|"done"|"error"
        "step_start_time": None,
        "score": 0,
        "session_done": False,
        "theme": "dark",
        "processor": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ──────────────────────────── CSS INJECTION ────────────────────────────────

DARK_CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

  html, body, [data-testid="stAppViewContainer"] {
    background: #0f1117 !important;
    color: #e6e9f0 !important;
    font-family: 'Inter', sans-serif !important;
  }
  [data-testid="stSidebar"] {
    background: #1a1d27 !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
  }
  .step-card {
    background: #1a1d27;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 12px 14px;
    margin-bottom: 8px;
    transition: all 0.2s;
  }
  .step-card.active {
    border-color: #4f8ef7;
    background: rgba(79,142,247,0.06);
    box-shadow: 0 0 0 2px rgba(79,142,247,0.14);
  }
  .step-card.done {
    border-color: #22c97a;
    background: rgba(34,201,122,0.05);
  }
  .step-card.error {
    border-color: #f05454;
    background: rgba(240,84,84,0.05);
  }
  .caption-box {
    background: rgba(79,142,247,0.06);
    border-left: 3px solid #4f8ef7;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 10px 0;
  }
  .score-box {
    background: #1a1d27;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 18px;
    text-align: center;
    margin-top: 16px;
  }
  .score-number {
    font-size: 48px;
    font-weight: 600;
    line-height: 1;
    margin-bottom: 4px;
  }
  .badge {
    display: inline-block;
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 20px;
    font-weight: 500;
  }
  .badge-active { background: rgba(79,142,247,0.15); color: #4f8ef7; }
  .badge-done   { background: rgba(34,201,122,0.15); color: #22c97a; }
  .badge-error  { background: rgba(240,84,84,0.15);  color: #f05454; }
  stButton > button { border-radius: 8px !important; }
</style>
"""

LIGHT_CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

  html, body, [data-testid="stAppViewContainer"] {
    background: #f5f6fa !important;
    color: #1a1d27 !important;
    font-family: 'Inter', sans-serif !important;
  }
  [data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid rgba(0,0,0,0.07) !important;
  }
  .step-card {
    background: #ffffff;
    border: 1px solid rgba(0,0,0,0.07);
    border-radius: 10px;
    padding: 12px 14px;
    margin-bottom: 8px;
  }
  .step-card.active {
    border-color: #2563eb;
    background: rgba(37,99,235,0.04);
    box-shadow: 0 0 0 2px rgba(37,99,235,0.1);
  }
  .step-card.done {
    border-color: #16a34a;
    background: rgba(22,163,74,0.04);
  }
  .step-card.error {
    border-color: #dc2626;
    background: rgba(220,38,38,0.04);
  }
  .caption-box {
    background: rgba(37,99,235,0.04);
    border-left: 3px solid #2563eb;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 10px 0;
  }
  .score-box {
    background: #ffffff;
    border: 1px solid rgba(0,0,0,0.07);
    border-radius: 12px;
    padding: 18px;
    text-align: center;
    margin-top: 16px;
  }
  .score-number {
    font-size: 48px;
    font-weight: 600;
    line-height: 1;
    margin-bottom: 4px;
  }
  .badge { display: inline-block; font-size: 11px; padding: 2px 8px; border-radius: 20px; font-weight: 500; }
  .badge-active { background: rgba(37,99,235,0.1); color: #2563eb; }
  .badge-done   { background: rgba(22,163,74,0.1); color: #16a34a; }
  .badge-error  { background: rgba(220,38,38,0.1); color: #dc2626; }
</style>
"""


# ─────────────────────────── STEP CARD RENDERER ────────────────────────────

def render_step_card(step: dict, status: str, is_current: bool):
    icon = step["icon"]
    name = step["name"]
    caption = step["caption"]

    if is_current:
        css_class = "step-card active"
        badge = '<span class="badge badge-active">Detecting…</span>'
    elif status == "done":
        css_class = "step-card done"
        badge = '<span class="badge badge-done">✓ Done</span>'
    elif status == "error":
        css_class = "step-card error"
        badge = '<span class="badge badge-error">✗ Redo</span>'
    else:
        css_class = "step-card"
        badge = ""

    st.markdown(
        f"""
        <div class="{css_class}">
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <span style="font-size:14px;font-weight:500;">{icon} {name}</span>
            {badge}
          </div>
          <div style="font-size:12px;opacity:0.65;margin-top:4px;">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ───────────────────────────── MAIN APP ────────────────────────────────────

def main():
    init_state()
    st.set_page_config(
        page_title="WHO Hand Hygiene Monitor",
        page_icon="🤲",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Inject CSS
    css = DARK_CSS if st.session_state.theme == "dark" else LIGHT_CSS
    st.markdown(css, unsafe_allow_html=True)

    # ── Sidebar ──────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🤲 Hand Hygiene Monitor")
        st.caption("WHO Protocol — AI Powered")
        st.divider()

        # Theme toggle
        theme_label = "☀️ Switch to Light" if st.session_state.theme == "dark" else "🌙 Switch to Dark"
        if st.button(theme_label, use_container_width=True):
            st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
            st.rerun()

        st.divider()

        # User name
        st.session_state.username = st.text_input(
            "Your name",
            value=st.session_state.username,
            placeholder="e.g. Dr. Ahmed",
        )

        # Mode selection
        st.session_state.mode = st.radio(
            "Protocol",
            ["Hand Rub (WHO)", "Hand Wash (WHO)"],
            index=0 if st.session_state.mode == "Hand Rub (WHO)" else 1,
        )

        steps = WHO_HANDRUB_STEPS if "Rub" in st.session_state.mode else WHO_HANDWASH_STEPS

        st.divider()

        if not st.session_state.started:
            if st.button("▶  Start Session", use_container_width=True, type="primary"):
                if not st.session_state.username.strip():
                    st.error("Please enter your name first.")
                else:
                    st.session_state.started = True
                    st.session_state.current_step = 0
                    st.session_state.step_statuses = {s["id"]: "pending" for s in steps}
                    st.session_state.step_start_time = time.time()
                    st.session_state.session_done = False
                    st.session_state.score = 0
                    st.rerun()
        else:
            if st.button("⏹  Reset", use_container_width=True):
                st.session_state.started = False
                st.session_state.current_step = 0
                st.session_state.step_statuses = {}
                st.session_state.session_done = False
                st.rerun()

        # WHO info
        st.divider()
        st.caption("**WHO reference**")
        st.caption("Hand rub: 20–30 seconds\nHand wash: 40–60 seconds")

    # ── Main area ────────────────────────────────────────────────
    steps = WHO_HANDRUB_STEPS if "Rub" in st.session_state.mode else WHO_HANDWASH_STEPS
    total_steps = len(steps)

    col_cam, col_steps = st.columns([3, 2], gap="large")

    # ── Camera column ────────────────────────────────────────────
    with col_cam:
        if st.session_state.username:
            st.markdown(f"#### Welcome, {st.session_state.username} 👋")
        else:
            st.markdown("#### WHO Hand Hygiene Monitor")

        if not st.session_state.started:
            st.info("Enter your name and press **Start Session** to begin monitoring.")

            # Demo: show snapshot camera to test
            st.caption("Quick test — take a snapshot:")
            snapshot = st.camera_input("Camera test (snapshot mode)")
            if snapshot:
                img_array = np.frombuffer(snapshot.read(), np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                analyser = HandAnalyser()
                result = analyser.analyse(frame)
                st.image(
                    cv2.cvtColor(result["annotated_frame"], cv2.COLOR_BGR2RGB),
                    caption=f"Hands detected: {result['num_hands']}",
                    use_column_width=True,
                )

        else:
            # Live WebRTC stream
            current_step_obj = steps[st.session_state.current_step] if st.session_state.current_step < total_steps else None

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

            # Step caption
            if current_step_obj and not st.session_state.session_done:
                status = st.session_state.step_statuses.get(current_step_obj["id"], "pending")
                color = "#4f8ef7" if status != "error" else "#f05454"
                st.markdown(
                    f"""
                    <div class="caption-box">
                      <div style="font-size:11px;font-weight:600;color:{color};margin-bottom:4px;">
                        STEP {current_step_obj['id']} of {total_steps} — {current_step_obj['name'].upper()}
                      </div>
                      <div style="font-size:14px;">{current_step_obj['icon']} {current_step_obj['caption']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Advance step button (manual override for demo / fallback)
                bcol1, bcol2 = st.columns(2)
                with bcol1:
                    if st.button("✅ Mark step correct", use_container_width=True):
                        _advance_step(steps, current_step_obj["id"], success=True)
                        st.rerun()
                with bcol2:
                    if st.button("❌ Mark step error", use_container_width=True):
                        _advance_step(steps, current_step_obj["id"], success=False)
                        st.rerun()

                # Auto-advance via processor result (polling)
                if ctx.video_processor and ctx.video_processor.step_result is not None:
                    elapsed = time.time() - (st.session_state.step_start_time or time.time())
                    if elapsed >= current_step_obj["duration"]:
                        success = ctx.video_processor.step_result
                        _advance_step(steps, current_step_obj["id"], success=success)
                        st.rerun()

    # ── Steps column ─────────────────────────────────────────────
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

        # Score display
        if st.session_state.started:
            done = sum(1 for s in st.session_state.step_statuses.values() if s == "done")
            errors = sum(1 for s in st.session_state.step_statuses.values() if s == "error")
            score_pct = int((done / total_steps) * 100) if total_steps else 0
            color = "#22c97a" if score_pct >= 80 else ("#f5a623" if score_pct >= 50 else "#f05454")

            st.markdown(
                f"""
                <div class="score-box">
                  <div style="font-size:12px;opacity:0.6;margin-bottom:8px;">SESSION SCORE</div>
                  <div class="score-number" style="color:{color};">{score_pct}%</div>
                  <div style="font-size:12px;opacity:0.65;margin-top:6px;">
                    {done} correct · {errors} errors · {total_steps - done - errors} remaining
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Session done summary
        if st.session_state.session_done:
            done = sum(1 for s in st.session_state.step_statuses.values() if s == "done")
            score_pct = int((done / total_steps) * 100)
            grade = "Excellent" if score_pct == 100 else "Good" if score_pct >= 80 else "Needs Practice"
            st.success(f"**Session complete!** {grade} — {score_pct}% ({done}/{total_steps} steps correct)")
            st.balloons()


# ──────────────────────────── STEP HELPERS ─────────────────────────────────

def _advance_step(steps, step_id: int, success: bool):
    st.session_state.step_statuses[step_id] = "done" if success else "error"
    next_idx = st.session_state.current_step + 1
    if next_idx >= len(steps):
        st.session_state.session_done = True
    else:
        st.session_state.current_step = next_idx
        st.session_state.step_start_time = time.time()


if __name__ == "__main__":
    main()
