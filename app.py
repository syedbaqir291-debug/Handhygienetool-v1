"""
WHO Hand Hygiene Monitor
Developed by OMAC Developer — S M Baqir
Dependencies: streamlit, fpdf2  (nothing else)
"""

import streamlit as st
import datetime
import io
from fpdf import FPDF

# ── WHO STEPS ────────────────────────────────────────────────────────────────

HANDRUB = [
    {"id":1, "icon":"💧", "name":"Apply product",           "caption":"Apply a palmful of alcohol-based product into a cupped hand."},
    {"id":2, "icon":"🤲", "name":"Palm to palm",            "caption":"Rub hands palm to palm in a circular motion."},
    {"id":3, "icon":"✋", "name":"Palm over dorsum",         "caption":"Right palm over left dorsum with interlaced fingers — then switch."},
    {"id":4, "icon":"🙌", "name":"Fingers interlaced",      "caption":"Palm to palm with fingers interlaced — rub back and forth."},
    {"id":5, "icon":"👐", "name":"Backs of fingers",        "caption":"Backs of fingers to opposing palms with fingers interlocked."},
    {"id":6, "icon":"👍", "name":"Rotational — thumb",      "caption":"Rotational rubbing of left thumb clasped in right palm — then switch."},
    {"id":7, "icon":"☝️", "name":"Rotational — fingertips", "caption":"Rotational rubbing of fingertips of right hand in left palm — then switch."},
]

HANDWASH = [
    {"id":1,  "icon":"💧", "name":"Wet hands",               "caption":"Wet your hands with clean, running water."},
    {"id":2,  "icon":"🧴", "name":"Apply soap",              "caption":"Apply enough soap to cover all hand surfaces."},
    {"id":3,  "icon":"🤲", "name":"Palm to palm",            "caption":"Rub hands palm to palm."},
    {"id":4,  "icon":"✋", "name":"Palm over dorsum",         "caption":"Right palm over left dorsum with interlaced fingers — then switch."},
    {"id":5,  "icon":"🙌", "name":"Fingers interlaced",      "caption":"Palm to palm with fingers interlaced."},
    {"id":6,  "icon":"👐", "name":"Backs of fingers",        "caption":"Backs of fingers to opposing palms with fingers interlocked."},
    {"id":7,  "icon":"👍", "name":"Rotational — thumb",      "caption":"Rotational rubbing of left thumb in right palm — then switch."},
    {"id":8,  "icon":"☝️", "name":"Rotational — fingertips", "caption":"Rotational rubbing of fingertips of right hand in left palm — then switch."},
    {"id":9,  "icon":"🚿", "name":"Rinse hands",             "caption":"Rinse hands well under running water."},
    {"id":10, "icon":"🧻", "name":"Dry with towel",          "caption":"Dry hands thoroughly using a single-use towel."},
    {"id":11, "icon":"🚰", "name":"Turn off tap",            "caption":"Use the towel to turn off the faucet."},
]

# ── SESSION STATE ─────────────────────────────────────────────────────────────

def init():
    defaults = {
        "full_name": "", "emp_code": "", "mode": "Hand Rub",
        "started": False, "step": 0,
        "statuses": {},   # step_id → "done" | "error"
        "done": False, "theme": "dark",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# ── PDF ───────────────────────────────────────────────────────────────────────

def make_pdf(name, code, mode, statuses, steps):
    done  = sum(1 for v in statuses.values() if v == "done")
    score = int(done / len(steps) * 100)
    errs  = sum(1 for v in statuses.values() if v == "error")

    pdf = FPDF()
    pdf.add_page()

    # Banner
    pdf.set_fill_color(15, 82, 186)
    pdf.rect(0, 0, 210, 26, "F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_xy(10, 6)
    pdf.cell(190, 8, "WHO Hand Hygiene Compliance Report", align="C")
    pdf.set_font("Helvetica", "", 8)
    pdf.set_xy(10, 16)
    pdf.cell(190, 6, "OMAC Developer  |  Developed by S M Baqir", align="C")

    # Info
    pdf.set_text_color(30, 30, 30)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_xy(10, 32)
    pdf.cell(65, 7, f"Name: {name}")
    pdf.cell(55, 7, f"Code: {code}")
    pdf.cell(70, 7, f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    pdf.set_xy(10, 40)
    pdf.cell(65, 7, f"Protocol: {mode}")

    # Score
    sc = (22,163,74) if score >= 80 else (217,119,6) if score >= 50 else (220,38,38)
    pdf.set_fill_color(*sc)
    pdf.set_text_color(255,255,255)
    pdf.set_font("Helvetica","B",18)
    pdf.set_xy(10, 50)
    pdf.cell(190, 14, f"Score: {score}%   |   {'PASS' if score>=80 else 'FAIL'}", align="C", fill=True)

    # Table
    pdf.set_text_color(30,30,30)
    pdf.set_font("Helvetica","B",9)
    pdf.set_xy(10, 70)
    pdf.set_fill_color(210,225,255)
    pdf.cell(10,7,"#",border=1,fill=True,align="C")
    pdf.cell(55,7,"Step",border=1,fill=True)
    pdf.cell(105,7,"Instruction",border=1,fill=True)
    pdf.cell(20,7,"Result",border=1,fill=True,align="C")
    pdf.ln()

    pdf.set_font("Helvetica","",9)
    for i, s in enumerate(steps):
        st_val = statuses.get(s["id"], "pending")
        res    = "OK" if st_val=="done" else ("ERR" if st_val=="error" else "—")
        pdf.set_fill_color(220,252,231) if st_val=="done" else (
            pdf.set_fill_color(254,226,226) if st_val=="error" else
            pdf.set_fill_color(249,250,251))
        cap = s["caption"][:72]+"…" if len(s["caption"])>72 else s["caption"]
        pdf.cell(10,7,str(i+1),border=1,fill=True,align="C")
        pdf.cell(55,7,s["name"],border=1,fill=True)
        pdf.cell(105,7,cap,border=1,fill=True)
        pdf.set_text_color(22,163,74) if st_val=="done" else (
            pdf.set_text_color(220,38,38) if st_val=="error" else
            pdf.set_text_color(120,120,120))
        pdf.cell(20,7,res,border=1,fill=True,align="C")
        pdf.set_text_color(30,30,30)
        pdf.ln()

    pdf.ln(4)
    pdf.set_font("Helvetica","B",10)
    pdf.cell(65,7,f"Correct: {done}/{len(steps)}")
    pdf.cell(65,7,f"Errors: {errs}")
    pdf.cell(60,7,f"{'PASS ✓' if score>=80 else 'FAIL ✗'}")

    pdf.set_y(-13)
    pdf.set_font("Helvetica","I",8)
    pdf.set_text_color(140,140,140)
    pdf.cell(190,6,"Developed by OMAC Developer — S M Baqir  |  who.int/gpsc/hand-hygiene",align="C")

    buf = io.BytesIO()
    pdf.output(buf)
    buf.seek(0)
    return buf.read(), score

# ── CSS ───────────────────────────────────────────────────────────────────────

def inject_css(dark):
    if dark:
        bg,surf,bdr,txt,txt2,acc,ok,err = (
            "#0f1117","#1a1d27","rgba(255,255,255,0.08)",
            "#e6e9f0","#9ca3b8","#4f8ef7","#22c97a","#f05454")
    else:
        bg,surf,bdr,txt,txt2,acc,ok,err = (
            "#f5f6fa","#ffffff","rgba(0,0,0,0.08)",
            "#1a1d27","#6b7280","#2563eb","#16a34a","#dc2626")

    st.markdown(f"""<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    html,body,[data-testid="stAppViewContainer"]{{
        background:{bg}!important;color:{txt}!important;
        font-family:'Inter',sans-serif!important}}
    [data-testid="stSidebar"]{{
        background:{surf}!important;border-right:1px solid {bdr}!important}}
    .scard{{background:{surf};border:1px solid {bdr};border-radius:10px;
            padding:10px 13px;margin-bottom:7px}}
    .scard.active{{border-color:{acc};
        background:rgba(79,142,247,0.07);
        box-shadow:0 0 0 2px rgba(79,142,247,0.15)}}
    .scard.done {{border-color:{ok}; background:rgba(34,201,122,0.06)}}
    .scard.error{{border-color:{err};background:rgba(240,84,84,0.06)}}
    .caption-box{{background:rgba(79,142,247,0.07);border-left:3px solid {acc};
                  border-radius:8px;padding:11px 15px;margin:10px 0}}
    .score-box{{background:{surf};border:1px solid {bdr};border-radius:12px;
                padding:18px;text-align:center;margin-top:12px}}
    .footer{{position:fixed;bottom:0;left:0;right:0;background:{surf};
             border-top:1px solid {bdr};padding:7px 0;text-align:center;
             font-size:12px;color:{txt2};z-index:999}}
    .badge{{display:inline-block;font-size:11px;padding:2px 8px;
            border-radius:20px;font-weight:500}}
    .ba{{background:rgba(79,142,247,0.15);color:{acc}}}
    .bd{{background:rgba(34,201,122,0.15);color:{ok}}}
    .be{{background:rgba(240,84,84,0.15);color:{err}}}
    div[data-testid="stVerticalBlock"]{{padding-bottom:46px}}
    </style>""", unsafe_allow_html=True)

# ── STEP CARD ─────────────────────────────────────────────────────────────────

def step_card(s, status, is_cur):
    cls   = "scard " + ("active" if is_cur else status if status in ("done","error") else "")
    badge = ('<span class="badge ba">▶ Current</span>'   if is_cur    else
             '<span class="badge bd">✓ Done</span>'      if status=="done"  else
             '<span class="badge be">✗ Error</span>'     if status=="error" else "")
    st.markdown(f"""<div class="{cls}">
      <div style="display:flex;justify-content:space-between;align-items:center">
        <span style="font-size:13px;font-weight:500">{s['icon']} {s['name']}</span>{badge}
      </div>
      <div style="font-size:11px;opacity:.6;margin-top:3px">{s['caption']}</div>
    </div>""", unsafe_allow_html=True)

# ── APP ───────────────────────────────────────────────────────────────────────

def main():
    init()
    st.set_page_config(page_title="WHO Hand Hygiene Monitor",
                       page_icon="🤲", layout="wide")
    dark = st.session_state.theme == "dark"
    inject_css(dark)

    steps = HANDRUB if st.session_state.mode == "Hand Rub" else HANDWASH
    total = len(steps)

    # ── SIDEBAR ──────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🤲 Hand Hygiene Monitor")
        st.caption("WHO Protocol · Compliance Tool")
        st.divider()

        if st.button("☀️ Light mode" if dark else "🌙 Dark mode",
                     use_container_width=True):
            st.session_state.theme = "light" if dark else "dark"
            st.rerun()

        st.divider()
        st.session_state.full_name = st.text_input(
            "Full Name", value=st.session_state.full_name,
            placeholder="e.g. Ahmed Ali Khan")
        st.session_state.emp_code = st.text_input(
            "Employee Code", value=st.session_state.emp_code,
            placeholder="e.g. EMP-0042")
        st.session_state.mode = st.radio(
            "Protocol", ["Hand Rub", "Hand Wash"],
            index=0 if st.session_state.mode == "Hand Rub" else 1)

        steps = HANDRUB if st.session_state.mode == "Hand Rub" else HANDWASH
        total = len(steps)

        st.divider()
        if not st.session_state.started:
            if st.button("▶  Start Session", use_container_width=True,
                         type="primary"):
                if not st.session_state.full_name.strip():
                    st.error("Enter your full name.")
                elif not st.session_state.emp_code.strip():
                    st.error("Enter employee code.")
                else:
                    st.session_state.started  = True
                    st.session_state.step     = 0
                    st.session_state.statuses = {s["id"]:"pending" for s in steps}
                    st.session_state.done     = False
                    st.rerun()
        else:
            if st.button("⏹  Reset", use_container_width=True):
                st.session_state.started  = False
                st.session_state.step     = 0
                st.session_state.statuses = {}
                st.session_state.done     = False
                st.rerun()

        st.divider()
        st.caption("**WHO reference**\nHand rub: 20–30 sec\nHand wash: 40–60 sec")

    # ── COLUMNS ──────────────────────────────────────────────────
    left, right = st.columns([3, 2], gap="large")

    with left:
        if st.session_state.full_name:
            st.markdown(f"#### Welcome, {st.session_state.full_name}")
            st.caption(f"Employee Code: **{st.session_state.emp_code}**")
        else:
            st.markdown("#### WHO Hand Hygiene Monitor")

        if not st.session_state.started:
            st.info("Enter your details in the sidebar and press **Start Session**.")
            st.image("https://www.who.int/images/default-source/health-topics/hand-hygiene/handrub-wash.png",
                     caption="WHO Hand Hygiene — 5 Moments", use_container_width=True)

        else:
            idx = st.session_state.step
            cur = steps[idx] if idx < total else None

            if cur and not st.session_state.done:
                acc = "#4f8ef7" if dark else "#2563eb"
                st.markdown(f"""<div class="caption-box">
                  <div style="font-size:11px;font-weight:600;color:{acc};margin-bottom:4px">
                    STEP {cur['id']} of {total} — {cur['name'].upper()}
                  </div>
                  <div style="font-size:15px">{cur['icon']}&nbsp; {cur['caption']}</div>
                </div>""", unsafe_allow_html=True)

                # Camera snapshot
                snap = st.camera_input("📷 Point camera at your hands and take a snapshot",
                                       key=f"cam_{idx}_{cur['id']}")

                if snap:
                    st.image(snap, caption="Your snapshot", use_container_width=True)
                    st.markdown("---")
                    st.markdown("**Did you perform this step correctly?**")
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("✅  Yes — step done correctly",
                                     use_container_width=True, type="primary"):
                            _next(steps, cur["id"], True)
                            st.rerun()
                    with c2:
                        if st.button("❌  No — mark as error",
                                     use_container_width=True):
                            _next(steps, cur["id"], False)
                            st.rerun()
                else:
                    st.caption("Take a snapshot, then mark the step correct or error.")

                # Progress bar
                done_so_far = sum(1 for v in st.session_state.statuses.values()
                                  if v in ("done","error"))
                st.progress(done_so_far / total,
                            text=f"Progress: {done_so_far}/{total} steps")

    with right:
        st.markdown("#### Steps Checklist")

        for i, s in enumerate(steps):
            status = st.session_state.statuses.get(s["id"], "pending")
            is_cur = (st.session_state.started
                      and i == st.session_state.step
                      and not st.session_state.done)
            step_card(s, status, is_cur)

        if st.session_state.started:
            done_n  = sum(1 for v in st.session_state.statuses.values() if v=="done")
            err_n   = sum(1 for v in st.session_state.statuses.values() if v=="error")
            pct     = int(done_n / total * 100)
            col     = "#22c97a" if pct>=80 else ("#f5a623" if pct>=50 else "#f05454")
            st.markdown(f"""<div class="score-box">
              <div style="font-size:11px;opacity:.6;margin-bottom:6px">SESSION SCORE</div>
              <div style="font-size:44px;font-weight:600;color:{col};line-height:1">{pct}%</div>
              <div style="font-size:12px;opacity:.6;margin-top:6px">
                {done_n} correct &nbsp;·&nbsp; {err_n} errors &nbsp;·&nbsp; {total-done_n-err_n} remaining
              </div>
            </div>""", unsafe_allow_html=True)

        # Session complete
        if st.session_state.done:
            done_n = sum(1 for v in st.session_state.statuses.values() if v=="done")
            pct    = int(done_n / total * 100)
            grade  = "Excellent 🏆" if pct==100 else "Good 👍" if pct>=80 else "Needs Practice ⚠️"
            st.success(f"**Session complete!** {grade} — {pct}%")
            st.balloons()

            pdf_bytes, _ = make_pdf(
                st.session_state.full_name, st.session_state.emp_code,
                st.session_state.mode, st.session_state.statuses, steps)
            fname = (f"HH_{st.session_state.emp_code}_"
                     f"{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf")
            st.download_button("📥  Download PDF Report", pdf_bytes, fname,
                               mime="application/pdf",
                               use_container_width=True, type="primary")

    # Footer
    st.markdown(
        '<div class="footer">OMAC Developer &nbsp;|&nbsp; '
        'Developed by <b>S M Baqir</b> &nbsp;|&nbsp; '
        'WHO Hand Hygiene Monitoring Tool</div>',
        unsafe_allow_html=True)


def _next(steps, step_id, success):
    st.session_state.statuses[step_id] = "done" if success else "error"
    nxt = st.session_state.step + 1
    if nxt >= len(steps):
        st.session_state.done = True
    else:
        st.session_state.step = nxt


if __name__ == "__main__":
    main()
