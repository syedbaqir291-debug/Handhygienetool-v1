# 🤲 WHO Hand Hygiene Monitor

A premium AI-powered Streamlit app that monitors **WHO hand rub** and **hand wash** technique in real time using your webcam and MediaPipe hand tracking.

---

## Features

- **Real-time webcam monitoring** via `streamlit-webrtc`
- **MediaPipe hand landmark detection** — 21-point hand skeleton per hand
- **WHO Hand Rub (7 steps)** and **WHO Hand Wash (11 steps)** protocols
- **Step-by-step captions** with green ✅ / red ❌ live feedback overlay
- **Scoring** at the end of each session
- **Premium UI** with Light / Dark theme toggle
- **Username** input — personalised session header
- **Manual override buttons** (fallback for testing without live camera)

---

## Quick Start (local)

```bash
# 1. Clone / download this repo
git clone https://github.com/YOUR_USERNAME/hand-hygiene-monitor.git
cd hand-hygiene-monitor

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
streamlit run app.py
```

---

## Deploy to Streamlit Community Cloud (Free)

1. Push this repo to **GitHub** (must be public or you must have a Streamlit account connected)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repo, branch `main`, and entry file `app.py`
4. Click **Deploy** — done!

> **Important:** Streamlit Community Cloud supports `streamlit-webrtc`. Your users must allow camera permissions in their browser.

---

## Project Structure

```
hand-hygiene-monitor/
├── app.py              ← main app
├── requirements.txt    ← dependencies
└── README.md
```

---

## How Detection Works

| Layer | Technology | What it does |
|---|---|---|
| Video capture | `streamlit-webrtc` + WebRTC | Streams live webcam frames |
| Hand tracking | `mediapipe.solutions.hands` | Detects 21 landmarks per hand |
| Step validation | Custom rule engine | Checks proximity, interlacing, posture |
| Overlay | OpenCV | Draws skeleton + colored border |

### Step Validation Rules (Hand Rub)

| Step | Gesture signal required |
|---|---|
| 1 Apply product | Any hand visible |
| 2 Palm to palm | Both hands close together |
| 3 Palm over dorsum | Both hands touching |
| 4 Fingers interlaced | Fingertips from both hands alternating (interlaced) |
| 5 Backs of fingers | Hands touching, fingers not fully interlaced |
| 6 Rotational thumb | Both hands touching |
| 7 Rotational fingertips | Both hands touching |

---

## Customisation

- **Add/edit steps:** modify `WHO_HANDRUB_STEPS` or `WHO_HANDWASH_STEPS` in `app.py`
- **Tune detection confidence:** adjust `min_detection_confidence` in `HandAnalyser.__init__`
- **Change step duration:** edit `"duration"` field in each step dict
- **Theme colours:** edit the CSS blocks `DARK_CSS` / `LIGHT_CSS` in `app.py`

---

## Known Limitations

- Detection rules are heuristic-based (proximity + interlacing geometry). For clinical-grade validation, a trained gesture classifier on WHO-labelled video data would be needed.
- `streamlit-webrtc` requires a STUN/TURN server. The default Google STUN server is used, which works for most connections but may fail behind strict corporate firewalls.

---

## WHO References

- [WHO Hand Rub Technique Poster](https://www.who.int/gpsc/5may/Hand_Rub_Technique_MULTI.pdf)
- [WHO Hand Wash Technique Poster](https://www.who.int/gpsc/5may/Hand_Wash_Technique_MULTI.pdf)
