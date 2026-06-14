import streamlit as st
import pandas as pd
import pickle
import os

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="Mental Tiredness Predictor",
    page_icon="🧠",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
.stApp { background: #f0f4f8; }
header[data-testid="stHeader"] { background: transparent; }

.hero {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 20px;
    padding: 2.5rem 2rem;
    text-align: center;
    margin-bottom: 2rem;
    color: white;
}
.hero h1 { font-size: 2.8rem; font-weight: 800; margin: 0; letter-spacing: -1px; }
.hero p  { font-size: 1.1rem; opacity: 0.85; margin: 0.5rem 0 0; }

.section-card {
    background: white;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}
.section-title {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #667eea;
    margin-bottom: 1rem;
}

.stSlider > div > div > div { background: #667eea !important; }

div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2rem !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    width: 100% !important;
    transition: transform 0.1s;
    box-shadow: 0 4px 15px rgba(102,126,234,0.4) !important;
}
div[data-testid="stButton"] > button:hover { transform: translateY(-2px); }

.result-box {
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin-top: 1.5rem;
}
.result-low    { background: #d4edda; border: 2px solid #28a745; }
.result-medium { background: #fff3cd; border: 2px solid #ffc107; }
.result-high   { background: #f8d7da; border: 2px solid #dc3545; }
.result-score  { font-size: 4rem; font-weight: 800; line-height: 1; }
.result-label  { font-size: 1.2rem; font-weight: 600; margin-top: 0.5rem; }
.result-sub    { font-size: 0.9rem; opacity: 0.75; margin-top: 0.25rem; }

.gauge-wrap { background: #e9ecef; border-radius: 50px; height: 14px; margin-top: 1rem; overflow: hidden; }
.gauge-fill { height: 100%; border-radius: 50px; }

label { font-weight: 600 !important; color: #2d3748 !important; }
.stSelectbox > div > div { border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────
model_path = "model.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)

# ── Read exact category values the encoder was trained on ─
preprocessor = model.named_steps["Preprocessor"]
encoder      = preprocessor.transformers_[1][1]          # OneHotEncoder
cat_cols     = preprocessor.transformers_[1][2]          # list of cat column names

# Build a dict: column_name -> [list of known categories]
cat_options = {col: list(cats) for col, cats in zip(cat_cols, encoder.categories_)}

# ── Hero ──────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🧠 Mental Tiredness Predictor</h1>
    <p>Fill in your daily details to predict your cognitive fatigue score</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# INPUT SECTIONS
# ══════════════════════════════════════════════════════════
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:

    # ── Cognitive Load ─────────────────────────────────────
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🧩 Cognitive Load</div>', unsafe_allow_html=True)
    number_of_decisions_made = st.slider("Decisions Made",         0, 100, 10)
    context_switch_count     = st.slider("Context Switches",       0, 50,  5)
    notifications_received   = st.slider("Notifications Received", 0, 200, 30)
    task_complexity_avg      = st.slider("Task Complexity (1–10)", 1, 10,  5)
    workload_score           = st.slider("Workload Score (1–10)",  1, 10,  6)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Work Context ───────────────────────────────────────
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">💼 Work Context</div>', unsafe_allow_html=True)
    screen_time_min  = st.slider("Screen Time (min)", 0, 720, 360)
    deep_work_min    = st.slider("Deep Work (min)",   0, 480, 90)
    break_frequency  = st.slider("Breaks Per Day",    0, 20,  3)
    work_type        = st.selectbox("Work Type",        cat_options.get("work_type",        ["Technical","Creative","Administrative","Managerial","Repetitive"]))
    work_environment = st.selectbox("Work Environment", cat_options.get("work_environment", ["Remote","Office","Hybrid","Outdoor","Co-working"]))
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:

    # ── Sleep & Recovery ───────────────────────────────────
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">😴 Sleep & Recovery</div>', unsafe_allow_html=True)
    sleep_hours    = st.slider("Sleep Hours",        0.0, 12.0, 7.0, step=0.5)
    deep_sleep_pct = st.slider("Deep Sleep (%)",     0,   100,  20)
    hydration_l    = st.slider("Hydration (litres)", 0.0, 5.0,  2.0, step=0.25)
    caffeine_mg    = st.slider("Caffeine (mg)",      0,   800,  200, step=25)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Environment & Mood ─────────────────────────────────
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🌡️ Environment & Mood</div>', unsafe_allow_html=True)
    noise_level_db = st.slider("Noise Level (dB)",  0,  120, 45)
    temperature_c  = st.slider("Temperature (°C)", -10, 50,  22)
    mood_options   = cat_options.get("mood", ["Very Bad","Bad","Neutral","Good","Excellent"])
    mood           = st.select_slider("Mood", options=mood_options,
                                      value=mood_options[len(mood_options)//2])
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    predict = st.button("🔍 Predict Mental Tiredness Score")

# ══════════════════════════════════════════════════════════
# PREDICTION OUTPUT
# ══════════════════════════════════════════════════════════
if predict:
    input_data = pd.DataFrame([{
        "number_of_decisions_made": number_of_decisions_made,
        "context_switch_count":     context_switch_count,
        "notifications_received":   notifications_received,
        "screen_time_min":          screen_time_min,
        "deep_work_min":            deep_work_min,
        "task_complexity_avg":      task_complexity_avg,
        "caffeine_mg":              caffeine_mg,
        "break_frequency":          break_frequency,
        "sleep_hours":              sleep_hours,
        "deep_sleep_pct":           deep_sleep_pct,
        "hydration_l":              hydration_l,
        "mood":                     str(mood),              # ← raw string, NOT encoded
        "work_type":                str(work_type),         # ← raw string, NOT encoded
        "work_environment":         str(work_environment),  # ← raw string, NOT encoded
        "noise_level_db":           noise_level_db,
        "temperature_c":            temperature_c,
        "workload_score":           workload_score,
    }])

    prediction = model.predict(input_data)[0]
    score = round(float(prediction), 1)
    pct   = int((score / 10) * 100)

    if score <= 3:
        css_class   = "result-low"
        emoji       = "🟢"
        label       = "Low Tiredness"
        sub         = "You're in great shape mentally. Keep it up!"
        gauge_color = "#28a745"
    elif score <= 6:
        css_class   = "result-medium"
        emoji       = "🟡"
        label       = "Moderate Tiredness"
        sub         = "Take a break soon. Consider a short walk or rest."
        gauge_color = "#ffc107"
    else:
        css_class   = "result-high"
        emoji       = "🔴"
        label       = "High Tiredness"
        sub         = "Rest is essential. Step away and recover."
        gauge_color = "#dc3545"

    st.markdown(f"""
    <div class="result-box {css_class}">
        <div class="result-score">{emoji} {score}<span style="font-size:1.5rem">/10</span></div>
        <div class="result-label">{label}</div>
        <div class="result-sub">{sub}</div>
        <div class="gauge-wrap">
            <div class="gauge-fill" style="width:{pct}%; background:{gauge_color};"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("😴 Sleep",      f"{sleep_hours}h",    delta=f"{sleep_hours-7:.1f}h vs ideal")
    c2.metric("☕ Caffeine",    f"{caffeine_mg}mg")
    c3.metric("📱 Screen Time", f"{screen_time_min}m")
    c4.metric("🧠 Workload",    f"{workload_score}/10")