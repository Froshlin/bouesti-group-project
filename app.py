import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model.joblib"

st.set_page_config(page_title="Cardio Risk Classifier", page_icon="+", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
    :root {
        --ink: #0d2238;
        --line: #dbe8f4;
        --card: #ffffff;
    }
    .stApp {
        font-family: 'Space Grotesk', sans-serif;
        background:
            radial-gradient(circle at 10% 10%, #d9eef8 0%, transparent 36%),
            radial-gradient(circle at 90% 20%, #fef3c7 0%, transparent 32%),
            linear-gradient(180deg, #eef7ff 0%, #f8fbfd 100%);
        color: var(--ink);
    }
    .hero {
        background: linear-gradient(135deg, #0f766e 0%, #0d2238 100%);
        border-radius: 18px;
        padding: 1.2rem 1.4rem;
        box-shadow: 0 12px 30px rgba(13, 34, 56, 0.22);
        color: #ffffff;
        margin-bottom: 0.9rem;
    }
    .panel {
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 1rem;
        box-shadow: 0 8px 24px rgba(25, 51, 85, 0.08);
    }
    .question {
        margin: 0.2rem 0 0.25rem 0;
        font-size: 0.92rem;
        font-weight: 600;
        color: #102a43;
    }
    .stTextInput label, .stSelectbox label {
        color: #102a43 !important;
        font-weight: 600 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>Cardio Risk Classifier (SVM)</h1>
      <p>Final class is built from two risk models: heart disease risk + stroke risk.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_payload(path: Path):
    return joblib.load(path)


if not MODEL_PATH.exists():
    st.error("`model.joblib` not found. Run `python train_model.py` first.")
    st.stop()

try:
    payload = load_payload(MODEL_PATH)
except Exception as exc:
    st.error("Failed to load `model.joblib`.")
    st.exception(exc)
    st.stop()

if payload.get("version") != 2:
    st.error("Model format is old. Please run `python train_model.py` to regenerate the model.")
    st.stop()

heart_model = payload["heart_model"]
stroke_model = payload["stroke_model"]
heart_features = payload["heart_features"]
stroke_features = payload["stroke_features"]
thresholds = payload["thresholds"]

heart_threshold = float(thresholds.get("heart", 0.5))
stroke_threshold = float(thresholds.get("stroke", 0.5))

with st.sidebar:
    st.subheader("System status")
    st.write(f"Model path: `{MODEL_PATH}`")
    st.write(f"Model format: `v{payload.get('version')}`")
    st.write(f"Heart threshold: `{heart_threshold:.2f}`")
    st.write(f"Stroke threshold: `{stroke_threshold:.2f}`")


def parse_value(raw: str, caster):
    text = raw.strip()
    if text == "":
        return np.nan
    try:
        return caster(float(text)) if caster is int else caster(text)
    except ValueError:
        return np.nan


left, right = st.columns(2)

with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Demographics")
    st.markdown('<p class="question">1) What is the patient age in years?</p>', unsafe_allow_html=True)
    age = st.text_input("Patient age (years)", placeholder="e.g., 54")
    st.markdown('<p class="question">2) What is the patient biological sex/gender?</p>', unsafe_allow_html=True)
    gender = st.selectbox("Biological sex / gender", ["Unknown", "Male", "Female", "Other"])
    st.subheader("History")
    st.markdown('<p class="question">3) Has the patient been diagnosed with hypertension?</p>', unsafe_allow_html=True)
    hypertension = st.selectbox("Hypertension diagnosis", ["Unknown", "0", "1"])
    st.markdown('<p class="question">4) Has the patient ever been married?</p>', unsafe_allow_html=True)
    ever_married = st.selectbox("Ever married", ["Unknown", "Yes", "No"])
    st.markdown('<p class="question">5) What is the patient occupation type?</p>', unsafe_allow_html=True)
    work_type = st.selectbox(
        "Occupation type", ["Unknown", "Private", "Self-employed", "Govt_job", "children", "Never_worked"]
    )
    st.markdown('<p class="question">6) Does the patient live in an urban or rural area?</p>', unsafe_allow_html=True)
    residence_type = st.selectbox("Residence area type", ["Unknown", "Urban", "Rural"])
    st.markdown('<p class="question">7) What is the patient smoking status?</p>', unsafe_allow_html=True)
    smoking_status = st.selectbox("Smoking status", ["Unknown", "never smoked", "formerly smoked", "smokes"])
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Clinical Inputs")
    st.markdown('<p class="question">8) What is the chest pain type code (cp)?</p>', unsafe_allow_html=True)
    cp = st.text_input("Chest pain type (cp code)", placeholder="0-3")
    st.markdown('<p class="question">9) What is the resting blood pressure in mmHg?</p>', unsafe_allow_html=True)
    trestbps = st.text_input("Resting blood pressure (trestbps, mmHg)", placeholder="e.g., 120")
    st.markdown('<p class="question">10) What is the serum cholesterol value in mg/dL?</p>', unsafe_allow_html=True)
    chol = st.text_input("Serum cholesterol (chol, mg/dL)", placeholder="e.g., 240")
    st.markdown('<p class="question">11) Is fasting blood sugar elevated (fbs code 0 or 1)?</p>', unsafe_allow_html=True)
    fbs = st.text_input("Fasting blood sugar flag (fbs code)", placeholder="0 or 1")
    st.markdown('<p class="question">12) What is the resting ECG result code?</p>', unsafe_allow_html=True)
    restecg = st.text_input("Resting ECG result (restecg code)", placeholder="0-2")
    st.markdown('<p class="question">13) What is the maximum heart rate achieved?</p>', unsafe_allow_html=True)
    thalach = st.text_input("Maximum heart rate achieved (thalach)", placeholder="e.g., 150")
    st.markdown('<p class="question">14) Did exercise induce angina (exang code 0 or 1)?</p>', unsafe_allow_html=True)
    exang = st.text_input("Exercise-induced angina (exang code)", placeholder="0 or 1")
    st.markdown('<p class="question">15) What is the ST depression value (oldpeak)?</p>', unsafe_allow_html=True)
    oldpeak = st.text_input("ST depression (oldpeak)", placeholder="e.g., 1.2")
    st.markdown('<p class="question">16) What is the ST segment slope code?</p>', unsafe_allow_html=True)
    slope = st.text_input("ST segment slope (slope code)", placeholder="0-2")
    st.markdown('<p class="question">17) How many major vessels are colored (ca code)?</p>', unsafe_allow_html=True)
    ca = st.text_input("Major vessels count (ca code)", placeholder="0-4")
    st.markdown('<p class="question">18) What is the thalassemia test code (thal)?</p>', unsafe_allow_html=True)
    thal = st.text_input("Thalassemia result (thal code)", placeholder="0-3")
    st.markdown('<p class="question">19) What is the average glucose level in mg/dL?</p>', unsafe_allow_html=True)
    avg_glucose_level = st.text_input("Average glucose level (mg/dL)", placeholder="e.g., 95.0")
    st.markdown('<p class="question">20) What is the patient BMI?</p>', unsafe_allow_html=True)
    bmi = st.text_input("Body Mass Index (BMI)", placeholder="e.g., 27.5")
    st.markdown("</div>", unsafe_allow_html=True)

if st.button("Predict", type="primary", use_container_width=True):
    data = {
        "age": parse_value(age, float),
        "gender": None if gender == "Unknown" else gender,
        "cp": parse_value(cp, int),
        "trestbps": parse_value(trestbps, float),
        "chol": parse_value(chol, float),
        "fbs": parse_value(fbs, int),
        "restecg": parse_value(restecg, int),
        "thalach": parse_value(thalach, float),
        "exang": parse_value(exang, int),
        "oldpeak": parse_value(oldpeak, float),
        "slope": parse_value(slope, int),
        "ca": parse_value(ca, int),
        "thal": parse_value(thal, int),
        "hypertension": None if hypertension == "Unknown" else int(hypertension),
        "ever_married": None if ever_married == "Unknown" else ever_married,
        "work_type": None if work_type == "Unknown" else work_type,
        "Residence_type": None if residence_type == "Unknown" else residence_type,
        "avg_glucose_level": parse_value(avg_glucose_level, float),
        "bmi": parse_value(bmi, float),
        "smoking_status": None if smoking_status == "Unknown" else smoking_status,
    }

    try:
        heart_df = pd.DataFrame([data])[heart_features]
        heart_classes = list(heart_model.named_steps["clf"].classes_)
        heart_idx = heart_classes.index(1)
        heart_prob = float(heart_model.predict_proba(heart_df)[0][heart_idx])
        heart_flag = int(heart_prob >= heart_threshold)

        stroke_data = dict(data)
        stroke_data["heart_disease"] = heart_flag
        stroke_df = pd.DataFrame([stroke_data])[stroke_features]
        stroke_classes = list(stroke_model.named_steps["clf"].classes_)
        stroke_idx = stroke_classes.index(1)
        stroke_prob = float(stroke_model.predict_proba(stroke_df)[0][stroke_idx])
        stroke_flag = int(stroke_prob >= stroke_threshold)

        if heart_flag == 1 and stroke_flag == 1:
            final_class = "both"
        elif heart_flag == 1 and stroke_flag == 0:
            final_class = "heart_disease"
        elif heart_flag == 0 and stroke_flag == 1:
            final_class = "stroke"
        else:
            final_class = "neither"

        st.success(f"Prediction: {final_class}")
        st.write("Intermediate risks")
        st.write(f"- `heart disease probability`: `{heart_prob:.3f}` (threshold `{heart_threshold:.2f}`)")
        st.write(f"- `stroke probability`: `{stroke_prob:.3f}` (threshold `{stroke_threshold:.2f}`)")

        with st.expander("Input sent to model"):
            st.dataframe(pd.DataFrame([data]))
    except Exception as exc:
        st.error("Prediction failed.")
        st.exception(exc)

st.caption("For educational use only. Not for clinical diagnosis.")
