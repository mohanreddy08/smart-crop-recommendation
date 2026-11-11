"""Streamlit web interface for the Smart Crop Recommendation System."""
from pathlib import Path
import pickle
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Smart Crop Recommendation", page_icon="🌾", layout="centered")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "crop_model.pkl"
FIGS_DIR = BASE_DIR / "figs"

# Valid range of every input (taken from the training data)
RANGES = {
    "N": (0, 140),
    "P": (5, 145),
    "K": (5, 205),
    "temperature": (8.0, 44.0),
    "ph": (3.5, 10.0),
    "rainfall": (20.0, 300.0),
}
LABELS = {
    "N": "Nitrogen (N)",
    "P": "Phosphorus (P)",
    "K": "Potassium (K)",
    "temperature": "Temperature (°C)",
    "ph": "Soil pH",
    "rainfall": "Rainfall (mm)",
}

PRESETS = {
    "Select a preset...": None,
    "Rice (High rainfall & balanced NPK)": {"N": 90, "P": 42, "K": 43, "temperature": 21.0, "ph": 6.5, "rainfall": 202.0},
    "Coffee (High rainfall & moderate temp)": {"N": 100, "P": 20, "K": 30, "temperature": 25.0, "ph": 6.8, "rainfall": 160.0},
    "Cotton (High temperature & lower rainfall)": {"N": 120, "P": 40, "K": 20, "temperature": 24.0, "ph": 7.0, "rainfall": 80.0},
    "Apple (Cool temperature & high potassium)": {"N": 20, "P": 130, "K": 200, "temperature": 22.0, "ph": 6.0, "rainfall": 110.0},
    "Chickpea (Low rainfall & low nitrogen)": {"N": 40, "P": 60, "K": 80, "temperature": 18.0, "ph": 7.5, "rainfall": 80.0},
}

# Load the trained model bundle once and reuse it for every prediction
@st.cache_resource
def load_bundle():
    if not MODEL_PATH.exists():
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

bundle = load_bundle()

st.title("🌾 Smart Crop Recommendation System")
st.write("Enter soil-test parameters and climate measurements to predict the optimal crop for cultivation.")

if bundle is None:
    st.error(
        "⚠️ Model file not found at `model/crop_model.pkl`.\n\n"
        "Please run `python train.py` in your terminal to train and save the model first."
    )
    st.stop()

model, features = bundle["model"], bundle["features"]

# Sidebar Presets & Info
with st.sidebar:
    st.header("⚙️ Quick Presets")
    selected_preset = st.selectbox("Load sample environmental profile:", list(PRESETS.keys()))
    preset_data = PRESETS[selected_preset] if selected_preset and PRESETS[selected_preset] else None
    
    st.markdown("---")
    st.markdown("### 📊 About the Model")
    st.write(
        "- **Algorithm:** Random Forest Classifier\n"
        "- **Accuracy:** ~98.4% on test set\n"
        "- **Supported Crops:** 22 varieties\n"
        "- **Features:** N, P, K, Temp, pH, Rainfall"
    )

default_vals = preset_data or {"N": 90, "P": 42, "K": 43, "temperature": 21.0, "ph": 6.5, "rainfall": 200.0}

c1, c2 = st.columns(2)
values = {}
with c1:
    values["N"] = st.number_input(LABELS["N"], value=int(default_vals["N"]), step=1, help="Nitrogen content ratio in soil (0-140)")
    values["P"] = st.number_input(LABELS["P"], value=int(default_vals["P"]), step=1, help="Phosphorus content ratio in soil (5-145)")
    values["K"] = st.number_input(LABELS["K"], value=int(default_vals["K"]), step=1, help="Potassium content ratio in soil (5-205)")
with c2:
    values["temperature"] = st.number_input(LABELS["temperature"], value=float(default_vals["temperature"]), step=0.1, help="Average temperature in Celsius (8-44 °C)")
    values["ph"] = st.number_input(LABELS["ph"], value=float(default_vals["ph"]), step=0.1, help="Soil pH level (3.5-10.0)")
    values["rainfall"] = st.number_input(LABELS["rainfall"], value=float(default_vals["rainfall"]), step=1.0, help="Annual rainfall in mm (20-300 mm)")

if st.button("Recommend Crop", type="primary", use_container_width=True):
    # Validate every input against its allowed range
    errors = [
        f"{LABELS[f]} must be between {lo} and {hi}."
        for f, (lo, hi) in RANGES.items()
        if not (lo <= values[f] <= hi)
    ]
    if errors:
        for e in errors:
            st.error(e)
    else:
        # Build one input row in exactly the feature order used during training
        row = pd.DataFrame([[values[f] for f in features]], columns=features)
        proba = model.predict_proba(row)[0]
        top = sorted(zip(model.classes_, proba), key=lambda t: -t[1])[:3]
        
        st.success(f"### Recommended Crop: **{top[0][0].upper()}** (confidence: {top[0][1]:.1%})")
        st.write("#### Top Alternatives:")
        st.table(pd.DataFrame({
            "Crop": [c.title() for c, _ in top],
            "Probability": [f"{v:.1%}" for _, v in top]
        }))
        st.caption(
            "The recommendation is learned from agricultural dataset records and serves as a decision guide, "
            "not a substitute for local agronomic soil and seed testing."
        )

# Optional Visualizations Expander
if FIGS_DIR.exists():
    with st.expander("📈 View Model Analytics & Feature Importance"):
        imp_path = FIGS_DIR / "importance.png"
        comp_path = FIGS_DIR / "compare.png"
        cm_path = FIGS_DIR / "cm_rf.png"
        if imp_path.exists():
            st.image(str(imp_path), caption="Random Forest Feature Importance")
        if comp_path.exists():
            st.image(str(comp_path), caption="Model Benchmark Comparison")
        if cm_path.exists():
            st.image(str(cm_path), caption="Random Forest Confusion Matrix")
