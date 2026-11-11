"""CLI script for Smart Crop Recommendation System.

Usage:
    python predict.py --N 90 --P 42 --K 43 --temp 21.0 --ph 6.5 --rain 200.0
    python predict.py (interactive mode)
"""
import argparse
from pathlib import Path
import pickle
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "crop_model.pkl"

RANGES = {
    "N": (0, 140),
    "P": (5, 145),
    "K": (5, 205),
    "temperature": (8.0, 44.0),
    "ph": (3.5, 10.0),
    "rainfall": (20.0, 300.0),
}

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. Run 'python train.py' first."
        )
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def recommend(n, p, k, temperature, ph, rainfall):
    bundle = load_model()
    model = bundle["model"]
    features = bundle["features"]

    values = {
        "N": float(n),
        "P": float(p),
        "K": float(k),
        "temperature": float(temperature),
        "ph": float(ph),
        "rainfall": float(rainfall),
    }

    # Validate inputs
    for feat, (lo, hi) in RANGES.items():
        val = values[feat]
        if not (lo <= val <= hi):
            print(f"[Warning] {feat} value {val} is outside typical range [{lo}, {hi}].")

    row = pd.DataFrame([[values[f] for f in features]], columns=features)
    proba = model.predict_proba(row)[0]
    ranked = sorted(zip(model.classes_, proba), key=lambda x: -x[1])

    best_crop, best_conf = ranked[0]
    return best_crop, best_conf, ranked[:3]

def main():
    parser = argparse.ArgumentParser(description="Crop Recommendation CLI")
    parser.add_argument("--N", type=float, help="Nitrogen ratio in soil (0 - 140)")
    parser.add_argument("--P", type=float, help="Phosphorus ratio in soil (5 - 145)")
    parser.add_argument("--K", type=float, help="Potassium ratio in soil (5 - 205)")
    parser.add_argument("--temp", "--temperature", dest="temp", type=float, help="Temperature in °C (8.0 - 44.0)")
    parser.add_argument("--ph", type=float, help="Soil pH value (3.5 - 10.0)")
    parser.add_argument("--rain", "--rainfall", dest="rain", type=float, help="Rainfall in mm (20.0 - 300.0)")

    args = parser.parse_args()

    # If any argument is omitted, enter interactive prompt
    if any(v is None for v in [args.N, args.P, args.K, args.temp, args.ph, args.rain]):
        print("--- Smart Crop Recommendation (Interactive Mode) ---")
        try:
            n = float(input("Enter Nitrogen (N) [e.g. 90]: ") or 90)
            p = float(input("Enter Phosphorus (P) [e.g. 42]: ") or 42)
            k = float(input("Enter Potassium (K) [e.g. 43]: ") or 43)
            temp = float(input("Enter Temperature (°C) [e.g. 21.0]: ") or 21.0)
            ph = float(input("Enter Soil pH [e.g. 6.5]: ") or 6.5)
            rain = float(input("Enter Rainfall (mm) [e.g. 200.0]: ") or 200.0)
        except (ValueError, EOFError):
            print("Invalid input or non-interactive environment. Using default sample parameters.")
            n, p, k, temp, ph, rain = 90, 42, 43, 21.0, 6.5, 200.0
    else:
        n, p, k, temp, ph, rain = args.N, args.P, args.K, args.temp, args.ph, args.rain

    best_crop, best_conf, top3 = recommend(n, p, k, temp, ph, rain)

    print("\n" + "=" * 45)
    print(f"🌾 Recommended Crop: {best_crop.upper()} ({best_conf:.1%} confidence)")
    print("=" * 45)
    print("Top 3 Candidates:")
    for rank, (crop, conf) in enumerate(top3, start=1):
        print(f"  {rank}. {crop.title():<15} : {conf:>6.1%}")
    print("=" * 45)

if __name__ == "__main__":
    main()
