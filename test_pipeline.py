"""Test suite for Smart Crop Recommendation System."""
import unittest
from pathlib import Path
import pickle
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Crop_recommendation.csv"
MODEL_PATH = BASE_DIR / "model" / "crop_model.pkl"

EXPECTED_FEATURES = ["N", "P", "K", "temperature", "ph", "rainfall"]

class TestCropRecommendationPipeline(unittest.TestCase):
    def test_dataset_exists_and_valid(self):
        self.assertTrue(DATA_PATH.exists(), f"Dataset not found at {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
        self.assertFalse(df.empty, "Dataset is empty")
        for col in EXPECTED_FEATURES + ["label"]:
            self.assertIn(col, df.columns, f"Missing column: {col}")
        self.assertEqual(int(df.isna().sum().sum()), 0, "Dataset contains missing values")

    def test_model_bundle_loads(self):
        self.assertTrue(MODEL_PATH.exists(), f"Model file not found at {MODEL_PATH}")
        with open(MODEL_PATH, "rb") as f:
            bundle = pickle.load(f)
        self.assertIn("model", bundle, "Bundle missing 'model' key")
        self.assertIn("features", bundle, "Bundle missing 'features' key")
        self.assertIn("classes", bundle, "Bundle missing 'classes' key")
        self.assertEqual(bundle["features"], EXPECTED_FEATURES, "Feature mismatch in model bundle")

    def test_prediction_output(self):
        with open(MODEL_PATH, "rb") as f:
            bundle = pickle.load(f)
        model = bundle["model"]
        features = bundle["features"]

        # Typical conditions for rice: high rainfall, moderate temp, balanced NPK
        sample = pd.DataFrame([[90, 42, 43, 21.0, 6.5, 202.0]], columns=features)
        pred = model.predict(sample)
        proba = model.predict_proba(sample)

        self.assertEqual(len(pred), 1)
        self.assertIn(pred[0], bundle["classes"])
        self.assertAlmostEqual(float(proba.sum()), 1.0, places=4)
        self.assertEqual(pred[0], "rice", f"Expected 'rice', got '{pred[0]}'")

if __name__ == "__main__":
    unittest.main()
