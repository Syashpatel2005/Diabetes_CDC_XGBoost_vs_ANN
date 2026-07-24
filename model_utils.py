"""
model_utils.py
===============
Loads models from the local models/ folder and runs predictions with the
correct preprocessing for each model type.

You should not need to edit this file — edit config.py if your filenames differ.
"""

import os
import joblib
import pandas as pd
import streamlit as st

from config import LOCAL_FILENAMES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# The exact 11 features the models were trained on, in the exact order
# the notebook used. DO NOT reorder — XGBoost/ANN both expect this order.
FEATURE_ORDER = [
    "HighBP", "HighChol", "CholCheck", "BMI", "HeartDiseaseorAttack",
    "HvyAlcoholConsump", "GenHlth", "DiffWalk", "Sex", "Age", "Income",
]

# Human-readable labels <-> the numeric codes the CDC BRFSS dataset uses.
# Used to build friendly form widgets instead of asking users to type raw codes.
AGE_LABELS = {
    1: "18–24", 2: "25–29", 3: "30–34", 4: "35–39", 5: "40–44",
    6: "45–49", 7: "50–54", 8: "55–59", 9: "60–64", 10: "65–69",
    11: "70–74", 12: "75–79", 13: "80+",
}
INCOME_LABELS = {
    1: "Less than $10,000", 2: "$10,000–$14,999", 3: "$15,000–$19,999",
    4: "$20,000–$24,999", 5: "$25,000–$34,999", 6: "$35,000–$49,999",
    7: "$50,000–$74,999", 8: "$75,000 or more",
}
GENHLTH_LABELS = {
    1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor",
}


# --------------------------------------------------------------------------
# File resolution + loading
# --------------------------------------------------------------------------
def check_local_files() -> dict:
    """Returns {key: found_bool} for each of the 4 expected files in models/.
    Used to show a clear checklist in the sidebar if something's missing."""
    status = {}
    for key, filename in LOCAL_FILENAMES.items():
        status[key] = os.path.exists(os.path.join(MODELS_DIR, filename))
    return status


def _ensure_file(key: str) -> str:
    local_path = os.path.join(MODELS_DIR, LOCAL_FILENAMES[key])
    if not os.path.exists(local_path):
        raise RuntimeError(
            f"'{LOCAL_FILENAMES[key]}' not found in {MODELS_DIR}/. "
            f"Make sure it's there, or update LOCAL_FILENAMES['{key}'] in "
            f"config.py to match your actual filename."
        )
    return local_path


@st.cache_resource(show_spinner="Loading models...")
def load_all():
    """Load XGBoost, ANN, scaler, and chosen thresholds. Cached across reruns.

    ANN loading is best-effort: if TensorFlow fails to import (e.g. blocked
    by a Windows Application Control / antivirus policy, a common corporate-
    laptop issue), XGBoost still loads and works — ann_model is set to None
    and ann_load_error carries the reason, which the UI checks for.
    """
    xgb_path = _ensure_file("xgb_model")
    scaler_path = _ensure_file("scaler")
    thresh_path = _ensure_file("thresholds")

    xgb_model = joblib.load(xgb_path)
    scaler = joblib.load(scaler_path)
    thresholds = joblib.load(thresh_path)

    ann_model = None
    ann_load_error = None
    try:
        ann_path = _ensure_file("ann_model")
        import tensorflow as tf
        ann_model = tf.keras.models.load_model(ann_path)
    except Exception as e:
        ann_load_error = str(e)

    return {
        "xgb_model": xgb_model,
        "ann_model": ann_model,
        "ann_load_error": ann_load_error,
        "scaler": scaler,
        "xgb_threshold": thresholds["xgb_threshold"],
        "ann_threshold": thresholds["ann_threshold"],
    }


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------
def build_input_df(raw: dict) -> pd.DataFrame:
    """raw: dict with the 11 feature keys -> values. Returns a 1-row DataFrame
    with columns in the exact order the models expect."""
    return pd.DataFrame([{col: raw[col] for col in FEATURE_ORDER}])


def predict_both(models: dict, input_df: pd.DataFrame) -> dict:
    """Run both models on a single-row input DataFrame (unscaled, raw feature
    values) and return probabilities + thresholded predictions for each.
    If the ANN failed to load, its fields come back as None."""
    xgb_model = models["xgb_model"]
    ann_model = models["ann_model"]
    scaler = models["scaler"]

    # XGBoost pipeline expects raw (unscaled) features.
    xgb_prob = float(xgb_model.predict_proba(input_df)[:, 1][0])
    xgb_pred = int(xgb_prob >= models["xgb_threshold"])

    result = {
        "xgb_prob": xgb_prob, "xgb_pred": xgb_pred,
        "xgb_threshold": models["xgb_threshold"],
        "ann_prob": None, "ann_pred": None,
        "ann_threshold": models["ann_threshold"],
    }

    if ann_model is not None:
        # ANN expects scaled features.
        scaled = scaler.transform(input_df)
        ann_prob = float(ann_model.predict(scaled, verbose=0).ravel()[0])
        result["ann_prob"] = ann_prob
        result["ann_pred"] = int(ann_prob >= models["ann_threshold"])

    return result


def xgb_feature_importance(models: dict) -> pd.Series:
    xgb_pipeline = models["xgb_model"]
    booster = xgb_pipeline.named_steps["xgb"]
    importances = booster.feature_importances_
    return pd.Series(importances, index=FEATURE_ORDER).sort_values(ascending=False)
