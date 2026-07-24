"""
model_utils.py
===============
All the "plumbing": loading models from the local models/ folder (preferred),
falling back to Google Drive only if a file is missing locally.

You should not need to edit this file — edit config.py only if you need the
Google Drive fallback (not required if your models/ folder is already
populated).
"""

import os
import joblib
import pandas as pd
import streamlit as st

from config import GDRIVE_FILE_IDS, LOCAL_FILENAMES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

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
# File resolution: models/ folder first, Google Drive only as fallback
# --------------------------------------------------------------------------
def check_local_files() -> dict:
    """Returns {key: found_bool} for each of the 4 expected files in models/.
    Handy for showing the user exactly what's found vs. missing."""
    status = {}
    for key, filename in LOCAL_FILENAMES.items():
        status[key] = os.path.exists(os.path.join(MODELS_DIR, filename))
    return status


def _download_from_drive(file_id: str, target_path: str):
    import gdown
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, target_path, quiet=False)
    if not os.path.exists(target_path) or os.path.getsize(target_path) == 0:
        raise RuntimeError(
            f"Download failed for {os.path.basename(target_path)}. "
            "Check that the Google Drive file is shared as "
            "'Anyone with the link' and that the file ID in config.py is correct."
        )


def _ensure_file(key: str) -> str:
    """Return a local path to the artifact. Uses models/<filename> if it's
    already there; only touches Google Drive if it's missing."""
    local_path = os.path.join(MODELS_DIR, LOCAL_FILENAMES[key])
    if os.path.exists(local_path):
        return local_path

    file_id = GDRIVE_FILE_IDS.get(key, "")
    if not file_id or "PUT_" in file_id:
        raise RuntimeError(
            f"'{LOCAL_FILENAMES[key]}' not found in {MODELS_DIR}/, and no "
            f"Google Drive file ID is configured in config.py for '{key}' "
            f"either. Either copy the file into models/, or fill in "
            f"GDRIVE_FILE_IDS['{key}'] in config.py."
        )

    with st.spinner(f"Downloading {LOCAL_FILENAMES[key]} from Google Drive..."):
        _download_from_drive(file_id, local_path)
    return local_path


@st.cache_resource(show_spinner="Loading models...")
def load_all():
    """Load XGBoost, ANN, scaler, and chosen thresholds. Cached across reruns."""
    import tensorflow as tf

    xgb_path = _ensure_file("xgb_model")
    ann_path = _ensure_file("ann_model")
    scaler_path = _ensure_file("scaler")
    thresh_path = _ensure_file("thresholds")

    xgb_model = joblib.load(xgb_path)
    ann_model = tf.keras.models.load_model(ann_path)
    scaler = joblib.load(scaler_path)
    thresholds = joblib.load(thresh_path)

    return {
        "xgb_model": xgb_model,
        "ann_model": ann_model,
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
    values) and return probabilities + thresholded predictions for each."""
    xgb_model = models["xgb_model"]
    ann_model = models["ann_model"]
    scaler = models["scaler"]

    # XGBoost pipeline expects raw (unscaled) features.
    xgb_prob = float(xgb_model.predict_proba(input_df)[:, 1][0])
    xgb_pred = int(xgb_prob >= models["xgb_threshold"])

    # ANN expects scaled features.
    scaled = scaler.transform(input_df)
    ann_prob = float(ann_model.predict(scaled, verbose=0).ravel()[0])
    ann_pred = int(ann_prob >= models["ann_threshold"])

    return {
        "xgb_prob": xgb_prob, "xgb_pred": xgb_pred,
        "xgb_threshold": models["xgb_threshold"],
        "ann_prob": ann_prob, "ann_pred": ann_pred,
        "ann_threshold": models["ann_threshold"],
    }


def xgb_feature_importance(models: dict) -> pd.Series:
    xgb_pipeline = models["xgb_model"]
    booster = xgb_pipeline.named_steps["xgb"]
    importances = booster.feature_importances_
    return pd.Series(importances, index=FEATURE_ORDER).sort_values(ascending=False)
