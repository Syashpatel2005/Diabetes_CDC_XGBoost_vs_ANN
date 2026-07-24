"""
config.py
=========
Filenames the app expects to find inside the models/ folder next to app.py.

If your files are named differently than what's already saved from the
notebook, just change the values below to match your actual filenames —
nothing else in the project needs to change.
"""

LOCAL_FILENAMES = {
    "xgb_model":  "xgb_diabetes_final_v2.pkl",
    "ann_model":  "ann_diabetes_final_v2.keras",
    "scaler":     "scaler_diabetes_final_v2.pkl",
    "thresholds": "chosen_thresholds_v2.pkl",
}
