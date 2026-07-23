"""
config.py
=========
Fill in the Google Drive FILE IDs for each artifact your notebook saved.

HOW TO GET A FILE ID:
1. In Google Drive, right-click the file -> "Share" -> "Copy link"
2. Make sure sharing is set to "Anyone with the link" (Viewer) — required,
   otherwise the app can't download it when deployed.
3. The link looks like:
       https://drive.google.com/file/d/1AbCdEfGhIjKlMnOpQrSt/view?usp=sharing
   The FILE ID is the part between /d/ and /view:
       1AbCdEfGhIjKlMnOpQrSt
4. Paste that ID below for each file.

You do NOT need to rename your files in Drive — the app downloads them and
saves them locally under the names in LOCAL_FILENAMES, regardless of what
they're called in Drive.
"""

GDRIVE_FILE_IDS = {
    "xgb_model":  "1RgwDsjANx4dboyU62gRk_b-3CyTDXgx5",
    "ann_model":  "1gqbUKS8Vaad_0JiaZpVpAldhVtefZctv",
    "scaler":     "1hIwVr_98RhS0_V3BQTKxILPWsS6f3FHM",
    "thresholds": "1LgDzoXYL_SjA0J62NdlQwZdOjQIqVtfC",
}

# Local filenames the app will save these under (inside the models/ folder).
# Change these only if you also change them in model_utils.py's expectations
# — otherwise leave as-is.
LOCAL_FILENAMES = {
    "xgb_model":  "xgb_diabetes_final_v2.pkl",
    "ann_model":  "ann_diabetes_final_v2.keras",
    "scaler":     "scaler_diabetes_final_v2.pkl",
    "thresholds": "chosen_thresholds_v2.pkl",
}
