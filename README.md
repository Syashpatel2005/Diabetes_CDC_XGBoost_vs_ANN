# Diabetes Risk Detector — End-to-End Project

A Streamlit frontend for the XGBoost vs ANN diabetes-risk models trained in
`Diabetes_CDC_XGBoost_vs_ANN_v2.ipynb`. Loads your `.pkl`/`.keras` files
from Google Drive (so you don't need to commit 25MB+ files to GitHub).

## Project structure

```
diabetes-detector/
├── app.py              # Streamlit frontend — the whole UI
├── model_utils.py       # Loads/downloads models, runs predictions (don't need to edit)
├── config.py             # <-- YOU EDIT THIS: paste your Google Drive file IDs here
├── requirements.txt
├── .gitignore
├── README.md
└── models/               # created automatically — downloaded files land here (gitignored)
```

You don't need to touch `app.py` or `model_utils.py` to get this running —
just `config.py`.

---

## Step 1 — Finish your notebook run

Make sure Section 14 of your notebook (`## 14. Save Final Models`) has
actually run and produced these 4 files in your Google Drive:

- `xgb_diabetes_final_v2.pkl`
- `ann_diabetes_final_v2.keras`
- `scaler_diabetes_final_v2.pkl`
- `chosen_thresholds_v2.pkl`

(`dropped_features_v2.pkl` is also saved but the app doesn't need it — the
feature list is already hardcoded correctly in `model_utils.py`.)

## Step 2 — Share each file and grab its file ID

For **each of the 4 files** in Google Drive:

1. Right-click the file → **Share** → **Share**
2. Under "General access", change to **"Anyone with the link"** (role: Viewer)
   — this step is essential, without it the deployed app can't download the file
3. Click **Copy link**. It'll look like:
   ```
   https://drive.google.com/file/d/1AbCdEfGhIjKlMnOpQrSt/view?usp=sharing
   ```
4. The **file ID** is the part between `/d/` and `/view`:
   ```
   1AbCdEfGhIjKlMnOpQrSt
   ```

## Step 3 — Fill in `config.py`

Open `config.py` and paste each file ID in:

```python
GDRIVE_FILE_IDS = {
    "xgb_model":  "1AbCdEfGhIjKlMnOpQrSt",   # your real xgb pkl file ID
    "ann_model":  "1XyZ...",                  # your real ann keras file ID
    "scaler":     "1Qwe...",                  # your real scaler pkl file ID
    "thresholds": "1Rty...",                  # your real thresholds pkl file ID
}
```

That's it — no other file needs editing for a basic working app.

> **Alternative (no Drive needed):** if you'd rather test locally without
> Google Drive at all, just create a `models/` folder next to `app.py` and
> drop the 4 files in there with their original names. The app checks
> `models/` first and only falls back to downloading from Drive if a file
> isn't already there.

## Step 4 — Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

First run will download the 4 files from Drive into `models/` (you'll see a
"Downloading..." spinner) — after that they're cached locally, so it's
instant on reruns. Delete the `models/` folder any time to force a
re-download (e.g. if you update the files in Drive).

## Step 5 — Try the app

- **Predict** — fill in the 11-field form, get XGBoost + ANN predictions
  side by side, each with its own probability and tuned decision threshold
- **Model Insights** — live XGBoost feature-importance chart, plus an
  explanation of what the tuned thresholds actually mean
- **Evaluate on a Test CSV** — upload a CSV with the 11 feature columns +
  a `Diabetes_binary` label column to get real accuracy/precision/recall/F1/
  ROC-AUC and confusion matrices (handy for demoing on your held-out test
  set, or any new labeled data)
- **About** — project summary, good for pasting into a resume/portfolio blurb

## Step 6 — Deploy (optional but recommended for a portfolio)

1. Push this folder to a GitHub repo (the `.gitignore` already excludes
   `models/` and any stray `.pkl`/`.keras` files, so your repo stays small)
2. Go to [share.streamlit.io](https://share.streamlit.io), connect your
   GitHub repo, point it at `app.py`
3. First load after deploy will download from Drive (~1–2 min depending on
   file size) — after that it's cached until the app restarts

**Heads up:** Streamlit Community Cloud spins your app down after a period
of inactivity. When it wakes back up, it'll re-download from Drive once,
then be fast again for the rest of that session.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| "Download failed" error | Check the file's Drive sharing is "Anyone with the link", and that the file ID in `config.py` is copied correctly (no extra spaces/characters) |
| App loads but Predict throws an error | Make sure all 4 files are the *final v2* versions from Section 14, not the earlier `xgb_diabetes_pipeline_v2.pkl` from Section 7 |
| ANN prediction looks wrong | Confirm `scaler_diabetes_final_v2.pkl` came from the *same* notebook run as `ann_diabetes_final_v2.keras` — mixing scalers/models from different runs will silently give bad predictions |
| Slow first load on Streamlit Cloud | Normal — it's downloading ~100MB from Drive. Consider trimming the ANN model size or hosting on a service with persistent storage if this becomes a real problem |
