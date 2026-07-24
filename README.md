# Diabetes Risk Detector — End-to-End Project

A Streamlit frontend for the XGBoost vs ANN diabetes-risk models trained in
`Diabetes_CDC_XGBoost_vs_ANN_v2.ipynb`.

## Project structure

```
diabetes-detector/
├── app.py              # Streamlit frontend — the whole UI
├── model_utils.py       # Loads models, runs predictions (don't need to edit)
├── config.py             # Filenames the app expects in models/
├── requirements.txt
├── .gitignore
├── README.md
└── models/               # <-- put your 4 downloaded files here
    ├── xgb_diabetes_final_v2.pkl
    ├── ann_diabetes_final_v2.keras
    ├── scaler_diabetes_final_v2.pkl
    └── chosen_thresholds_v2.pkl
```

## Setup

1. Create a `models/` folder next to `app.py` (if it doesn't already exist)
2. Put your 4 files in it, named **exactly** as shown above

   If your files are named differently, open `config.py` and update
   `LOCAL_FILENAMES` to match your actual filenames — nothing else needs
   to change.

3. Install dependencies and run:
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```

That's it — the app loads everything from `models/` on startup (cached, so
it only reads the files once per session).

## Pages

- **Predict** — an 11-field form with friendly labels (e.g. "50–54" instead
  of raw code `7`) that maps back to the exact encoding your models were
  trained on. Shows XGBoost + ANN predictions side by side, each with its
  own probability and tuned decision threshold, and flags it if the two
  models disagree.
- **Model Insights** — live XGBoost feature-importance chart (computed from
  your loaded model, not hardcoded), plus an explanation of what the
  threshold tuning trade-off means.
- **Evaluate on a Test CSV** — upload any CSV with the 11 feature columns
  plus a `Diabetes_binary` label column to get real accuracy/precision/
  recall/F1/ROC-AUC and confusion matrices for both models — useful for
  demoing on your actual held-out test set.
- **About** — project summary, good for a resume/portfolio blurb.

## Deploying (optional)

If you want to put this on Streamlit Community Cloud or similar:

- The 4 model files (especially the ANN `.keras`) are likely too large for a
  plain GitHub push (25–100MB limits depending on method). Options:
  - Use [Git LFS](https://git-lfs.com/) to commit them anyway
  - Host them externally (Google Drive, S3, Hugging Face Hub, etc.) and add
    a small download step back into `model_utils.py` if you go this route
- `.gitignore` currently excludes `models/` and any stray `.pkl`/`.keras`
  files — remove those lines if you do want them committed via Git LFS

## Troubleshooting

| Problem | Fix |
|---|---|
| "not found in models/" error | Check the sidebar's "What's found in models/?" checklist to see exactly which file is missing or misnamed |
| Predict throws an error | Make sure all 4 files are the *final v2* versions from Section 14 of the notebook, not the earlier `xgb_diabetes_pipeline_v2.pkl` from Section 7 |
| ANN prediction looks wrong | Confirm `scaler_diabetes_final_v2.pkl` came from the *same* notebook run as `ann_diabetes_final_v2.keras` — mixing scalers/models from different runs will silently give bad predictions |
