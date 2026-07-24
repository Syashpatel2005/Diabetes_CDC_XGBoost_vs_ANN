"""
app.py
======
Diabetes Risk Detector — end-to-end Streamlit frontend for the
XGBoost vs ANN models trained in Diabetes_CDC_XGBoost_vs_ANN_v2.ipynb.

Run with:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

from model_utils import (
    load_all, build_input_df, predict_both, xgb_feature_importance, check_local_files,
    FEATURE_ORDER, AGE_LABELS, INCOME_LABELS, GENHLTH_LABELS,
)

sns.set_theme(style="whitegrid")
st.set_page_config(page_title="Diabetes Risk Detector", page_icon="🩺", layout="wide")

# --------------------------------------------------------------------------
# Load models once (cached inside model_utils.load_all)
# --------------------------------------------------------------------------
try:
    models = load_all()
    models_ready = True
except Exception as e:
    models_ready = False
    load_error = str(e)

# --------------------------------------------------------------------------
# Sidebar navigation
# --------------------------------------------------------------------------
st.sidebar.title("🩺 Diabetes Risk Detector")
page = st.sidebar.radio(
    "Navigate",
    ["Home", "Predict", "Model Insights", "Evaluate on a Test CSV", "About"],
)
st.sidebar.divider()
if models_ready:
    st.sidebar.success("✅ Models loaded")
else:
    st.sidebar.error("⚠️ Models not loaded")
    with st.sidebar.expander("What's found in models/?"):
        status = check_local_files()
        for key, found in status.items():
            st.write(("✅ " if found else "❌ ") + key)

# ==========================================================================
# PAGE: Home
# ==========================================================================
if page == "Home":
    st.title("🩺 Diabetes Risk Detector")
    st.markdown(
        "This app estimates diabetes/prediabetes risk from 11 health "
        "indicators, using two models trained on the **CDC Diabetes Health "
        "Indicators** dataset (BRFSS survey data):"
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            "**🌲 XGBoost** — gradient-boosted trees, tuned with GridSearchCV, "
            "operating on the 11 highest-importance features."
        )
    with c2:
        st.markdown(
            "**🧠 ANN** — a Keras neural network tuned with Keras Tuner, "
            "using the same 11 (standardized) features."
        )

    st.info(
        "Both models use a **tuned decision threshold** (not the default "
        "0.5) chosen on a validation set to reach ~80% accuracy while "
        "keeping recall as high as possible. See **Model Insights** for "
        "what that trade-off means."
    )

    if not models_ready:
        st.error(
            f"Models aren't loaded yet, so Predict/Model Insights won't work. "
            f"Details: {load_error}"
        )
        st.markdown(
            "**Fix:** make sure your 4 files are in a `models/` folder next "
            "to `app.py`, named exactly:\n"
            "- `xgb_diabetes_final_v2.pkl`\n"
            "- `ann_diabetes_final_v2.keras`\n"
            "- `scaler_diabetes_final_v2.pkl`\n"
            "- `chosen_thresholds_v2.pkl`\n\n"
            "Check the sidebar (\"What's found in models/?\") to see exactly "
            "which files are missing."
        )
    else:
        st.success("Models are loaded and ready — head to **Predict** to try it.")

# ==========================================================================
# PAGE: Predict
# ==========================================================================
elif page == "Predict":
    st.title("🔍 Predict Diabetes Risk")

    if not models_ready:
        st.error(f"Models aren't loaded: {load_error}")
        st.stop()

    st.caption("Fill in the form and click Predict. All fields use the same "
               "encoding as the CDC BRFSS survey the models were trained on.")

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Medical history**")
            high_bp = st.selectbox("High blood pressure?", ["No", "Yes"])
            high_chol = st.selectbox("High cholesterol?", ["No", "Yes"])
            chol_check = st.selectbox("Cholesterol checked in last 5 years?", ["No", "Yes"])
            heart_disease = st.selectbox("Heart disease or heart attack history?", ["No", "Yes"])

        with c2:
            st.markdown("**Lifestyle & body**")
            bmi = st.number_input("BMI", min_value=12.0, max_value=70.0, value=27.0, step=0.1)
            hvy_alcohol = st.selectbox("Heavy alcohol consumption?", ["No", "Yes"])
            diff_walk = st.selectbox("Serious difficulty walking/climbing stairs?", ["No", "Yes"])
            gen_hlth = st.select_slider(
                "General health (self-rated)",
                options=list(GENHLTH_LABELS.keys()),
                format_func=lambda k: GENHLTH_LABELS[k],
                value=3,
            )

        with c3:
            st.markdown("**Demographics**")
            sex = st.selectbox("Sex", ["Female", "Male"])
            age = st.select_slider(
                "Age group",
                options=list(AGE_LABELS.keys()),
                format_func=lambda k: AGE_LABELS[k],
                value=7,
            )
            income = st.select_slider(
                "Household income",
                options=list(INCOME_LABELS.keys()),
                format_func=lambda k: INCOME_LABELS[k],
                value=6,
            )

        submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

    if submitted:
        raw = {
            "HighBP": 1 if high_bp == "Yes" else 0,
            "HighChol": 1 if high_chol == "Yes" else 0,
            "CholCheck": 1 if chol_check == "Yes" else 0,
            "BMI": bmi,
            "HeartDiseaseorAttack": 1 if heart_disease == "Yes" else 0,
            "HvyAlcoholConsump": 1 if hvy_alcohol == "Yes" else 0,
            "GenHlth": gen_hlth,
            "DiffWalk": 1 if diff_walk == "Yes" else 0,
            "Sex": 1 if sex == "Male" else 0,
            "Age": age,
            "Income": income,
        }
        input_df = build_input_df(raw)
        result = predict_both(models, input_df)

        st.divider()
        st.subheader("Results")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 🌲 XGBoost")
            label = "🔴 Higher risk" if result["xgb_pred"] == 1 else "🟢 Lower risk"
            st.metric("Prediction", label, f"probability {result['xgb_prob']:.1%}")
            st.progress(min(max(result["xgb_prob"], 0.0), 1.0))
            st.caption(f"Decision threshold: {result['xgb_threshold']:.2f}")
        with c2:
            st.markdown("### 🧠 ANN")
            label = "🔴 Higher risk" if result["ann_pred"] == 1 else "🟢 Lower risk"
            st.metric("Prediction", label, f"probability {result['ann_prob']:.1%}")
            st.progress(min(max(result["ann_prob"], 0.0), 1.0))
            st.caption(f"Decision threshold: {result['ann_threshold']:.2f}")

        if result["xgb_pred"] != result["ann_pred"]:
            st.warning(
                "The two models disagree on the classification at their "
                "chosen thresholds — worth looking at both probabilities "
                "rather than trusting either label alone."
            )

        st.info(
            "⚠️ This is a portfolio/educational tool based on a public "
            "survey dataset, **not a medical diagnosis**. Please consult a "
            "healthcare professional for real medical concerns."
        )

# ==========================================================================
# PAGE: Model Insights
# ==========================================================================
elif page == "Model Insights":
    st.title("📊 Model Insights")

    if not models_ready:
        st.error(f"Models aren't loaded: {load_error}")
        st.stop()

    st.subheader("XGBoost feature importance")
    imp = xgb_feature_importance(models)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=imp.values, y=imp.index, hue=imp.index, palette="viridis",
                legend=False, ax=ax)
    ax.set_xlabel("Importance"); ax.set_ylabel("")
    st.pyplot(fig, clear_figure=True)

    st.divider()
    st.subheader("Chosen decision thresholds")
    c1, c2 = st.columns(2)
    c1.metric("XGBoost threshold", f"{models['xgb_threshold']:.2f}", "vs. default 0.50")
    c2.metric("ANN threshold", f"{models['ann_threshold']:.2f}", "vs. default 0.50")
    st.markdown(
        "Both thresholds were tuned **on a validation set** (not the test "
        "set) to reach ≥80% accuracy while keeping the best recall "
        "available at that point. Raising the threshold above 0.5 trades "
        "away recall (catching fewer true diabetic cases) for a higher "
        "headline accuracy — it does not mean the model itself improved. "
        "ROC-AUC (~0.82–0.83 in the original notebook run) is the "
        "threshold-independent number that reflects actual model quality."
    )

# ==========================================================================
# PAGE: Evaluate on a Test CSV
# ==========================================================================
elif page == "Evaluate on a Test CSV":
    st.title("🧪 Evaluate on Your Own Test CSV")
    st.caption(
        "Upload a CSV with the 11 feature columns plus a `Diabetes_binary` "
        "label column to see real accuracy/precision/recall/F1/ROC-AUC and "
        "a confusion matrix for both models on that data."
    )

    if not models_ready:
        st.error(f"Models aren't loaded: {load_error}")
        st.stop()

    up = st.file_uploader("Test CSV", type="csv")
    if up is not None:
        test_df = pd.read_csv(up)
        missing = [c for c in FEATURE_ORDER + ["Diabetes_binary"] if c not in test_df.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            X_test = test_df[FEATURE_ORDER]
            y_test = test_df["Diabetes_binary"].astype(int)

            xgb_prob = models["xgb_model"].predict_proba(X_test)[:, 1]
            xgb_pred = (xgb_prob >= models["xgb_threshold"]).astype(int)

            scaled = models["scaler"].transform(X_test)
            ann_prob = models["ann_model"].predict(scaled, verbose=0).ravel()
            ann_pred = (ann_prob >= models["ann_threshold"]).astype(int)

            def metrics_row(name, y_true, pred, prob):
                return {
                    "Model": name,
                    "Accuracy": accuracy_score(y_true, pred),
                    "Precision": precision_score(y_true, pred, zero_division=0),
                    "Recall": recall_score(y_true, pred, zero_division=0),
                    "F1": f1_score(y_true, pred, zero_division=0),
                    "ROC AUC": roc_auc_score(y_true, prob),
                }

            results_df = pd.DataFrame([
                metrics_row("XGBoost", y_test, xgb_pred, xgb_prob),
                metrics_row("ANN", y_test, ann_pred, ann_prob),
            ]).set_index("Model")

            st.subheader("Metrics")
            st.dataframe(results_df.style.format("{:.4f}"), use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                cm = confusion_matrix(y_test, xgb_pred)
                fig, ax = plt.subplots(figsize=(4, 3.5))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                            xticklabels=["No", "Yes"], yticklabels=["No", "Yes"], ax=ax)
                ax.set_title("XGBoost — Confusion Matrix")
                ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
                st.pyplot(fig, clear_figure=True)
            with c2:
                cm = confusion_matrix(y_test, ann_pred)
                fig, ax = plt.subplots(figsize=(4, 3.5))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                            xticklabels=["No", "Yes"], yticklabels=["No", "Yes"], ax=ax)
                ax.set_title("ANN — Confusion Matrix")
                ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
                st.pyplot(fig, clear_figure=True)

# ==========================================================================
# PAGE: About
# ==========================================================================
elif page == "About":
    st.title("ℹ️ About this project")
    st.markdown(
        "**Dataset:** CDC Diabetes Health Indicators (UCI ML Repository, "
        "id=891), derived from the BRFSS health survey.\n\n"
        "**Models:** XGBoost (GridSearchCV-tuned) and a Keras ANN "
        "(Keras Tuner-tuned), both trained on 11 features selected by "
        "XGBoost feature importance from a 21-feature starting set.\n\n"
        "**Threshold tuning:** both models use a validation-set-tuned "
        "decision threshold to reach ~80% accuracy; ROC-AUC is reported "
        "separately since it doesn't depend on the threshold.\n\n"
        "**Disclaimer:** built for educational/portfolio purposes on a "
        "public survey dataset. Not a medical device and not a substitute "
        "for professional medical advice."
    )
