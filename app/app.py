import os
import json

import numpy as np
import pandas as pd
import streamlit as st
import joblib

st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(BASE_DIR, "model")

MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
COLS_PATH  = os.path.join(MODEL_DIR, "columns.json")


@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        st.error(f"model.pkl not found at: {MODEL_PATH}")
        return None, None
    if not os.path.exists(COLS_PATH):
        st.error(f"columns.json not found at: {COLS_PATH}")
        return None, None

    model = joblib.load(MODEL_PATH)
    with open(COLS_PATH, "r") as f:
        feature_names = json.load(f)

    return model, feature_names


def predict_df(model, feature_names, df_input: pd.DataFrame) -> pd.DataFrame:
    X = df_input.copy()
    X = X.reindex(columns=feature_names, fill_value=0.0)
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= 0.5).astype(int)

    out = df_input.copy()
    out["default_prob"] = prob
    out["default_pred"] = pred
    return out


def main():
    model, feature_names = load_artifacts()
    if model is None or feature_names is None:
        st.stop()

    st.title("💳 Credit Risk Prediction")
    st.caption("Random Forest model trained on UCI Default of Credit Card Clients dataset.")

    tab_single, tab_batch = st.tabs(["Single Prediction", "Batch Prediction"])

    # -------- Single --------
    with tab_single:
        col_left, col_right = st.columns([1.5, 1])

        with col_left:
            st.subheader("Customer Input")
            cols = st.columns(2)
            user_inputs = {}
            for i, feat in enumerate(feature_names):
                with cols[i % 2]:
                    user_inputs[feat] = st.number_input(
                        feat,
                        value=0.0,
                        step=1.0,
                        format="%.2f",
                        key=f"single_{feat}",
                    )

            submitted = st.button("Predict Default Risk", type="primary")

        with col_right:
            st.subheader("Prediction Result")
            if submitted:
                df_single = pd.DataFrame([user_inputs])
                out = predict_df(model, feature_names, df_single)
                prob = float(out["default_prob"].iloc[0])
                pred = int(out["default_pred"].iloc[0])

                st.metric("Default Probability", f"{prob:.3f}")
                if pred == 1:
                    st.error("Prediction: HIGH RISK of default (class 1)")
                else:
                    st.success("Prediction: LOW RISK of default (class 0)")

                st.write("Details:")
                st.dataframe(out)
            else:
                st.info("Fill the form and click Predict.")

    # -------- Batch --------
    with tab_batch:
        st.subheader("Batch Prediction via CSV")
        st.write("CSV must contain these columns (any order):")
        st.code(", ".join(feature_names))

        file = st.file_uploader("Upload CSV", type=["csv"])
        if file is not None:
            df_up = pd.read_csv(file)
            st.write("Preview:")
            st.dataframe(df_up.head())

            if st.button("Run Batch Prediction", type="primary"):
                out = predict_df(model, feature_names, df_up)
                st.write("Sample output:")
                st.dataframe(out.head())

                csv_bytes = out.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download predictions CSV",
                    data=csv_bytes,
                    file_name="credit_risk_predictions.csv",
                    mime="text/csv",
                )
        else:
            st.info("Upload a CSV file to run batch predictions.")


if __name__ == "__main__":
    main()
