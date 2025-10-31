"""
Streamlit multi-page app (single-file)
Project: Credit Risk Prediction using Random Forests
Style: Glassmorphic (Blue-Purple Neon theme - T1)
Pages: Home, Predict, Insights, About

Place model artifacts in ./model/: model.pkl, scaler.pkl, columns.json
Run: `streamlit run streamlit_app.py`

Notes:
- This is a single-file Streamlit app that uses a sidebar to switch pages.
- SHAP visualizations are produced using matplotlib and displayed via st.pyplot.
- For production deployment, consider splitting into modules and securing the model file.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib.pyplot as plt
import shap

from io import BytesIO

# -----------------------------
# Config & Utilities
# -----------------------------
st.set_page_config(page_title="Credit Risk Predictor", layout="wide", initial_sidebar_state="expanded")

MODEL_DIR = os.path.join(os.getcwd(), 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
COLUMNS_PATH = os.path.join(MODEL_DIR, 'columns.json')

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at {MODEL_PATH}. Put your trained model (model.pkl) there.")
        return None, None, None
    if not os.path.exists(SCALER_PATH):
        st.error(f"Scaler file not found at {SCALER_PATH}. Put your scaler (scaler.pkl) there.")
        return None, None, None
    if not os.path.exists(COLUMNS_PATH):
        st.error(f"Columns file not found at {COLUMNS_PATH}. Put your columns.json there.")
        return None, None, None

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(COLUMNS_PATH, 'r') as f:
        columns = json.load(f)
    return model, scaler, columns

@st.cache_data
def predict_df(model, scaler, columns, df_in):
    # assumes df_in has raw columns matching 'columns' keys or same processed features
    X = df_in.copy()
    # If scaler exists and numeric columns in X
    try:
        # if columns is list of feature names in order
        X = X[columns]
    except Exception:
        # best-effort: align intersection
        X = X.reindex(columns=columns, fill_value=0)
    try:
        X_scaled = scaler.transform(X)
    except Exception:
        X_scaled = X.values
    probs = model.predict_proba(X_scaled)[:,1] if hasattr(model, 'predict_proba') else model.predict(X_scaled)
    preds = (probs >= 0.5).astype(int)
    out = df_in.copy()
    out['default_prob'] = probs
    out['default_pred'] = preds
    return out

# -----------------------------
# Styling (Glassmorphic - Blue/Purple Neon)
# -----------------------------
def local_css():
    st.markdown(f"""
    <style>
    /* Background gradient */
    .stApp {{
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 35%, #7c3aed 100%);
        color: white;
    }}
    /* Glass card */
    .glass {{
        background: rgba(255,255,255,0.06);
        border-radius: 16px;
        padding: 18px;
        box-shadow: 0 8px 32px 0 rgba(31,38,135,0.37);
        backdrop-filter: blur(6px);
        -webkit-backdrop-filter: blur(6px);
        border: 1px solid rgba(255,255,255,0.1);
    }}
    .small-muted {{
        color: rgba(255,255,255,0.7);
        font-size:12px;
    }}
    .neon-btn {{
        background: linear-gradient(90deg,#7c3aed,#06b6d4);
        color: white;
        padding: 8px 14px;
        border-radius: 10px;
        border: none;
    }}
    </style>
    """, unsafe_allow_html=True)

local_css()

# -----------------------------
# App Pages
# -----------------------------

def page_home():
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.title('Credit Risk Prediction — Random Forest')
    st.markdown('''
    **Project:** Credit Risk Prediction using Random Forests (Model Interpretability with SHAP)

    This interactive demo allows you to input applicant details manually or upload a CSV of applicants to get predicted default probability along with interpretability insights (SHAP).
    ''')
    st.markdown('<div class="small-muted">Built with Streamlit • Model: Random Forest • Explainability: SHAP</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2,1])
    with col1:
        st.header('Demo')
        st.write('Try the Predict page to make single or batch predictions. Visit Insights for model-level explanations.')
    with col2:
        st.image('https://raw.githubusercontent.com/streamlit/streamlit/main/frontend/public/logo.png', width=120)


def page_predict(model, scaler, columns):
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.header('Credit Risk Predictor')
    st.write('Choose Manual Input or Upload CSV for batch predictions.')
    st.markdown('</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(['Manual Input', 'Batch CSV Upload'])

    with tab1:
        st.subheader('Manual Applicant Input')
        # Build a dynamic form from columns.json (if it contains feature metadata). We'll assume columns is a list of feature names.
        example_values = {}
        inputs = {}
        with st.form('manual_form'):
            for feat in columns:
                # Simple heuristic for input type
                if feat.lower().endswith(('age','amount','amt','limit','income','bill','pay','bal','credit')):
                    val = st.number_input(feat, value=0.0, format='%f')
                else:
                    val = st.number_input(feat, value=0.0, format='%f')
                inputs[feat] = val
            submitted = st.form_submit_button('Predict', use_container_width=True)
        if submitted:
            df_in = pd.DataFrame([inputs])
            out = predict_df(model, scaler, columns, df_in)
            prob = out['default_prob'].iloc[0]
            pred = out['default_pred'].iloc[0]
            st.metric('Default Probability', f"{prob:.3f}")
            st.markdown('**Risk:** ' + ('🔴 High Risk' if pred==1 else '🟢 Low Risk'))
            # Show SHAP local explanation
            st.subheader('Local Explanation (SHAP)')
            try:
                explainer = shap.TreeExplainer(model)
                background = df_in
                sv = explainer.shap_values(df_in)
                plt.figure(figsize=(8,4))
                shap.plots.bar(sv, df_in, max_display=12, show=False) if hasattr(shap.plots, 'bar') else shap.summary_plot(sv, df_in, plot_type='bar', show=False)
                st.pyplot(plt)
                plt.clf()
            except Exception as e:
                st.error('SHAP explanation failed: ' + str(e))

    with tab2:
        st.subheader('Batch CSV Upload')
        uploaded_file = st.file_uploader('Upload CSV with applicant features (columns must match model features)', type=['csv'])
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.write('Preview of uploaded data:')
            st.dataframe(data.head())
            if st.button('Run Batch Prediction'):
                out = predict_df(model, scaler, columns, data)
                st.dataframe(out.head())
                csv = out.to_csv(index=False).encode('utf-8')
                st.download_button('Download predictions CSV', data=csv, file_name='predictions.csv', mime='text/csv')


def page_insights(model, scaler, columns):
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.header('Model Insights & Explainability')
    st.markdown('</div>', unsafe_allow_html=True)

    # Load processed selected data (if present) for global SHAP
    processed_path = os.path.join('data','processed','processed_selected_top_features.csv')
    if os.path.exists(processed_path):
        df_proc = pd.read_csv(processed_path)
        st.write('Processed dataset preview:')
        st.dataframe(df_proc.head())
    else:
        st.info('Processed dataset not found in ./data/processed/. Global SHAP will use a small sample from input data if available.')

    st.subheader('Global Feature Importance (Model)')
    try:
        fi = pd.Series(model.feature_importances_, index=columns).sort_values(ascending=False)
        st.bar_chart(fi.head(12))
    except Exception as e:
        st.error('Feature importance failed: ' + str(e))

    st.subheader('SHAP Summary (global)')
    if st.button('Generate SHAP Summary'):
        try:
            # Prepare background and test sample
            if 'df_proc' in locals():
                X = df_proc.drop(columns=['default_payment_next_month'], errors='ignore')
            else:
                st.warning('No processed dataset found; please upload sample data via Predict page or place processed CSV in data/processed/ folder.')
                X = None
            if X is not None:
                explainer = shap.TreeExplainer(model)
                # sample to speed up
                Xs = X.sample(min(300, X.shape[0]), random_state=42)
                shap_vals = explainer.shap_values(Xs)
                plt.figure(figsize=(10,6))
                shap.summary_plot(shap_vals, Xs, show=False)
                st.pyplot(plt)
                plt.clf()
        except Exception as e:
            st.error('SHAP summary failed: ' + str(e))

    st.subheader('Local SHAP Explanations')
    st.write('Upload a CSV with observations and pick one index to explain (or use last batch predictions).')
    uploaded = st.file_uploader('Upload CSV for local SHAP', type=['csv'], key='insights_upload')
    if uploaded is not None:
        df_local = pd.read_csv(uploaded)
        st.write(df_local.head())
        idx = st.number_input('Index to explain (0-based)', min_value=0, max_value=max(0, len(df_local)-1), value=0)
        if st.button('Explain index'):
            sample = df_local.iloc[[idx]]
            try:
                explainer = shap.TreeExplainer(model)
                sv = explainer.shap_values(sample)
                plt.figure(figsize=(8,4))
                shap.plots.waterfall(explainer(sample), show=False)
                st.pyplot(plt)
                plt.clf()
            except Exception as e:
                st.error('Local SHAP failed: ' + str(e))


def page_about():
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.title('About this Project')
    st.markdown('''
    **Credit Risk Prediction using Random Forests**

    - Dataset: UCI Credit Card Default (Taiwan)
    - Model: Random Forest (feature selection + tuning)
    - Explainability: SHAP (global & local explanations)

    This app demonstrates a full pipeline from raw data preprocessing to model explainability.
    ''')
    st.markdown('</div>', unsafe_allow_html=True)

    st.subheader('How to run locally')
    st.code('''
    pip install -r requirements.txt
    streamlit run streamlit_app.py
    ''')

# -----------------------------
# Main App
# -----------------------------

def main():
    st.sidebar.title('Navigation')
    pages = ['Home', 'Predict', 'Insights', 'About']
    choice = st.sidebar.radio('Go to', pages)

    model, scaler, columns = load_model()

    if choice == 'Home':
        page_home()
    elif choice == 'Predict':
        if model is None:
            page_home()
            st.error('Model artifacts missing. Place model, scaler, columns in ./model folder.')
        else:
            page_predict(model, scaler, columns)
    elif choice == 'Insights':
        if model is None:
            page_home()
            st.error('Model artifacts missing. Place model, scaler, columns in ./model folder.')
        else:
            page_insights(model, scaler, columns)
    elif choice == 'About':
        page_about()

if __name__ == '__main__':
    main()
