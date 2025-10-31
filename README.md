# Credit Risk Predictor (Streamlit)

This is a single-file Streamlit demo for credit risk prediction using a Random Forest model with SHAP explainability.

Files added:

- `streamlit_app.py` — main Streamlit app.
- `requirements.txt` — Python dependencies.
- `model/columns.json` — example features file (replace with your real columns).

Place your trained model artifacts in the `model/` folder:

- `model.pkl` — trained scikit-learn RandomForest (or similar) saved with `joblib`.
- `scaler.pkl` — optional scaler used during training (StandardScaler, etc.).
- `columns.json` — JSON list of feature names in the order expected by the model.

Run locally:

```powershell
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

Notes:
- The app will show an error if model artifacts are missing. Add them to `./model/` before using Predict/Insights.
- For production, split the app into modules, secure model artifacts, and add tests.
