import os
import sys
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from lime import lime_tabular

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(BASE_DIR)
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
STATIC_OUTPUT = os.path.join(OUTPUT_DIR, 'user')
os.makedirs(STATIC_OUTPUT, exist_ok=True)

# Load pipeline components once at startup
from src.data.loader import download_and_load_data, inspect_dataset
from src.data.preprocessing import preprocess_data, ITEM_LABEL_MAP
from src.models.baseline_model import train_baseline_model

# Load data and train model (cached for the Flask process)
print("[FLASK] Loading dataset and training model…")
_df_clean = download_and_load_data(raw_dir=os.path.join(BASE_DIR, 'data', 'raw'), force_download=False)
_X_train, _X_test, y_train, _y_test, feature_names = preprocess_data(_df_clean, test_size=0.2, random_state=42)
model = train_baseline_model(_X_train, y_train, model_type='random_forest', random_state=42)
print("[FLASK] Model ready.")

def preprocess_user_input(raw_dict: dict) -> pd.DataFrame:
    """Convert raw user dict (same keys as raw CSV) into the preprocessed feature frame.
    Returns a single‑row DataFrame with the same column names as the training X.
    """
    df = pd.DataFrame([raw_dict])
    # Standardize Q items (0‑3)
    q_cols = [c for c in df.columns if c.startswith('Q') and c.endswith('A')]
    df[q_cols] = df[q_cols].apply(lambda col: col.astype(float) - 1)
    # Fill missing with median of training data
    df = df.fillna(_X_train.median())
    # Rename to readable for visualisations (SHAP/LIME use original column names)
    return df[feature_names]

def compute_shap_explanation(model, X_train, X_user, feature_names, out_dir, uid):
    # Use TreeExplainer if possible, fallback to generic
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_user)
    except Exception:
        background = X_train.sample(n=min(100, len(X_train)), random_state=42)
        explainer = shap.Explainer(model.predict_proba, background)
        shap_values = explainer(X_user)
    # Binary class handling
    if len(shap_values.shape) == 3:
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values
    # Local plot
    plt.figure(figsize=(10, 6))
    values = sv.values[0]
    top_idx = np.argsort(np.abs(values))[-10:]
    colors = ['#d73027' if v > 0 else '#4575b4' for v in values[top_idx]]
    plt.barh([feature_names[i] for i in top_idx], values[top_idx], color=colors)
    plt.axvline(0, color='gray', linestyle='--')
    plt.title('Your SHAP Local Explanation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    shap_path = os.path.join(out_dir, f'shap_local_{uid}.png')
    plt.savefig(shap_path, dpi=300, bbox_inches='tight')
    plt.close()
    # Return probability and plot path
    prob = model.predict_proba(X_user)[0, 1] if hasattr(model, 'predict_proba') else model.predict(X_user)[0]
    pred_label = 'Moderate‑to‑Severe Depression Risk (1)' if prob >= 0.5 else 'Normal/Mild Depression (0)'
    return prob, pred_label, shap_path

def compute_lime_explanation(model, X_train, X_user, feature_names, out_dir, uid):
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=feature_names,
        class_names=['Normal/Mild (0)', 'Mod‑Severe (1)'],
        mode='classification',
        random_state=42
    )
    exp = explainer.explain_instance(
        data_row=np.array(X_user.iloc[0]),
        predict_fn=model.predict_proba,
        num_features=10,
        labels=(0, 1)
    )
    label = 1 if 1 in exp.available_labels() else exp.available_labels()[0]
    explanation = exp.as_list(label=label)
    # Plot
    plt.figure(figsize=(10, 6))
    feats = [t[0] for t in explanation][::-1]
    weights = [t[1] for t in explanation][::-1]
    colors = ['#d73027' if w > 0 else '#4575b4' for w in weights]
    plt.barh(feats, weights, color=colors)
    plt.axvline(0, color='gray', linestyle='--')
    plt.title('Your LIME Local Explanation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    lime_path = os.path.join(out_dir, f'lime_local_{uid}.png')
    plt.savefig(lime_path, dpi=300, bbox_inches='tight')
    plt.close()
    return lime_path, explanation

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/assessment', methods=['GET', 'POST'])
def assessment():
    if request.method == 'POST':
        # Build raw dict from form values
        raw = {}
        for col in feature_names:
            val = request.form.get(col)
            if val is None:
                flash(f'Missing value for {col}')
                return redirect(url_for('assessment'))
            # Convert to appropriate type (numeric)
            try:
                raw[col] = float(val)
            except ValueError:
                raw[col] = val
        # Preprocess
        X_user = preprocess_user_input(raw)
        # Prediction & explanations
        uid = str(int(pd.Timestamp.utcnow().timestamp()))
        prob, pred_label, shap_path = compute_shap_explanation(model, _X_train, X_user, feature_names, STATIC_OUTPUT, uid)
        lime_path, lime_expl = compute_lime_explanation(model, _X_train, X_user, feature_names, STATIC_OUTPUT, uid)
        # Store results in session (simple approach using query params)
        results = {
            'prob': f"{prob:.4f}",
            'label': pred_label,
            'shap_img': os.path.relpath(shap_path, OUTPUT_DIR).replace('\\', '/'),
            'lime_img': os.path.relpath(lime_path, OUTPUT_DIR).replace('\\', '/'),
            'lime_exp': lime_expl,
        }
        return render_template('results.html', results=results)
    else:
        # GET – render questionnaire using readable labels
        readable = [ITEM_LABEL_MAP.get(col, col) for col in feature_names]
        items = list(zip(feature_names, readable))
        return render_template('assessment.html', items=items)

@app.route('/outputs/<path:filename>')
def serve_output(filename):
    return send_from_directory(OUTPUT_DIR, filename)

@app.route('/ping')
def ping():
    return 'pong'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
