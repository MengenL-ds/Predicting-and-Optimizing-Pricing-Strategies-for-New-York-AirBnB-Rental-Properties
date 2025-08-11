from flask import Flask, request, render_template
import joblib, pandas as pd, numpy as np
import os, sys, warnings

# Make custom transformers importable before loading
sys.path.append("../")
from custom_transformers import HostSinceTransformer, BathroomExtractor, UKHostBinaryEncoder, HostResponseOrdinalEncoder

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
    category=UserWarning,
)

app = Flask(__name__)
BASE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE, "../model/final_model.pkl")  # full pipeline

model = joblib.load(MODEL_PATH)
RAW_FEATURES = list(model.input_schema_)  # schema saved during training

def build_row(payload, expected):
    base = {c: np.nan for c in expected}
    for k, v in (payload or {}).items():
        if k in base:
            base[k] = v
    return pd.DataFrame([base], columns=expected)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", features=RAW_FEATURES)

@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or request.form.to_dict(flat=True)
    X_raw = build_row(payload, RAW_FEATURES)
    y_log = model.predict(X_raw)                   # prediction in log scale
    y_price = np.exp(y_log)                        # revert log -> original
    return render_template(
        "index.html",
        features=RAW_FEATURES,
        prediction=float(np.squeeze(y_price))
    )


@app.route("/health")
def health():
    return {"status": "ok", "features_expected": len(RAW_FEATURES)}, 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True, threaded=True)
