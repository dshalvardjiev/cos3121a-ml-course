"""SageMaker SKLearn container entry point for the churn model endpoint (Session 4).
The container calls model_fn once at startup, then input_fn -> predict_fn -> output_fn per request."""
import json, os
import joblib
import pandas as pd


def model_fn(model_dir):
    return joblib.load(os.path.join(model_dir, "model.joblib"))


def input_fn(body, content_type="application/json"):
    if content_type != "application/json":
        raise ValueError(f"unsupported content type {content_type}")
    payload = json.loads(body)
    return pd.DataFrame(payload["instances"])


def predict_fn(df, model):
    proba = model.predict_proba(df)[:, 1]
    return proba


def output_fn(proba, accept="application/json"):
    return json.dumps({"churn_probability": [round(float(p), 4) for p in proba]})
