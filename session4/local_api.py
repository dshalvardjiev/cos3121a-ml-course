#!/usr/bin/env python3
"""The Session 4 endpoint pattern without AWS: same model, local FastAPI.
Run:  uvicorn session4.local_api:app --reload   (from repo root)
Call: curl -X POST localhost:8000/predict -H 'content-type: application/json' \
        -d '{"instances": [{...one-hot feature dict...}]}'
"""
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from course_utils.models import load_model

app = FastAPI(title="Churn model - local endpoint twin")
model = load_model("churn_xgb_v1")
COLUMNS = list(getattr(model, "feature_names_in_", []))


class Payload(BaseModel):
    instances: list[dict]


@app.post("/predict")
def predict(p: Payload):
    df = pd.DataFrame(p.instances)
    if COLUMNS:
        df = df.reindex(columns=COLUMNS, fill_value=0)
    proba = model.predict_proba(df)[:, 1]
    return {"churn_probability": [round(float(x), 4) for x in proba]}
