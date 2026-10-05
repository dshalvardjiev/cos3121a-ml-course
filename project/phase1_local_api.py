from course_utils.models import load_model
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from pathlib import Path
import sys

# Make course utilities available
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


app = FastAPI(title="Bank Marketing Model - Local Endpoint")

# Load your saved Phase 1 model
model = load_model("phase1_bank_marketing")

# Get the feature names expected by the model
COLUMNS = list(getattr(model, "feature_names_in_", []))


class Payload(BaseModel):
    instances: list[dict]


@app.post("/predict")
def predict(p: Payload):
    df = pd.DataFrame(p.instances)

    if COLUMNS:
        df = df.reindex(columns=COLUMNS, fill_value=0)

    probabilities = model.predict_proba(df)[:, 1]

    return {
        "subscription_probability": [
            round(float(x), 4) for x in probabilities
        ]
    }
