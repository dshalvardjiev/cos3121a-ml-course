
import json
import os
from typing import Literal

import boto3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from session4.managed_ai_bank import (
    generate_marketing_recommendation,
    BedrockRecommendationError,
)

app = FastAPI(title="Bank Marketing ML Application")

ENDPOINT_NAME = os.getenv(
    "SAGEMAKER_ENDPOINT_NAME",
    "phase1-bank-marketing",
)
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

session = boto3.Session()
runtime = session.client(
    "sagemaker-runtime",
    region_name=AWS_REGION,
)


class BankCustomer(BaseModel):
    age: int
    balance: int
    day: int
    campaign: int
    pdays: int
    previous: int

    job: Literal[
        "admin.", "blue-collar", "entrepreneur", "housemaid",
        "management", "retired", "self-employed", "services",
        "student", "technician", "unemployed", "unknown",
    ]
    marital: Literal["divorced", "married", "single"]
    education: Literal["primary", "secondary", "tertiary", "unknown"]
    default: Literal["yes", "no"]
    housing: Literal["yes", "no"]
    loan: Literal["yes", "no"]
    contact: Literal["cellular", "telephone", "unknown"]
    month: Literal[
        "jan", "feb", "mar", "apr", "may", "jun",
        "jul", "aug", "sep", "oct", "nov", "dec",
    ]
    poutcome: Literal["failure", "other", "success", "unknown"]


class PredictionRequest(BaseModel):
    customers: list[BankCustomer] = Field(min_length=1)


NUMERIC_FEATURES = [
    "age", "balance", "day", "campaign",
    "pdays", "previous", "prev_contacted",
]

DUMMY_FEATURES = [
    "job_blue-collar", "job_entrepreneur", "job_housemaid",
    "job_management", "job_retired", "job_self-employed",
    "job_services", "job_student", "job_technician",
    "job_unemployed", "job_unknown",
    "marital_married", "marital_single",
    "education_secondary", "education_tertiary",
    "education_unknown",
    "default_yes", "housing_yes", "loan_yes",
    "contact_telephone", "contact_unknown",
    "month_aug", "month_dec", "month_feb", "month_jan",
    "month_jul", "month_jun", "month_mar", "month_may",
    "month_nov", "month_oct", "month_sep",
    "poutcome_other", "poutcome_success", "poutcome_unknown",
]

FEATURES = NUMERIC_FEATURES + DUMMY_FEATURES

# Categories omitted by drop_first=True are the baselines.
DUMMY_CATEGORIES = {
    "job": set(DUMMY_FEATURES[0:11]),
    "marital": {"marital_married", "marital_single"},
    "education": {
        "education_secondary", "education_tertiary",
        "education_unknown",
    },
    "default": {"default_yes"},
    "housing": {"housing_yes"},
    "loan": {"loan_yes"},
    "contact": {"contact_telephone", "contact_unknown"},
    "month": {
        "month_aug", "month_dec", "month_feb", "month_jan",
        "month_jul", "month_jun", "month_mar", "month_may",
        "month_nov", "month_oct", "month_sep",
    },
    "poutcome": {
        "poutcome_other", "poutcome_success", "poutcome_unknown",
    },
}


def encode_customer(customer: BankCustomer) -> dict:
    raw = customer.model_dump()

    # Initialize all 42 features, including every dummy column.
    row = {feature: 0 for feature in FEATURES}

    for feature in ("age", "balance", "day", "campaign", "previous"):
        row[feature] = raw[feature]

    # The training preprocessing treats -1 as "never contacted".
    row["prev_contacted"] = int(raw["pdays"] != -1)
    row["pdays"] = 999 if raw["pdays"] == -1 else raw["pdays"]

    for category, value in raw.items():
        if category in DUMMY_CATEGORIES:
            feature = f"{category}_{value}"
            if feature in DUMMY_CATEGORIES[category]:
                row[feature] = 1

    # Preserve the exact feature order expected by the model.
    return {feature: row[feature] for feature in FEATURES}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest):
    instances = [
        encode_customer(customer)
        for customer in request.customers
    ]

    # Step 1: Ask SageMaker for predictions.
    try:
        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Accept="application/json",
            Body=json.dumps({"instances": instances}),
        )
        model_result = json.loads(response["Body"].read())

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="The SageMaker prediction request failed.",
        ) from exc

    predictions = model_result.get("predictions")

    if (
        not isinstance(predictions, list)
        or len(predictions) != len(instances)
    ):
        raise HTTPException(
            status_code=502,
            detail="The model returned an unexpected response.",
        )

    results = []

    # Step 2: Ask Bedrock for a marketing recommendation.
    for prediction in predictions:
        try:
            recommendation = generate_marketing_recommendation(
                prediction["subscribe_probability"],
                prediction["call"],
            )
            recommendation_status = "success"

        except BedrockRecommendationError:
            # Keep the valid prediction if Bedrock is unavailable.
            recommendation = None
            recommendation_status = "unavailable"

        results.append({
            "subscribe_probability":
                prediction["subscribe_probability"],
            "call": prediction["call"],
            "ai_recommendation": recommendation,
            "recommendation_status": recommendation_status,
        })

    return {"predictions": results}
