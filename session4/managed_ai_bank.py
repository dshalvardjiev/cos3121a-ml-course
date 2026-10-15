"""Amazon Bedrock step for the Bank Marketing pipeline.

Takes the SageMaker result (subscribe_probability, call) and turns it into a short
marketing recommendation. Bedrock does NOT predict or change the call decision.

Usage from the API:
    from session4.managed_ai_bank import generate_marketing_recommendation
    prediction = resp["predictions"][0]
    result = generate_marketing_recommendation(
        prediction["subscribe_probability"], prediction["call"])
"""
import json
import os

import boto3
from botocore.exceptions import BotoCoreError, ClientError

# Same defaults as config/course_config.py (BEDROCK_MODEL_FAST, AWS_REGION).
REGION = os.environ.get("COURSE_AWS_REGION", "us-east-1")
MODEL_ID = os.environ.get("COURSE_BEDROCK_FAST", "amazon.nova-lite-v1:0")
CUTOFF = 0.20
REQUIRED_FIELDS = {"marketing_summary", "recommended_action"}

# Credentials come from the standard AWS chain (env vars / ~/.aws), never from code.
bedrock = boto3.client("bedrock-runtime", region_name=REGION)


class BedrockRecommendationError(RuntimeError):
    """Raised when Bedrock fails or returns an unusable response."""


def build_prompt(subscribe_probability, call):
    pct = f"{subscribe_probability * 100:.2f}%"
    call_text = "true" if call else "false"
    return f"""You are assisting a bank marketing employee.

A machine learning model predicts the probability that a customer will subscribe to a term deposit.

Model prediction:
- Predicted subscription probability: {pct}
- Business decision: call = {call_text}

The business cutoff is {CUTOFF:.2f}:
  - probability >= {CUTOFF:.2f} means call = true
  - probability < {CUTOFF:.2f} means call = false

The call decision has already been made by the predictive model. Do not change it.

Your responsibility:
- Provide a concise, human-readable explanation of the model's numerical prediction.
- Suggest a practical action for the bank employee.
- Respect the existing call decision (call = {call_text}).

Strictly forbidden:
- Do not recalculate or change the probability.
- Do not make a new prediction or decision.
- Do not override call = {call_text}.
- Do not invent customer information.
- Do not claim to know which features caused this prediction.
- Do not introduce a risk or priority classification.

Return ONLY valid JSON (no markdown, no extra text) with exactly these fields:
{{
  "marketing_summary": "one short sentence explaining the prediction",
  "recommended_action": "one short sentence for the employee to take"
}}"""


def _parse_response(text):
    cleaned = text.strip()
    # Some models wrap JSON in ```json fences despite instructions.
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise BedrockRecommendationError(
            f"Bedrock returned invalid JSON: {e}. Raw response: {text!r}") from e

    if not isinstance(result, dict):
        raise BedrockRecommendationError(
            f"Bedrock JSON must be a dict. Got: {type(result).__name__}")
    
    if set(result) != REQUIRED_FIELDS:
        raise BedrockRecommendationError(
            f"Bedrock JSON must have exactly {sorted(REQUIRED_FIELDS)}. Got: {sorted(result.keys())}")
    
    for field in REQUIRED_FIELDS:
        if not isinstance(result[field], str) or not result[field].strip():
            raise BedrockRecommendationError(
                f"Field {field!r} must be a non-empty string. Got: {result[field]!r}")
    
    return result


def generate_marketing_recommendation(subscribe_probability, call):
    """Return {"marketing_summary", "recommended_action"} for one prediction."""
    prompt = build_prompt(subscribe_probability, call)
    try:
        response = bedrock.converse(
            modelId=MODEL_ID,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"temperature": 0.1, "maxTokens": 300},
        )
        text = response["output"]["message"]["content"][0]["text"]
    except (ClientError, BotoCoreError) as e:
        raise BedrockRecommendationError(
            f"AWS/Bedrock call failed (model={MODEL_ID}, region={REGION}): {e}") from e
    except (KeyError, IndexError, TypeError) as e:
        raise BedrockRecommendationError(
            f"Unexpected Bedrock response structure: {e}") from e
    return _parse_response(text)


if __name__ == "__main__":
    test_cases = [
        (0.0393, False),
        (0.19, False),
        (0.20, True),
        (0.50, True),
        (0.9388, True),
    ]
    for prob, call in test_cases:
        print(f"\nsubscribe_probability={prob}  call={call}")
        try:
            print(json.dumps(generate_marketing_recommendation(prob, call), indent=2))
        except BedrockRecommendationError as err:
            print(f"ERROR: {err}")
