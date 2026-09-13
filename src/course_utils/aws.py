"""Thin AWS wrappers with a mock mode so class never blocks on the sandbox.
Everything defaults to config/course_config.py settings (region us-east-1)."""
import json, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "config"))
import course_config as cfg  # noqa: E402


def _client(service):
    import boto3
    return boto3.client(service, region_name=cfg.AWS_REGION)


def default_bucket() -> str:
    if cfg.S3_BUCKET:
        return cfg.S3_BUCKET
    import sagemaker
    return sagemaker.Session().default_bucket()


# ---------------- Bedrock (Session 3) ----------------
def _mock_response(kind: str, key: str):
    """Recorded responses used when USE_MOCK_AWS is on (or Bedrock unavailable)."""
    for f in REPO_ROOT.glob(f"session*/fallback_responses/{kind}.json"):
        bank = json.loads(f.read_text())
        if key in bank:
            return bank[key]
        return bank.get("_default", "[mock] no recorded response for this input")
    return "[mock] fallback_responses not found"


def bedrock_chat(prompt: str, model: str | None = None,
                 temperature: float = 0.2, max_tokens: int = 500) -> str:
    """One-call LLM chat via the Bedrock Converse API (or recorded mock)."""
    if cfg.USE_MOCK_AWS:
        return _mock_response("llm", prompt[:60])
    rt = _client("bedrock-runtime")
    resp = rt.converse(
        modelId=model or cfg.BEDROCK_MODEL_FAST,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"temperature": temperature, "maxTokens": max_tokens},
    )
    return resp["output"]["message"]["content"][0]["text"]


# ---------------- Rekognition / Comprehend (Session 4) ----------------
def detect_labels(image_path: str, min_confidence: float = 80.0) -> dict:
    if cfg.USE_MOCK_AWS:
        return _mock_response("rekognition", Path(image_path).name)
    with open(image_path, "rb") as f:
        return _client("rekognition").detect_labels(
            Image={"Bytes": f.read()}, MinConfidence=min_confidence)


def analyze_text(text: str) -> dict:
    """Sentiment + entities in one dict (Comprehend or mock)."""
    if cfg.USE_MOCK_AWS:
        return _mock_response("comprehend", text[:60])
    com = _client("comprehend")
    return {
        "sentiment": com.detect_sentiment(Text=text, LanguageCode="en"),
        "entities": com.detect_entities(Text=text, LanguageCode="en")["Entities"],
    }


def parse_comprehend(result: dict) -> dict:
    s = result["sentiment"]
    return {
        "sentiment": s["Sentiment"],
        "confidence": round(max(s["SentimentScore"].values()), 3),
        "entities": [(e["Text"], e["Type"]) for e in result["entities"]],
    }


# ---------------- SageMaker deployment (Session 4) ----------------
def deploy_sklearn_model(model_path: str, endpoint_name: str,
                         instance_type: str = "ml.m5.large"):
    """Deploy a local joblib model as a SageMaker real-time endpoint.
    Uses the SKLearnModel container; entry point session4/inference.py."""
    from sagemaker.sklearn import SKLearnModel
    import sagemaker, tarfile, tempfile, os
    sess = sagemaker.Session()
    with tempfile.TemporaryDirectory() as td:
        tar = os.path.join(td, "model.tar.gz")
        with tarfile.open(tar, "w:gz") as t:
            t.add(model_path, arcname="model.joblib")
        s3_uri = sess.upload_data(tar, bucket=default_bucket(), key_prefix="mlcourse/models")
    model = SKLearnModel(model_data=s3_uri, role=sagemaker.get_execution_role(),
                         entry_point=str(REPO_ROOT / "session4" / "inference.py"),
                         framework_version="1.2-1")
    predictor = model.deploy(initial_instance_count=1, instance_type=instance_type,
                             endpoint_name=endpoint_name)
    print(f"endpoint '{endpoint_name}' is deploying - this bills hourly; delete after class!")
    return predictor


def invoke_endpoint(endpoint_name: str, payload: dict) -> dict:
    rt = _client("sagemaker-runtime")
    resp = rt.invoke_endpoint(EndpointName=endpoint_name, ContentType="application/json",
                              Body=json.dumps(payload))
    return json.loads(resp["Body"].read())


def delete_endpoint(endpoint_name: str):
    """THE most important function in this module. Endpoints bill while idle."""
    sm = _client("sagemaker")
    for fn, kw in ((sm.delete_endpoint, {"EndpointName": endpoint_name}),
                   (sm.delete_endpoint_config, {"EndpointConfigName": endpoint_name})):
        try:
            fn(**kw)
        except Exception as e:
            print(f"(already gone?) {e}")
    print(f"deleted endpoint + config: {endpoint_name}")
