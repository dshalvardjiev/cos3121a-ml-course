"""Thin AWS wrappers with a mock mode so class never blocks on AWS.
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
    """Recorded responses used when USE_MOCK_AWS is on (or Bedrock unavailable).
    A recorded key matches when the input STARTS WITH it."""
    for f in REPO_ROOT.glob(f"session*/fallback_responses/{kind}.json"):
        bank = json.loads(f.read_text())
        if key in bank:
            return bank[key]
        for k, v in bank.items():
            if not k.startswith("_") and key.startswith(k):
                return v
        return bank.get("_default", "[mock] no recorded response for this input")
    return "[mock] fallback_responses not found"


_MOCK_TURN = {"n": 0}   # cycles through recorded variants at high temperature


def _mock_llm(prompt: str, model: str | None, temperature: float, max_tokens: int) -> str:
    """Replays session3/fallback_responses/llm.json.
    Lookup order: prompt prefix -> keyword rules (_rules) -> _default.
    An entry may be a string, a list (variants; temperature >= 0.5 cycles them),
    or a dict {"fast": ..., "strong": ...} keyed by model tier."""
    bank = {}
    for f in REPO_ROOT.glob("session*/fallback_responses/llm.json"):
        bank = json.loads(f.read_text())
    entry = None
    for k, v in bank.items():
        if not k.startswith("_") and prompt.startswith(k):
            entry = v
            break
    if entry is None:
        low = prompt.lower()
        for rule in bank.get("_rules", []):
            if all(t.lower() in low for t in rule.get("if_all", [])) and \
               (not rule.get("if_any") or any(t.lower() in low for t in rule["if_any"])):
                entry = rule["response"]
                break
    if entry is None:
        entry = bank.get("_default", "[mock] no recorded response for this input")
    if isinstance(entry, dict):
        tier = "strong" if model and model == cfg.BEDROCK_MODEL_STRONG else "fast"
        entry = entry.get(tier) or next(iter(entry.values()))
    if isinstance(entry, list):
        if temperature >= 0.5:
            _MOCK_TURN["n"] += 1
            entry = entry[_MOCK_TURN["n"] % len(entry)]
        else:
            entry = entry[0]
    words = entry.split(" ")
    limit = max(1, int(max_tokens * 0.75))          # ~0.75 words per token
    return " ".join(words[:limit]) if len(words) > limit else entry


def bedrock_chat(prompt: str, model: str | None = None,
                 temperature: float = 0.2, max_tokens: int = 500,
                 return_usage: bool = False):
    """One-call LLM chat via the Bedrock Converse API (or recorded mock).
    return_usage=True returns (text, {"inputTokens", "outputTokens", "stopReason"})."""
    if cfg.USE_MOCK_AWS:
        text = _mock_llm(prompt, model, temperature, max_tokens)
        if not return_usage:
            return text
        out_tok = int(len(text.split()) / 0.75)
        usage = {"inputTokens": int(len(prompt.split()) / 0.75), "outputTokens": out_tok,
                 "stopReason": "max_tokens" if out_tok >= max_tokens - 2 else "end_turn",
                 "note": "mock estimate"}
        return text, usage
    rt = _client("bedrock-runtime")
    resp = rt.converse(
        modelId=model or cfg.BEDROCK_MODEL_FAST,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"temperature": temperature, "maxTokens": max_tokens},
    )
    text = resp["output"]["message"]["content"][0]["text"]
    if not return_usage:
        return text
    u = resp.get("usage", {})
    return text, {"inputTokens": u.get("inputTokens"), "outputTokens": u.get("outputTokens"),
                  "stopReason": resp.get("stopReason")}


# ---------------- Rekognition / text analysis (Session 4) ----------------
def detect_labels(image_path: str, min_confidence: float = 80.0) -> dict:
    if cfg.USE_MOCK_AWS:
        return _mock_response("rekognition", Path(image_path).name)
    if not Path(image_path).exists():
        raise FileNotFoundError(
            f"No image at {image_path}.\n"
            "Rekognition needs a real photograph, so none is committed to the repo "
            "(see session4/assets/README.md). Either drop any rights-cleared product "
            "photo in at that path, or set USE_MOCK_AWS=1 to replay the recorded "
            "response and run the demo without AWS."
        )
    with open(image_path, "rb") as f:
        return _client("rekognition").detect_labels(
            Image={"Bytes": f.read()}, MinConfidence=min_confidence)


ENTITY_TYPES = "ORGANIZATION, PERSON, LOCATION, DATE, COMMERCIAL_ITEM, OTHER"


def analyze_text(text: str) -> dict:
    """Sentiment + entities for a short text, via a Bedrock model (or recorded mock).

    Amazon Comprehend does the same job, but it is not offered on the AWS free account
    plan, so the course uses a Bedrock model instead. Returns
    {"sentiment": "NEGATIVE", "entities": [{"Text": ..., "Type": ...}, ...]}.
    """
    if cfg.USE_MOCK_AWS:
        return _mock_response("text_analysis", text[:60])
    prompt = (
        "Analyze the text below. Reply with JSON only, no other words, in this shape:\n"
        '{"sentiment": "POSITIVE|NEGATIVE|NEUTRAL|MIXED", '
        '"entities": [{"Text": "...", "Type": "' + ENTITY_TYPES.replace(", ", "|") + '"}]}\n\n'
        "Text:\n" + text
    )
    raw = bedrock_chat(prompt, model=cfg.BEDROCK_MODEL_FAST, temperature=0.0, max_tokens=300)
    cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model did not return valid JSON: {raw!r}") from e


def parse_analysis(result: dict) -> dict:
    return {
        "sentiment": result["sentiment"],
        "entities": [(e["Text"], e["Type"]) for e in result["entities"]],
    }


# ---------------- SageMaker deployment (Session 4) ----------------
# The serving container ships scikit-learn but NOT xgboost, and every model in this
# course is an XGBClassifier. SageMaker pip-installs a requirements.txt found in the
# source directory at container start, so we ship one alongside the entry point.
# Without it, model_fn's joblib.load raises ModuleNotFoundError and the endpoint
# never reaches InService - with no obvious error in the notebook.
SERVING_REQUIREMENTS = "xgboost>=2.0\nscikit-learn>=1.3\n"


def deploy_sklearn_model(model_path: str, endpoint_name: str,
                         instance_type: str = "ml.m5.large"):
    """Deploy a local joblib model as a SageMaker real-time endpoint.

    Uses the SKLearn serving container with session4/inference.py as the entry point,
    plus a generated requirements.txt so the container can unpickle XGBoost models.
    """
    from sagemaker.sklearn import SKLearnModel
    import sagemaker, tarfile, tempfile, os, shutil
    sess = sagemaker.Session()
    # The source directory has to still exist when .deploy() packages it, so the
    # whole deployment happens inside the temporary directory.
    with tempfile.TemporaryDirectory() as td:
        tar = os.path.join(td, "model.tar.gz")
        with tarfile.open(tar, "w:gz") as t:
            t.add(model_path, arcname="model.joblib")
        s3_uri = sess.upload_data(tar, bucket=default_bucket(), key_prefix="mlcourse/models")

        src = os.path.join(td, "serving")
        os.makedirs(src, exist_ok=True)
        shutil.copy(str(REPO_ROOT / "session4" / "inference.py"),
                    os.path.join(src, "inference.py"))
        with open(os.path.join(src, "requirements.txt"), "w") as f:
            f.write(SERVING_REQUIREMENTS)

        model = SKLearnModel(model_data=s3_uri, role=sagemaker.get_execution_role(),
                             source_dir=src, entry_point="inference.py",
                             framework_version="1.2-1")
        predictor = model.deploy(initial_instance_count=1, instance_type=instance_type,
                                 endpoint_name=endpoint_name)
    print(f"endpoint '{endpoint_name}' is deploying - this bills hourly; delete after class!")
    print("  (first start takes ~2 min longer than you expect: the container pip-installs xgboost)")
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
