#!/usr/bin/env python3
"""Pre-flight: which AWS services can this account actually reach?
Run before every session (AWS access varies). Usage:
    python scripts/check_aws_access.py [--service bedrock|bedrock-invoke]
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "config"))
import course_config as cfg

CHECKS = {
    "s3":        lambda c: c.list_buckets(),
    "sagemaker": lambda c: c.list_endpoints(MaxResults=1),
    "bedrock":   lambda c: c.list_foundation_models(),
    "rekognition": lambda c: c.list_collections(MaxResults=1),
    "comprehend":  lambda c: c.list_document_classifiers(MaxResults=1),
}


def _invoke_probe(model_id):
    """A real 5-token call. Listing models does not prove you can invoke them."""
    import boto3
    rt = boto3.client("bedrock-runtime", region_name=cfg.AWS_REGION)
    rt.converse(modelId=model_id,
                messages=[{"role": "user", "content": [{"text": "Say OK."}]}],
                inferenceConfig={"maxTokens": 5, "temperature": 0.0})


def main():
    import boto3
    only = sys.argv[sys.argv.index("--service") + 1] if "--service" in sys.argv else None
    for svc, probe in CHECKS.items():
        if only and svc != only:
            continue
        try:
            probe(boto3.client(svc, region_name=cfg.AWS_REGION))
            print(f"  OK    {svc}")
        except Exception as e:
            print(f"  FAIL  {svc}: {type(e).__name__} - {str(e)[:90]}")
    if only in (None, "bedrock-invoke"):
        for label, mid in (("fast", cfg.BEDROCK_MODEL_FAST), ("strong", cfg.BEDROCK_MODEL_STRONG)):
            try:
                _invoke_probe(mid)
                print(f"  OK    bedrock-invoke {label}: {mid}")
            except Exception as e:
                print(f"  FAIL  bedrock-invoke {label}: {mid}: {type(e).__name__} - {str(e)[:90]}")
    print(f"region: {cfg.AWS_REGION} | mock mode: {cfg.USE_MOCK_AWS}")


if __name__ == "__main__":
    main()
