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
    "comprehend":  lambda c: c.detect_sentiment(Text="ok", LanguageCode="en"),
}

# Services that are expected to fail on the AWS free account plan. A failure here prints
# N/A, not FAIL. Comprehend is not offered on the free plan; Session 4 uses Bedrock for the
# same sentiment/entities task (see course_utils.aws.analyze_text).
OPTIONAL = {"comprehend": "not on the free account plan; Session 4 uses Bedrock instead"}


def _invoke_probe(model_id):
    """A real 5-token call. Listing models does not prove you can invoke them."""
    import boto3
    rt = boto3.client("bedrock-runtime", region_name=cfg.AWS_REGION)
    rt.converse(modelId=model_id,
                messages=[{"role": "user", "content": [{"text": "Say OK."}]}],
                inferenceConfig={"maxTokens": 5, "temperature": 0.0})


def _identity():
    """Show who boto3 resolves to. A failure here means credentials/profile are
    wrong (e.g. a profile whose role_arn cannot be assumed), not the service."""
    import os
    import boto3
    print(f"  profile: {os.environ.get('AWS_PROFILE') or os.environ.get('AWS_DEFAULT_PROFILE') or '(default chain)'}")
    try:
        who = boto3.client("sts", region_name=cfg.AWS_REGION).get_caller_identity()
        print(f"  identity: {who['Arn']}")
        return True
    except Exception as e:
        print(f"  FAIL  credentials: {type(e).__name__} - {e}")
        print("  Fix credentials first (aws configure list; ~/.aws/config); service checks skipped.")
        return False


def main():
    import boto3
    if not cfg.USE_MOCK_AWS and not _identity():
        print(f"region: {cfg.AWS_REGION} | mock mode: {cfg.USE_MOCK_AWS}")
        return
    only = sys.argv[sys.argv.index("--service") + 1] if "--service" in sys.argv else None
    for svc, probe in CHECKS.items():
        if only and svc != only:
            continue
        try:
            probe(boto3.client(svc, region_name=cfg.AWS_REGION))
            print(f"  OK    {svc}")
        except Exception as e:
            if svc in OPTIONAL:
                print(f"  N/A   {svc}: {OPTIONAL[svc]}")
            else:
                print(f"  FAIL  {svc}: {type(e).__name__} - {str(e)[:300]}")
    if only in (None, "bedrock-invoke"):
        for label, mid in (("fast", cfg.BEDROCK_MODEL_FAST), ("strong", cfg.BEDROCK_MODEL_STRONG)):
            try:
                _invoke_probe(mid)
                print(f"  OK    bedrock-invoke {label}: {mid}")
            except Exception as e:
                print(f"  FAIL  bedrock-invoke {label}: {mid}: {type(e).__name__} - {str(e)[:300]}")
    print(f"region: {cfg.AWS_REGION} | mock mode: {cfg.USE_MOCK_AWS}")


if __name__ == "__main__":
    main()
