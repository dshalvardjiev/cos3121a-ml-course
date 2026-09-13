#!/usr/bin/env python3
"""Pre-flight: which AWS services can this account actually reach?
Run before every session (AWS Academy sandboxes vary). Usage:
    python scripts/check_aws_access.py [--service bedrock]
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
    print(f"region: {cfg.AWS_REGION} | mock mode: {cfg.USE_MOCK_AWS}")


if __name__ == "__main__":
    main()
