"""Single place for environment-specific settings. Edit HERE, not in notebooks."""
import os

AWS_REGION = os.environ.get("COURSE_AWS_REGION", "us-east-1")   # course default region; free account plan supports SageMaker here
S3_BUCKET = os.environ.get("COURSE_S3_BUCKET", "")               # empty -> sagemaker default bucket

# Bedrock model IDs churn. Verify with:  python scripts/check_aws_access.py --service bedrock-invoke
# One fast/cheap model and one stronger model for contrast in Session 3.
# Both are Amazon Nova models: they work on a new account with no extra step.
# Anthropic models on Bedrock need (1) the one-time Anthropic use-case form per account,
# (2) a payment method valid for AWS Marketplace, and (3) an inference-profile ID with
# the bedrock-runtime endpoint, e.g. "us.anthropic.claude-haiku-4-5-20251001-v1:0".
# The bare ID "anthropic.claude-haiku-4-5" fails with bedrock-runtime Converse.
BEDROCK_MODEL_FAST = os.environ.get("COURSE_BEDROCK_FAST", "amazon.nova-lite-v1:0")
BEDROCK_MODEL_STRONG = os.environ.get("COURSE_BEDROCK_STRONG", "amazon.nova-pro-v1:0")

# USD per 1M tokens (input, output), us-east-1 on-demand. Used only for the cost cell
# in the Session 3 demo. Re-check at aws.amazon.com/bedrock/pricing before class.
BEDROCK_PRICE_PER_1M = {
    "amazon.nova-lite-v1:0": (0.06, 0.24),
    "amazon.nova-pro-v1:0": (0.80, 3.20),
}

# When true, all AWS calls (Bedrock/Rekognition) return recorded
# responses from sessionN/fallback_responses/ so the class never blocks on AWS.
USE_MOCK_AWS = os.environ.get("USE_MOCK_AWS", "0") == "1"

RANDOM_SEED = 42
