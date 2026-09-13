"""Single place for environment-specific settings. Edit HERE, not in notebooks."""
import os

AWS_REGION = os.environ.get("COURSE_AWS_REGION", "us-east-1")   # course default region; free account plan supports SageMaker here
S3_BUCKET = os.environ.get("COURSE_S3_BUCKET", "")               # empty -> sagemaker default bucket

# Bedrock model IDs churn quarterly. Verify with:  aws bedrock list-foundation-models
# Pick one fast/cheap model and one stronger model for contrast in Session 3.
BEDROCK_MODEL_FAST = os.environ.get("COURSE_BEDROCK_FAST", "amazon.nova-lite-v1:0")
BEDROCK_MODEL_STRONG = os.environ.get("COURSE_BEDROCK_STRONG", "anthropic.claude-haiku-4-5")

# When true, all AWS calls (Bedrock/Rekognition/Comprehend) return recorded
# responses from sessionN/fallback_responses/ so the class never blocks on AWS.
USE_MOCK_AWS = os.environ.get("USE_MOCK_AWS", "0") == "1"

RANDOM_SEED = 42
