
import json
from unittest.mock import patch

from fastapi.testclient import TestClient
from session4 import bank_marketing_api as api


client = TestClient(api.app)


def fake_sagemaker_response(instances):
    """Simulate a successful SageMaker response."""
    return {
        "predictions": [
            {
                "subscribe_probability": 0.73,
                "call": True,
            }
            for _ in instances
        ]
    }


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_prediction_and_recommendation():
    with (
        patch.object(api.runtime, "invoke_endpoint") as mock_endpoint,
        patch.object(
            api,
            "generate_marketing_recommendation",
            return_value={
                "marketing_summary": "High likelihood of subscription.",
                "recommended_action": "Prioritize outreach.",
            },
        ) as mock_bedrock,
    ):
        mock_endpoint.return_value = {
            "Body": type(
                "FakeBody",
                (),
                {
                    "read": lambda self: json.dumps(
                        fake_sagemaker_response(
                            json.loads(
                                mock_endpoint.call_args.kwargs["Body"]
                            )["instances"]
                        )
                    ).encode()
                },
            )()
        }

        response = client.post(
            "/predict",
            json={
                "customers": [
                    {
                        "age": 35,
                        "balance": 1500,
                        "day": 15,
                        "campaign": 1,
                        "pdays": -1,
                        "previous": 0,
                        "job": "management",
                        "marital": "married",
                        "education": "tertiary",
                        "default": "no",
                        "housing": "yes",
                        "loan": "no",
                        "contact": "cellular",
                        "month": "may",
                        "poutcome": "unknown",
                    }
                ]
            },
        )

        assert response.status_code == 200, response.text

        result = response.json()["predictions"][0]
        assert result["subscribe_probability"] == 0.73
        assert result["call"] is True
        assert result["ai_recommendation"]["recommended_action"] == (
            "Prioritize outreach."
        )
        assert result["recommendation_status"] == "success"

        # Verify the exact request sent to SageMaker.
        args = mock_endpoint.call_args.kwargs
        assert args["EndpointName"] == "phase1-bank-marketing"
        assert args["ContentType"] == "application/json"

        payload = json.loads(args["Body"])
        instance = payload["instances"][0]

        assert len(instance) == 42
        assert all(isinstance(v, (int, float)) for v in instance.values())
        assert instance["pdays"] == 999
        assert instance["prev_contacted"] == 0
        assert instance["job_management"] == 1
        assert instance["marital_married"] == 1
        assert instance["month_may"] == 1

        mock_bedrock.assert_called_once_with(0.73, True)


def test_bedrock_failure_does_not_lose_prediction():
    with (
        patch.object(api.runtime, "invoke_endpoint") as mock_endpoint,
        patch.object(
            api,
            "generate_marketing_recommendation",
            side_effect=api.BedrockRecommendationError("Simulated failure"),
        ),
    ):
        mock_endpoint.return_value = {
            "Body": type(
                "FakeBody",
                (),
                {
                    "read": lambda self: json.dumps(
                        {"predictions": [{
                            "subscribe_probability": 0.15,
                            "call": False,
                        }]}
                    ).encode()
                },
            )()
        }

        response = client.post(
            "/predict",
            json={
                "customers": [{
                    "age": 40,
                    "balance": 2000,
                    "day": 10,
                    "campaign": 2,
                    "pdays": 20,
                    "previous": 1,
                    "job": "admin.",
                    "marital": "divorced",
                    "education": "primary",
                    "default": "no",
                    "housing": "no",
                    "loan": "no",
                    "contact": "cellular",
                    "month": "apr",
                    "poutcome": "failure",
                }]
            },
        )

        assert response.status_code == 200, response.text
        result = response.json()["predictions"][0]
        assert result["subscribe_probability"] == 0.15
        assert result["call"] is False
        assert result["ai_recommendation"] is None
        assert result["recommendation_status"] == "unavailable"


if __name__ == "__main__":
    test_health()
    test_prediction_and_recommendation()
    test_bedrock_failure_does_not_lose_prediction()
    print("ALL LOCAL API TESTS PASSED")
