"""Save/load model artifacts. The file S2 saves is the file S4 deploys and S5 audits."""
from pathlib import Path
import json, joblib

MODELS = Path(__file__).resolve().parents[2] / "models"


def save_model(model, name: str, metrics: dict | None = None) -> Path:
    MODELS.mkdir(parents=True, exist_ok=True)
    path = MODELS / f"{name}.joblib"
    joblib.dump(model, path)
    if metrics:
        (MODELS / f"{name}_metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    print(f"saved -> {path}")
    return path


def load_model(name: str):
    path = MODELS / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run the Session 2 demo notebook that trains it")
    return joblib.load(path)
