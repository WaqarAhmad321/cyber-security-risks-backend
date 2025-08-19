from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, conint
import joblib
import os
from typing import Dict, List

# Model paths
MODEL_PATHS = {
    "phishing_risk": "models/phishing_risk_model.pkl",
    "weak_password_risk": "models/weak_password_risk_model.pkl",
    "oversharing_risk": "models/oversharing_risk_model.pkl",
    "emotional_manipulation_risk": "models/emotional_manipulation_risk_model.pkl",
    "update_ignorance_risk": "models/update_ignorance_risk_model.pkl"
}

# Load models
models: Dict[str, object] = {}
for risk, path in MODEL_PATHS.items():
    if not os.path.exists(path):
        raise RuntimeError(f"Model file not found: {path}")
    models[risk] = joblib.load(path)

app = FastAPI()

# Enable CORS for production
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

class PersonalityInput(BaseModel):
    openness: conint(ge=0, le=100)
    conscientiousness: conint(ge=0, le=100)
    extraversion: conint(ge=0, le=100)
    agreeableness: conint(ge=0, le=100)
    neuroticism: conint(ge=0, le=100)

class RiskScores(BaseModel):
    phishing_risk: float
    weak_password_risk: float
    oversharing_risk: float
    emotional_manipulation_risk: float
    update_ignorance_risk: float


@app.get("/")
def root():
    return {
        "status": "healthy",
        "models_loaded": len(models),
        "available_predictions": list(models.keys())
    }

@app.post("/predict", response_model=RiskScores)
def predict_risks(input: PersonalityInput):
    # Base feature vector from request (ensure floats)
    base_features = [
        float(input.openness),
        float(input.conscientiousness),
        float(input.extraversion),
        float(input.agreeableness),
        float(input.neuroticism),
    ]

    results = {}
    for risk, model in models.items():
        # Work on a copy for each model
        features = base_features.copy()

        # Try to detect expected feature count from common attributes
        expected = None
        if hasattr(model, "n_features_in_"):
            try:
                expected = int(getattr(model, "n_features_in_") )
            except Exception:
                expected = None
        elif hasattr(model, "n_features_"):
            try:
                expected = int(getattr(model, "n_features_") )
            except Exception:
                expected = None

        # If expected is known, pad or trim the features vector
        if expected is not None and expected != len(features):
            if expected > len(features):
                # pad with zeros (neutral/default value)
                features += [0.0] * (expected - len(features))
            else:
                features = features[:expected]

        # Attempt prediction, and if shape mismatch occurs try to adapt
        try:
            pred = model.predict([features])[0]
        except ValueError as e:
            msg = str(e)
            import re
            m = re.search(r"expected[: ]+(\d+)[, ]+got[: ]+(\d+)", msg)
            if m:
                exp = int(m.group(1))
                if exp > len(features):
                    # pad and retry prediction
                    features += [0.0] * (exp - len(features))
                    try:
                        pred = model.predict([features])[0]
                    except Exception as e2:
                        raise HTTPException(status_code=500, detail=f"Prediction error after padding: {str(e2)}")
                else:
                    # shape mismatch but can't resolve by padding
                    raise HTTPException(status_code=500, detail=f"Prediction error: {msg}")
            else:
                # some other ValueError from predict
                raise HTTPException(status_code=500, detail=f"Prediction error: {msg}")
        except Exception as e:
            # Any other errors from predict
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

        # Ensure numeric and clamp
        try:
            val = float(pred)
        except Exception:
            raise HTTPException(status_code=500, detail=f"Model returned non-numeric prediction for {risk}")

        results[risk] = max(0.0, min(100.0, val))

    return RiskScores(**results)
