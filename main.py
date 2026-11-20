# -*- coding: utf-8 -*-
"""
main.py  -  API for the Telangana crop yield model

Needs:   yield_model.pkl (made by train_model.py) in the same folder
Install: pip install fastapi uvicorn scikit-learn pandas numpy joblib
Run:     uvicorn main:app --reload
Docs:    http://127.0.0.1:8000/docs   (try the endpoints in the browser)

Endpoints
  GET  /            quick check that the API is alive
  GET  /options     districts, crops, seasons, rain types (for dropdowns)
  POST /predict     predicted yield + likely range + uncertainty level
  GET  /insights    model comparison, input importance, accuracy by crop/district, weather slopes
"""

from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------
# Load the saved model once, when the server starts
# ---------------------------------------------------------------
bundle = joblib.load("yield_model.pkl")
model = bundle["model"]
valid = bundle["valid"]                     # {district: {crop: [seasons]}}
last_area = bundle["last_area"]             # {(district, crop, season): area in hectares}
crop_median_area = bundle["crop_median_area"]
ranges = bundle["ranges"]                   # {crop: {"low", "high", "level"}}
rain_types = bundle["rain_types"]

app = FastAPI(title="Telangana Crop Yield Predictor",
              description="College project: Random Forest model trained on 1997-2019 data.")

app.add_middleware(CORSMiddleware, allow_origins=["*"],     # lets a website call this API
                   allow_methods=["*"], allow_headers=["*"])


# ---------------------------------------------------------------
# What the farmer / frontend sends
# ---------------------------------------------------------------
class PredictRequest(BaseModel):
    district: str
    crop: str
    season: str
    rain_type: str                                   # dry / normal / wet
    area: Optional[float] = Field(None, gt=0,        # optional: your own area in hectares
                                  description="Leave empty to use the last known area for this district, crop and season")


def match(value, options, name):
    """Accept any upper/lower case and extra spaces, return the exact spelling the model knows."""
    lookup = {str(o).strip().lower(): o for o in options}
    key = str(value).strip().lower()
    if key not in lookup:
        raise HTTPException(status_code=400,
                            detail=f"Unknown {name} '{value}'. Choose one of: {sorted(map(str, options))}")
    return lookup[key]


# ---------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------
@app.get("/")
def home():
    return {"status": "ok", "model": bundle["model_name"], "trained_years": bundle["trained_years"]}


@app.get("/options")
def options():
    """Everything needed to fill the dropdowns. valid[district][crop] = seasons that exist in the data."""
    return {"districts": sorted(valid), "rain_types": rain_types, "valid": valid}


@app.post("/predict")
def predict(req: PredictRequest):
    district = match(req.district, valid.keys(), "district")
    crop = match(req.crop, valid[district].keys(), f"crop for {district}")
    season = match(req.season, valid[district][crop], f"season for {crop} in {district}")
    rain = match(req.rain_type, rain_types, "rain_type")

    # Area: use the farmer's number, else the last known area, else the crop's median area
    if req.area is not None:
        area, area_source = req.area, "provided by you"
    elif (district, crop, season) in last_area:
        area, area_source = last_area[(district, crop, season)], "last known area for this district, crop and season"
    else:
        area, area_source = crop_median_area[crop], "median area for this crop"

    row = pd.DataFrame([{"District": district, "Crop": crop, "Season": season,
                         "Rain_Type": rain, "log_area": np.log1p(area)}])
    predicted = float(np.expm1(model.predict(row)[0]))      # model predicts log yield -> back to t/ha

    r = ranges[crop]
    low, high = predicted * r["low"], predicted * r["high"]
    low, high = min(low, predicted), max(high, predicted)   # keep the estimate inside its own range

    return {
        "district": district, "crop": crop, "season": season, "rain_type": rain,
        "area_hectares": round(area, 1), "area_source": area_source,
        "predicted_yield_t_per_ha": round(predicted, 3),
        "likely_range_t_per_ha": [round(low, 3), round(high, 3)],
        "uncertainty": r["level"],                           # Low / Medium / High
        "estimated_production_tonnes": round(predicted * area, 1),
        "note": "Likely range covers about 7 in 10 to 8 in 10 past cases. This is a student project estimate.",
    }


@app.get("/insights")
def insights():
    """Numbers for the report page: how good the model is and where it is unsure."""
    return {
        "model": bundle["model_name"],
        "model_comparison": bundle["model_comparison"],
        "input_importance_pct": bundle["input_importance"],
        "accuracy_by_crop": bundle["accuracy_by_crop"],
        "accuracy_by_district": bundle["accuracy_by_district"],
        "weather_slopes": bundle["weather_slopes"],
    }
