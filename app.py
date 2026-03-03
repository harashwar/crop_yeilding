# ==============================================================
# app.py  -  Crop Yield Prediction API  (v3 - OWM Edition)
#
# Weather sources:
#   ① OpenWeatherMap (OWM)  - current temp, humidity  [requires OWM_API_KEY]
#   ② Open-Meteo Archive    - annual rainfall + temp  [FREE, no key needed]
#
# If OWM_API_KEY is not set, falls back to Open-Meteo only.
# Country/state for the ML model is resolved silently via Nominatim.
#
# Usage:
#   1. Copy .env.example → .env  and add your OWM key  (optional)
#   2. python train.py && python export_metadata.py
#   3. python app.py  →  http://localhost:5000
# ==============================================================

import os
from typing import Optional
import json
import time
import datetime
import requests
import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

load_dotenv()   # loads .env if it exists

app = Flask(__name__)

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
OWM_API_KEY   = os.getenv("OWM_API_KEY", "").strip()
ACRES_TO_HA   = 0.4047        # 1 acre = 0.4047 ha
HG_TO_TONNES  = 0.0001        # 1 hg/ha  = 0.0001 t/ha
CACHE_TTL     = 6 * 3600      # 6-hour cache

SOIL_MULTIPLIERS = {
    "black":  1.15,   # Vertisol  — very fertile
    "loamy":  1.10,   # Loam      — best all-rounder
    "clay":   0.95,   # Clay      — fertile but poor drainage
    "red":    0.90,   # Laterite  — moderate fertility
    "sandy":  0.80,   # Sandy     — poor retention
}

_cache: dict = {}

# ------------------------------------------------------------------
# Load artefacts
# ------------------------------------------------------------------
for path in ["crop_yield_model.pkl"]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"'{path}' not found. Run 'python train.py' first.")

model             = joblib.load("crop_yield_model.pkl")
country_defaults  = json.load(open("country_defaults.json"))  if os.path.exists("country_defaults.json")  else {}
model_metadata    = json.load(open("model_metadata.json"))    if os.path.exists("model_metadata.json")    else {}

print(f"[OK] Model loaded | {len(model_metadata.get('crops',[]))} crops | OWM key: {'SET' if OWM_API_KEY else 'NOT SET (Open-Meteo fallback)'}")


# ------------------------------------------------------------------
# Cache helpers
# ------------------------------------------------------------------
def _cget(key):
    e = _cache.get(key)
    return e[1] if e and (time.time() - e[0]) < CACHE_TTL else None

def _cset(key, val):
    _cache[key] = (time.time(), val)


# ------------------------------------------------------------------
# ① OpenWeatherMap  →  current temp + humidity + recent rain
# ------------------------------------------------------------------
def owm_current(lat: float, lon: float) -> Optional[dict]:
    """Returns current weather from OWM, or None if key missing / invalid."""
    if not OWM_API_KEY:
        return None

    ck = f"owm:{lat:.2f}:{lon:.2f}"
    if (c := _cget(ck)):
        return c

    try:
        r = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"lat": lat, "lon": lon, "appid": OWM_API_KEY, "units": "metric"},
            timeout=10,
        )
        if r.status_code == 401:
            print("[!] OWM API key is invalid — falling back to Open-Meteo")
            return None
        r.raise_for_status()
        d = r.json()

        result = {
            "temperature": round(d["main"]["temp"], 2),
            "humidity":    d["main"]["humidity"],
            "rain_1h":     round(d.get("rain", {}).get("1h", 0.0), 2),
            "city":        d.get("name", ""),
            "country_code": d.get("sys", {}).get("country", ""),
        }
        _cset(ck, result)
        return result

    except Exception as e:
        print(f"[!] OWM fetch failed: {e}")
        return None


# ------------------------------------------------------------------
# ② Open-Meteo Archive  →  annual avg temp + total rainfall
# ------------------------------------------------------------------
def openmeteo_annual(lat: float, lon: float) -> dict:
    ck = f"om:{lat:.2f}:{lon:.2f}"
    if (c := _cget(ck)):
        return c

    data_year  = datetime.datetime.now().year - 1
    r = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude":   lat,
            "longitude":  lon,
            "start_date": f"{data_year}-01-01",
            "end_date":   f"{data_year}-12-31",
            "daily":      "temperature_2m_mean,precipitation_sum",
            "timezone":   "auto",
        },
        timeout=15,
    )
    r.raise_for_status()
    daily  = r.json().get("daily", {})
    temps  = [t for t in daily.get("temperature_2m_mean", []) if t is not None]
    precip = [p for p in daily.get("precipitation_sum",   []) if p is not None]
    if not temps:
        raise ValueError("No temperature data returned by Open-Meteo.")

    result = {
        "avg_temp":     round(sum(temps) / len(temps), 2),
        "annual_rain":  round(sum(precip), 2),
        "data_year":    data_year,
    }
    _cset(ck, result)
    return result


# ------------------------------------------------------------------
# ③ Nominatim  →  country name for ML model  (silent, user never sees it)
# ------------------------------------------------------------------
def get_country_silent(lat: float, lon: float) -> str:
    ck = f"country:{lat:.2f}:{lon:.2f}"
    if (c := _cget(ck)):
        return c

    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"lat": lat, "lon": lon, "format": "json", "zoom": 3},
            headers={"User-Agent": "CropYieldPredictor/3.0"},
            timeout=10,
        )
        r.raise_for_status()
        nominatim_country = r.json().get("address", {}).get("country", "")
        country = _match_dataset_country(nominatim_country)
        _cset(ck, country)
        return country
    except Exception:
        return "__global__"   # fall back to global median pesticides


def _match_dataset_country(name: str) -> str:
    """Fuzzy-match Nominatim country name to our dataset's country list."""
    known = model_metadata.get("countries", [])
    if name in known:
        return name
    lower = {c.lower(): c for c in known}
    if name.lower() in lower:
        return lower[name.lower()]
    for k in known:
        if name.lower() in k.lower() or k.lower() in name.lower():
            return k
    return name


# ------------------------------------------------------------------
# Combined weather fetch: OWM current  +  Open-Meteo annual
# ------------------------------------------------------------------
def get_full_weather(lat: float, lon: float) -> dict:
    ck = f"fullweather:{lat:.2f}:{lon:.2f}"
    if (c := _cget(ck)):
        return c

    annual = openmeteo_annual(lat, lon)
    owm    = owm_current(lat, lon)

    if owm:
        result = {
            "display_temp":   owm["temperature"],      # live — shown in UI
            "humidity":       owm["humidity"],          # live — shown in UI
            "rain_now_mm":    owm["rain_1h"],           # live — shown in UI
            "model_temp":     annual["avg_temp"],       # annual — fed to ML model
            "model_rainfall": annual["annual_rain"],    # annual — fed to ML model
            "data_year":      annual["data_year"],
            "source":         "OpenWeatherMap + Open-Meteo",
            "city":           owm["city"],
        }
    else:
        result = {
            "display_temp":   annual["avg_temp"],
            "humidity":       None,
            "rain_now_mm":    None,
            "model_temp":     annual["avg_temp"],
            "model_rainfall": annual["annual_rain"],
            "data_year":      annual["data_year"],
            "source":         "Open-Meteo",
            "city":           "",
        }

    _cset(ck, result)
    return result


# ══════════════════════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════════════════════

# ------------------------------------------------------------------
# GET /crops  →  crop dropdown list
# ------------------------------------------------------------------
@app.route("/crops")
def get_crops():
    return jsonify({"status": "success", "crops": model_metadata.get("crops", [])})


# ------------------------------------------------------------------
# GET /get-weather?lat=X&lon=Y
# Called by the frontend after GPS resolves to show weather chips
# ------------------------------------------------------------------
@app.route("/get-weather")
def get_weather():
    try:
        lat = float(request.args["lat"])
        lon = float(request.args["lon"])
    except (KeyError, ValueError):
        return jsonify({"status": "error", "message": "lat and lon are required."}), 400

    try:
        w = get_full_weather(lat, lon)
        return jsonify({
            "status":       "success",
            "temperature":  w["display_temp"],
            "humidity":     w["humidity"],
            "rain_now_mm":  w["rain_now_mm"],
            "annual_rain":  w["model_rainfall"],
            "data_year":    w["data_year"],
            "source":       w["source"],
            "city":         w["city"],
        })
    except requests.exceptions.ConnectionError:
        return jsonify({"status": "error", "message": "Cannot reach weather API. Check internet."}), 503
    except requests.exceptions.Timeout:
        return jsonify({"status": "error", "message": "Weather API timed out. Try again."}), 504
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ------------------------------------------------------------------
# POST /predict
# Body: { lat, lon, Crop, Soil_Type, Land_acres }
# Country & weather resolved silently from lat/lon
# ------------------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        crop       = str(data.get("Crop", "")).strip()
        soil_type  = str(data.get("Soil_Type", "loamy")).strip().lower()
        land_acres = float(data.get("Land_acres", 1.0))
        lat        = float(data["lat"])
        lon        = float(data["lon"])

        if not crop:
            return jsonify({"status": "error", "message": "Crop is required."}), 400
        if land_acres <= 0:
            return jsonify({"status": "error", "message": "Land size must be > 0."}), 400

        # ── Resolve weather & country (silently) ──────────────────
        w       = get_full_weather(lat, lon)
        country = get_country_silent(lat, lon)

        country_info = country_defaults.get(country, country_defaults.get("__global__", {}))
        pesticides   = float(country_info.get("avg_pesticides", 50.0))
        year         = datetime.datetime.now().year

        # ── Build model input ──────────────────────────────────────
        input_df = pd.DataFrame([{
            "Crop":        crop,
            "State":       country,
            "Year":        year,
            "Rainfall":    w["model_rainfall"],
            "Temperature": w["model_temp"],
            "Pesticides":  pesticides,
        }])

        # ── Predict (hg/ha) ────────────────────────────────────────
        raw_hg_per_ha = float(model.predict(input_df)[0])

        # ── Soil correction ────────────────────────────────────────
        soil_mult    = SOIL_MULTIPLIERS.get(soil_type, 1.0)
        adj_hg_per_ha = raw_hg_per_ha * soil_mult

        # ── Convert to tonnes + scale by land area ─────────────────
        yield_t_per_ha  = adj_hg_per_ha * HG_TO_TONNES
        land_ha         = land_acres * ACRES_TO_HA
        total_tonnes    = yield_t_per_ha * land_ha

        return jsonify({
            "status":                  "success",
            "total_production_tonnes": round(total_tonnes, 3),
            "yield_per_ha_tonnes":     round(yield_t_per_ha, 4),
            "land_ha":                 round(land_ha, 2),
            "land_acres":              land_acres,
            "soil_multiplier":         soil_mult,
            "weather_used": {
                "temperature": w["model_temp"],
                "rainfall":    w["model_rainfall"],
                "humidity":    w["humidity"],
                "source":      w["source"],
            },
        })

    except KeyError as e:
        return jsonify({"status": "error", "message": f"Missing field: {e}"}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ------------------------------------------------------------------
# GET /  →  UI
# ------------------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


# ------------------------------------------------------------------
# GET /health
# ------------------------------------------------------------------
@app.route("/health")
def health():
    return jsonify({
        "status":      "ok",
        "owm_key_set": bool(OWM_API_KEY),
        "crops":       len(model_metadata.get("crops", [])),
        "cache_items": len(_cache),
    })


# ------------------------------------------------------------------
if __name__ == "__main__":
    print("\nCrop Yield Prediction API  v3  (OWM Edition)")
    print(f"  OWM Key : {'SET [OK]' if OWM_API_KEY else 'not set (using Open-Meteo fallback)'}")
    print( "  URL     : http://localhost:5000\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
