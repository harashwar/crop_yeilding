# ==============================================================
# export_metadata.py
# Quick standalone script to export crop list and country list
# from the dataset WITHOUT re-running the full training pipeline.
# Run: python export_metadata.py
# ==============================================================

import pandas as pd
import json

FILEPATH = r"C:\Users\HP\Downloads\yealding_data\yield_df.csv"

df = pd.read_csv(FILEPATH)

# Rename to logical names (same as train.py)
df = df.rename(columns={
    "Area":                           "State",
    "Item":                           "Crop",
    "hg/ha_yield":                    "Yield",
    "average_rain_fall_mm_per_year":  "Rainfall",
    "avg_temp":                       "Temperature",
    "Year":                           "Year",
    "pesticides_tonnes":              "Pesticides",
})
df = df.dropna()

crops     = sorted(df["Crop"].dropna().unique().tolist())
countries = sorted(df["State"].dropna().unique().tolist())

metadata = {
    "crops":     crops,
    "countries": countries,
    "records":   int(len(df)),
}

with open("model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"[OK] Exported {len(crops)} crops and {len(countries)} countries -> model_metadata.json")
print(f"     Sample crops: {crops[:5]}")
