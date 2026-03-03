# ==============================================================
# train.py - Crop Yield Prediction ML Pipeline
# Dataset columns:
#   Area      -> Country / State
#   Item      -> Crop name
#   Year      -> Year
#   hg/ha_yield -> Yield (target)
#   average_rain_fall_mm_per_year -> Rainfall
#   pesticides_tonnes -> Pesticides
#   avg_temp  -> Average Temperature
# ==============================================================

import pandas as pd
import numpy as np
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------------
def load_data(filepath):
    print("=" * 60)
    print(" CROP YIELD PREDICTION - ML PIPELINE")
    print("=" * 60)
    print(f"\n[1] Loading data from:\n    {filepath}\n")

    df = pd.read_csv(filepath)

    print(f"    Dataset Shape   : {df.shape}")
    print(f"    Columns         : {df.columns.tolist()}")
    print("\n    First 5 rows:")
    print(df.head().to_string())
    return df


# ------------------------------------------------------------------
# 2. PREPROCESS DATA
# ------------------------------------------------------------------
def preprocess_data(df):
    print("\n[2] Preprocessing Data...")

    # ---- Column mapping from actual CSV to logical names ----
    column_map = {
        "Area":                           "State",
        "Item":                           "Crop",
        "hg/ha_yield":                    "Yield",
        "average_rain_fall_mm_per_year":  "Rainfall",
        "avg_temp":                       "Temperature",
        "Year":                           "Year",
        "pesticides_tonnes":              "Pesticides",
    }

    rename = {k: v for k, v in column_map.items() if k in df.columns}
    df = df.rename(columns=rename)
    print(f"    Renamed columns : {rename}")

    before = df.shape[0]
    df = df.dropna()
    after = df.shape[0]
    print(f"    Rows after dropping NaN: {after} (dropped {before - after})")

    if "Yield" in df.columns:
        df = df[df["Yield"] > 0]

    categorical_features = ["Crop", "State"]
    numerical_features   = ["Rainfall", "Temperature", "Pesticides", "Year"]

    categorical_features = [c for c in categorical_features if c in df.columns]
    numerical_features   = [c for c in numerical_features   if c in df.columns]

    all_features = categorical_features + numerical_features
    target       = "Yield"

    missing = [c for c in all_features + [target] if c not in df.columns]
    if missing:
        raise ValueError(f"Required columns not found after renaming: {missing}")

    print(f"\n    Categorical features : {categorical_features}")
    print(f"    Numerical features   : {numerical_features}")
    print(f"    Target               : {target}")
    print(f"    Final dataset shape  : {df.shape}")

    X = df[all_features]
    y = df[target]

    return X, y, categorical_features, numerical_features, df


# ------------------------------------------------------------------
# 3. EXPORT PER-COUNTRY DEFAULTS (pesticides fallback for API-mode)
# ------------------------------------------------------------------
def export_country_defaults(df, filename="country_defaults.json"):
    """
    Compute per-country averages for Pesticides, Rainfall, Temperature.
    These are used as fallback values when the weather API is used
    and manual inputs are not provided.
    """
    print("\n[3] Exporting per-country defaults...")

    agg = df.groupby("State").agg(
        avg_pesticides=("Pesticides", "median"),
        avg_rainfall=("Rainfall",   "median"),
        avg_temperature=("Temperature", "median"),
    ).round(2)

    defaults = agg.to_dict(orient="index")

    # Also compute global medians as fallback
    global_defaults = {
        "avg_pesticides":  round(float(df["Pesticides"].median()), 2),
        "avg_rainfall":    round(float(df["Rainfall"].median()),   2),
        "avg_temperature": round(float(df["Temperature"].median()), 2),
    }
    defaults["__global__"] = global_defaults

    with open(filename, "w") as f:
        json.dump(defaults, f, indent=2)

    print(f"    Saved {len(defaults) - 1} country entries -> '{filename}'")
    return defaults


# ------------------------------------------------------------------
# 4. BUILD SKLEARN PIPELINE
# ------------------------------------------------------------------
def build_pipeline(model, categorical_features, numerical_features):
    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(),                       numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ])
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor",    model),
    ])


# ------------------------------------------------------------------
# 5. EVALUATE A TRAINED PIPELINE
# ------------------------------------------------------------------
def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    return {"MAE": mae, "MSE": mse, "R2 Score": r2}


# ------------------------------------------------------------------
# 6. TRAIN, TUNE & COMPARE MODELS
# ------------------------------------------------------------------
def train_and_evaluate(X, y, categorical_features, numerical_features):
    print("\n[4] Splitting data  ->  80% train / 20% test")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"    Train rows: {len(X_train)}  |  Test rows: {len(X_test)}")

    results   = {}
    pipelines = {}

    # ---- Linear Regression ----------------------------------------
    print("\n[5] Training Linear Regression...")
    lr_pipeline = build_pipeline(LinearRegression(), categorical_features, numerical_features)
    lr_pipeline.fit(X_train, y_train)
    results["Linear Regression"]   = evaluate(lr_pipeline, X_test, y_test)
    pipelines["Linear Regression"] = lr_pipeline
    print("    Done.")

    cv_lr = cross_val_score(lr_pipeline, X_train, y_train, cv=5, scoring="r2")
    print(f"    CV R2 (5-fold): {cv_lr.mean():.4f} +/- {cv_lr.std():.4f}")

    # ---- Random Forest with GridSearchCV --------------------------
    print("\n[6] Training Random Forest with Hyperparameter Tuning...")
    print("    (This may take a few minutes...)")

    rf_base = build_pipeline(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        categorical_features,
        numerical_features
    )
    param_grid = {
        "regressor__n_estimators":      [50, 100],
        "regressor__max_depth":         [None, 10, 20],
        "regressor__min_samples_split": [2, 5],
    }
    grid_search = GridSearchCV(
        rf_base, param_grid, cv=3, scoring="r2", n_jobs=-1, verbose=0
    )
    grid_search.fit(X_train, y_train)
    best_rf = grid_search.best_estimator_
    print(f"    Best hyperparameters: {grid_search.best_params_}")

    results["Random Forest (Tuned)"]   = evaluate(best_rf, X_test, y_test)
    pipelines["Random Forest (Tuned)"] = best_rf

    cv_rf = cross_val_score(best_rf, X_train, y_train, cv=5, scoring="r2")
    print(f"    CV R2 (5-fold): {cv_rf.mean():.4f} +/- {cv_rf.std():.4f}")

    # ---- Model Comparison Table -----------------------------------
    print("\n[7] Model Comparison Table:")
    comp_df = pd.DataFrame(results).T.round(4)
    print(comp_df.to_string())

    # ---- Pick Best Model ------------------------------------------
    best_name  = max(results, key=lambda k: results[k]["R2 Score"])
    best_model = pipelines[best_name]
    best_r2    = results[best_name]["R2 Score"]
    print(f"\n    >> Best Model: {best_name}  |  R2 Score: {best_r2:.4f}")

    # ---- Feature Importance (Random Forest) -----------------------
    if "Random Forest" in best_name:
        print("\n[8] Feature Importance (Top 10):")
        try:
            rf_reg         = best_model.named_steps["regressor"]
            preprocessor   = best_model.named_steps["preprocessor"]
            cat_encoder    = preprocessor.named_transformers_["cat"]
            cat_names      = cat_encoder.get_feature_names_out(categorical_features).tolist()
            all_feat_names = numerical_features + cat_names
            importances    = rf_reg.feature_importances_
            imp_df = (
                pd.DataFrame({"Feature": all_feat_names, "Importance": importances})
                .sort_values("Importance", ascending=False)
                .reset_index(drop=True)
            )
            print(imp_df.head(10).to_string(index=False))
        except Exception as e:
            print(f"    Could not extract feature importance: {e}")

    return best_model, best_name


# ------------------------------------------------------------------
# 7. SAVE MODEL
# ------------------------------------------------------------------
def save_model(model, filename="crop_yield_model.pkl"):
    joblib.dump(model, filename)
    print(f"\n[9] Model saved -> '{filename}'")


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def main():
    filepath = r"C:\Users\HP\Downloads\yealding_data\yield_df.csv"
    try:
        df = load_data(filepath)
        X, y, cat_feats, num_feats, df_clean = preprocess_data(df)
        export_country_defaults(df_clean)
        best_model, best_name = train_and_evaluate(X, y, cat_feats, num_feats)
        save_model(best_model, "crop_yield_model.pkl")
        print("\n" + "=" * 60)
        print(" PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 60)
    except Exception as e:
        print(f"\nERROR: {e}")
        raise


if __name__ == "__main__":
    main()
