# Poprawiony skrypt poniżej — Optuna z większym zakresem i więcej prób

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
import datetime
import requests
import optuna
from sklearn.model_selection import cross_val_score, KFold
from xgboost import XGBRegressor
from difflib import get_close_matches
import matplotlib.pyplot as plt

MODEL_PATH = "models/xgb_expected_corners_optuna.pkl"
FEATURES_PATH = "models/xgb_expected_corners_columns.json"
FALLBACK_PATH = "fallbacks/fallback_home_away_total.json"
PARAMS_PATH = "models/xgb_expected_corners_params.json"
os.makedirs("models", exist_ok=True)
os.makedirs("fallbacks", exist_ok=True)

def load_fallback():
    with open(FALLBACK_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def prepare_training_data(fallback):
    rows = []
    for team, stats in fallback.items():
        crosses = stats.get("total_crosses")
        shots = stats.get("total_shots")
        possession = stats.get("total_possession")
        pressing = stats.get("total_pressing")
        blocked = stats.get("total_blocked_shots")
        saves = stats.get("total_gk_saves")
        conceded = stats.get("conceded_corners_avg")
        corners = stats.get("total_corners")
        if None in [crosses, shots, possession, pressing, blocked, saves, conceded, corners]:
            continue
        row = {
            "crosses": crosses,
            "shots": shots,
            "possession": possession,
            "pressing": pressing,
            "blocked_shots": blocked,
            "gk_saves": saves,
            "crosses_per_shot": crosses / shots if shots else 0,
            "crosses_per_pressing": crosses / pressing if pressing else 0,
            "interaction_crosses_pressing": crosses * pressing,
            "tempo": pressing + pressing,
            "avg_corners_conceded": conceded,
            "is_home": 1,
            "team_strength": crosses + shots - blocked - saves
        }
        rows.append((list(row.values()), corners))
    X, y = zip(*rows)
    feature_names = list(row.keys())
    with open(FEATURES_PATH, "w") as f:
        json.dump(feature_names, f)
    return pd.DataFrame(X, columns=feature_names), pd.Series(np.log1p(y))

def train_model():
    def objective(trial):
        fallback = load_fallback()
        X, y = prepare_training_data(fallback)
        params = {
            "max_depth": trial.suggest_int("max_depth", 4, 16),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.2),
            "n_estimators": trial.suggest_int("n_estimators", 300, 1200),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 1.5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.5),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 3.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
            "random_state": 42
        }

        model = XGBRegressor(**params)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=kf, scoring="neg_root_mean_squared_error")
        return -scores.mean()

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100)

    fallback = load_fallback()
    X, y = prepare_training_data(fallback)
    best_params = study.best_params
    with open(PARAMS_PATH, "w") as f:
        json.dump(best_params, f, indent=4)
    model = XGBRegressor(**best_params)
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    return model

def predict_expected_corners(home_team, away_team, fallback, model):
    with open(FEATURES_PATH, "r") as f:
        feature_names = json.load(f)

    def get_features(role_team, opp_team, is_home):
        stats = fallback.get(role_team, {})
        opp_stats = fallback.get(opp_team, {})
        crosses = stats.get("total_crosses", 0)
        shots = stats.get("total_shots", 0)
        pressing = stats.get("total_pressing", 1)
        return [
            crosses,
            shots,
            stats.get("total_possession", 0),
            pressing,
            opp_stats.get("total_blocked_shots", 0),
            opp_stats.get("total_gk_saves", 0),
            crosses / shots if shots else 0,
            crosses / pressing if pressing else 0,
            crosses * pressing,
            pressing + opp_stats.get("total_pressing", 0),
            opp_stats.get("conceded_corners_avg", 0),
            int(is_home),
            crosses + shots - opp_stats.get("total_blocked_shots", 0) - opp_stats.get("total_gk_saves", 0)
        ]

    home_features = get_features(home_team, away_team, True)
    away_features = get_features(away_team, home_team, False)

    home_df = pd.DataFrame([home_features], columns=feature_names)
    away_df = pd.DataFrame([away_features], columns=feature_names)

    home_pred = np.expm1(model.predict(home_df)[0])
    away_pred = np.expm1(model.predict(away_df)[0])

    return round(home_pred + away_pred, 2)


# --- API-Football + fixtures + mapowanie drużyn ---
API_KEY = "ce0f6bfa9b5bae64ec7c59deda487d2d"
API_URL = "https://v3.football.api-sports.io"
TOP5_LEAGUES = [39, 140, 78, 61, 135]
TEAM_MAP_PATH = "fallbacks/team_name_map.json"

def load_team_map():
    with open(TEAM_MAP_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize_team_name(name, fallback_keys, team_map):
    name = team_map.get(name, name)
    match = get_close_matches(name, fallback_keys, n=1, cutoff=0.85)
    return match[0] if match else None

# === CACHED FIXTURE LOGIC ===

import os
import json
import datetime
import requests
import streamlit as st

API_KEY = "ce0f6bfa9b5bae64ec7c59deda487d2d"
API_URL = "https://v3.football.api-sports.io"
TOP5_LEAGUES = [39, 140, 78, 61, 135]
CACHE_DIR = "data"

def load_cached_fixtures(date_str, league_id):
    path = os.path.join(CACHE_DIR, f"fixtures_{date_str}_league_{league_id}.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def save_cached_fixtures(date_str, league_id, data):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"fixtures_{date_str}_league_{league_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_fixtures_with_cache(date):
    headers = {"x-apisports-key": API_KEY}
    fixtures = []
    date_str = date.strftime("%Y-%m-%d")

    for league_id in TOP5_LEAGUES:
        cached = load_cached_fixtures(date_str, league_id)
        if cached:
            fixtures += cached
        else:
            params = {
                "league": league_id,
                "season": 2024,
                "date": date_str
            }
            try:
                r = requests.get(f"{API_URL}/fixtures", headers=headers, params=params)
                if r.status_code == 200:
                    data = r.json().get("response", [])
                    save_cached_fixtures(date_str, league_id, data)
                    fixtures += data
            except Exception as e:
                st.error(f"❌ API call failed: {e}")

    return fixtures




def get_fixtures(selected_date):
    return get_fixtures_with_cache(selected_date)
  
# STREAMLIT UI
st.title("⚽ Expected Corners Predictor (MAX Optuna)")
if st.button("🔁 Przetrenuj model z rozszerzoną Optuną"):
    with st.spinner("Tuning i trening modelu..."):
        model = train_model()
    st.success("Model przetrenowany i zapisany!")

def load_team_map():
    path = os.path.join(BASE_DIR, "..", "fallbacks", "team_name_map.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def normalize_team_name(name, fallback_teams, team_map):
    name = team_map.get(name, name)
    return name if name in fallback_teams else None

# === Przewidywanie dzisiejszych i jutrzejszych meczów ===


st.title("⚽ Expected Corners Predictor (MAX Optuna)")

selected_date = st.date_input("📅 Wybierz datę meczu", datetime.date.today())
st.markdown(f"### 🔎 Wybrane mecze na {selected_date}")


fallback = load_fallback()
team_map = load_team_map()
model = joblib.load(MODEL_PATH)
fixtures = get_fixtures(selected_date)
fallback_teams = list(fallback.keys())

predictions = []

print("📦 Łączna liczba meczów:", len(fixtures))
for match in fixtures:
    print("📋", match["teams"]["home"]["name"], "vs", match["teams"]["away"]["name"])
    home_raw = match["teams"]["home"]["name"]
    away_raw = match["teams"]["away"]["name"]

    home_team = normalize_team_name(home_raw, fallback_teams, team_map)
    away_team = normalize_team_name(away_raw, fallback_teams, team_map)

    if not home_team or not away_team:
        continue

    expected = predict_expected_corners(home_team, away_team, fallback, model)
    predictions.append({
        "Mecz": f"{home_raw} vs {away_raw}",
        "Expected Corners": expected
    })

if predictions:
    df_preds = pd.DataFrame(predictions).sort_values("Expected Corners", ascending=False)
    st.dataframe(df_preds)
else:
    st.warning("Brak dopasowanych meczów do fallbacka.")