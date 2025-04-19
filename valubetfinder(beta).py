
import os
import json
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from difflib import get_close_matches
import requests
import streamlit as st
from scipy.stats import norm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
FALLBACK_PATH = os.path.join(BASE_DIR, "fallbacks", "fallback_home_away_total.json")
TEAM_MAP_PATH = os.path.join(BASE_DIR, "fallbacks", "team_name_map.json")
MODEL_PATH = os.path.join(MODELS_DIR, "xgb_expected_corners_optuna.pkl")

ODDS_API_KEY = "b545b6af09f3618f7343de11c1fb6f23"
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1360722279922925609/4y2zO7rDfPFr-tvViV8AYEXyBeoGkaTwU3xeaMgtlAJHOHmGoBARErTeo26xZlMjeoG1"
ODDS_URL = "https://api.the-odds-api.com/v4/sports/soccer/odds"


def normalize_team_name(name, fallback_teams, team_map):
    mapped = team_map.get(name, name)
    match = get_close_matches(mapped, fallback_teams, n=1, cutoff=0.85)
    return match[0] if match else None

def get_features(team, opponent, fallback):
    stats = fallback.get(team, {})
    opp_stats = fallback.get(opponent, {})

    crosses = stats.get("total_crosses", 0)
    shots = stats.get("total_shots", 0)
    pressing = stats.get("total_pressing", 1)

    return {
        "crosses": crosses,
        "shots": shots,
        "possession": stats.get("total_possession", 0),
        "pressing": pressing,
        "blocked_shots": opp_stats.get("total_blocked_shots", 0),
        "gk_saves": opp_stats.get("total_gk_saves", 0),
        "crosses_per_shot": crosses / shots if shots else 0,
        "crosses_per_pressing": crosses / pressing if pressing else 0,
        "interaction_crosses_pressing": crosses * pressing,
        "tempo": pressing + opp_stats.get("total_pressing", 0),
        "avg_corners_conceded": opp_stats.get("conceded_corners_avg", 0),
    }

def calculate_probability(expected, line):
    std_dev = 1.8
    return 1 - norm.cdf(line, loc=expected, scale=std_dev)

def send_discord_notification(value_bet):
    content = (
        f"🎯 **Value Bet**:\n"
        f"📊 {value_bet['match']}\n"
        f"📈 Linia: {value_bet['line']} @ {value_bet['odds']} ({value_bet['bookmaker']})\n"
        f"📌 Expected: {value_bet['expected']} | Value: {value_bet['value']}"
    )
    data = {"content": content}
    try:
        response = requests.post(DISCORD_WEBHOOK_URL, json=data)
        if response.status_code == 204:
            print(f"✅ Wysłano powiadomienie Discord dla {value_bet['match']}")
    except Exception as e:
        print("❌ Błąd wysyłania na Discord:", e)

def prepare_features(df: pd.DataFrame, is_home_flag: int) -> pd.DataFrame:
    """Uzupełnia brakujące cechy wymagane przez model."""
    df["is_home"] = is_home_flag
    df["team_strength"] = 1  # Można zaktualizować na podstawie rankingu lub miejsca w tabeli
    return df
def detect_value_bets(model, fallback, team_map):
    today = datetime.today().strftime("%Y-%m-%d")
    params = {
        "regions": "eu",
        "markets": "totals",
        "dateFormat": "iso",
        "oddsFormat": "decimal",
        "apiKey": ODDS_API_KEY
    }

    response = requests.get(ODDS_URL, params=params)
    if response.status_code != 200:
        st.error("❌ Błąd w API OddsAPI")
        return []

    try:
        matches = response.json()
    except Exception as e:
        st.error(f"❌ Błąd dekodowania JSON: {e}")
        return []

    if not isinstance(matches, list):
        st.warning("⚠️ Odpowiedź z OddsAPI nie zawiera listy meczów.")
        st.json(matches)
        return []
    fallback_teams = list(fallback.keys())
    value_bets = []

    for match in matches:
        home = normalize_team_name(match['home_team'], fallback_teams, team_map)
        away = normalize_team_name(match['away_team'], fallback_teams, team_map)
        if not home or not away:
            continue

        home_features = get_features(home, away, fallback)
        away_features = get_features(away, home, fallback)

        home_df = pd.DataFrame([home_features])
        home_df = prepare_features(home_df, is_home_flag=1)
        away_df = pd.DataFrame([away_features])
        away_df = prepare_features(away_df, is_home_flag=0)

        home_pred = model.predict(home_df)[0]
        away_pred = model.predict(away_df)[0]
        expected = round(home_pred + away_pred, 2)
        st.markdown(f"### 🔍 {home} vs {away}")
        st.write("Expected corners:", expected)


        for bookmaker in match['bookmakers']:
            for market in bookmaker['markets']:
                if market['key'] != 'totals':
                    continue
                for outcome in market['outcomes']:
                    point = float(outcome['point'])
                    odds = outcome['price']
                    direction = outcome['name']

                    prob = calculate_probability(expected, point)
                    if direction == "Under":
                        prob = 1 - prob

                    value = prob * odds - 1
                    
                    st.write(f"📋 Kierunek: {direction}, Linia: {point}, Kurs: {odds}")
                    st.write(f"📊 Prawdopodobieństwo: {prob:.2f}, Value: {value:.3f}")
                    if value > 0.1:
                        vb = {
                            "match": f"{home} vs {away}",
                            "expected": expected,
                            "line": f"{direction} {point}",
                            "odds": odds,
                            "value": round(value, 3),
                            "bookmaker": bookmaker['title']
                        }
                        value_bets.append(vb)
                        send_discord_notification(vb)

    return value_bets

# Streamlit GUI
st.set_page_config(page_title="📈 Value Bet Detector", layout="wide")
st.title("💰 Value Bet Detector (Corners)")

model = joblib.load(MODEL_PATH)
with open(FALLBACK_PATH, "r", encoding="utf-8") as f:
    fallback = json.load(f)
with open(TEAM_MAP_PATH, "r", encoding="utf-8") as f:
    team_map = json.load(f)

with st.spinner("🔍 Wyszukiwanie value betów..."):
    value_bets = detect_value_bets(model, fallback, team_map)

if value_bets:
    df = pd.DataFrame(value_bets)
    df = df.sort_values("value", ascending=False)
    st.success(f"✅ Znaleziono {len(df)} value betów!")
    st.dataframe(df, use_container_width=True)
else:
    st.warning("⚠️ Nie znaleziono value betów. Spróbuj ponownie później.")