
import pandas as pd
import joblib
import json
import numpy as np
from difflib import get_close_matches
import os

# Ścieżki
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
FALLBACK_PATH = os.path.join(BASE_DIR, "..", "fallbacks", "fallback_home_away_total.json")
RESULTS_PATH = os.path.join(BASE_DIR, "..", "results5months.csv")

# Wczytaj model i kolumny
model = joblib.load(os.path.join(MODELS_DIR, "xgb_expected_corners_optuna.pkl"))

with open(os.path.join(MODELS_DIR, "xgb_expected_corners_columns.json")) as f:
    columns = json.load(f)

with open(FALLBACK_PATH, "r", encoding="utf-8") as f:
    fallback = json.load(f)

df = pd.read_csv(RESULTS_PATH)

def normalize_team_name(name, all_teams):
    match = get_close_matches(name, all_teams, n=1, cutoff=0.85)
    return match[0] if match else name

all_teams = list(fallback.keys())

def get_features(team, opponent, is_home):
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
        "is_home": int(is_home),
        "team_strength": crosses + shots - opp_stats.get("total_blocked_shots", 0) - opp_stats.get("total_gk_saves", 0)
    }

# Predykcje
results = []
for _, row in df.iterrows():
    home = normalize_team_name(row["home_team"], all_teams)
    away = normalize_team_name(row["away_team"], all_teams)
    total_corners = row["total_corners"]

    if home not in fallback or away not in fallback:
        continue

    try:
        home_features = get_features(home, away, True)
        away_features = get_features(away, home, False)

        home_df = pd.DataFrame([home_features])[columns]
        away_df = pd.DataFrame([away_features])[columns]

        home_pred = np.expm1(model.predict(home_df)[0])
        away_pred = np.expm1(model.predict(away_df)[0])
        expected = round(home_pred + away_pred, 2)

        expected_rounded = round(expected)
        total_corners_rounded = round(total_corners)

        error = abs(expected_rounded - total_corners_rounded)
        extreme_miss = error >= 5

        results.append({
            "home_team": home,
            "away_team": away,
            "expected_corners": expected,
            "total_corners": float(total_corners),
            "expected_rounded": expected_rounded,
            "total_rounded": total_corners_rounded,
            "hit": expected_rounded <= total_corners_rounded,
            "abs_error": error,
            "extreme_miss": extreme_miss
        })
    except:
        continue

df_results = pd.DataFrame(results)
df_results.to_csv("eval_results_optuna_model_final.csv", index=False)

# Raport ogólny
total = len(df_results)
hits = df_results["hit"].sum()
accuracy = (hits / total) * 100 if total > 0 else 0.0
mean_abs_error = df_results["abs_error"].mean()
extreme_miss_count = df_results["extreme_miss"].sum()

# Raport po drużynach (jako home + away)
team_stats = {}
for _, row in df_results.iterrows():
    for team in [row["home_team"], row["away_team"]]:
        if team not in team_stats:
            team_stats[team] = {"total": 0, "hits": 0}
        team_stats[team]["total"] += 1
        if row["hit"]:
            team_stats[team]["hits"] += 1

team_report = [
    {"team": k, "accuracy": (v["hits"] / v["total"]) * 100 if v["total"] else 0, "games": v["total"]}
    for k, v in team_stats.items() if v["total"] >= 3
]
df_team_report = pd.DataFrame(team_report).sort_values("accuracy", ascending=False)

best_team = df_team_report.iloc[0]
worst_team = df_team_report.iloc[-1]

# Wyniki
print(f"✅ Skuteczność: {accuracy:.2f}% ({hits}/{total})")
print(f"📉 Średni błąd bezwzględny: {mean_abs_error:.2f}")
print(f"❌ Ekstremalnych przestrzeleń (>=5): {extreme_miss_count}")
print(f"🏆 Najlepsza skuteczność: {best_team['team']} ({best_team['accuracy']:.2f}%, {best_team['games']} meczów)")
print(f"🫣 Najgorsza skuteczność: {worst_team['team']} ({worst_team['accuracy']:.2f}%, {worst_team['games']} meczów)")

print("\n🔍 Przykładowe predykcje:")
print(df_results.head(10))
