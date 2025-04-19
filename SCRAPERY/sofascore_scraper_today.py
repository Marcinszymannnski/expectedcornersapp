import requests
import pandas as pd
from datetime import datetime, timedelta

OUTPUT_FILE = "sofascore_top5_month.csv"
TOP5_LEAGUE_IDS = {8, 17, 23, 34, 35}  # Premier League, Bundesliga, Serie A, Ligue 1, LaLiga
HEADERS = {"User-Agent": "Mozilla/5.0"}

def get_fixtures_for_date(date_str):
    url = f"https://api.sofascore.com/api/v1/sport/football/scheduled-events/{date_str}"
    try:
        resp = requests.get(url, headers=HEADERS)
        if resp.status_code != 200:
            return []
        events = resp.json().get("events", [])
        return [
            {
                "id": e["id"],
                "league_id": e["tournament"]["uniqueTournament"]["id"],
                "home_team": e["homeTeam"]["name"],
                "away_team": e["awayTeam"]["name"]
            }
            for e in events
            if e["tournament"]["uniqueTournament"]["id"] in TOP5_LEAGUE_IDS
        ]
    except:
        return []

def get_match_stats(match_id, home_team, away_team):
    url = f"https://api.sofascore.com/api/v1/event/{match_id}/statistics"
    resp = requests.get(url, headers=HEADERS)
    if resp.status_code != 200:
        return None

    data = resp.json()
    stats = {
        "match_id": match_id,
        "home_team": home_team,
        "away_team": away_team,
        "home_corner_kicks": None,
        "away_corner_kicks": None,
        "home_crosses": None,
        "away_crosses": None,
        "home_possession": None,
        "away_possession": None,
        "home_pressing": None,
        "away_pressing": None,
        "goalkeeper_saves_home": None,
        "goalkeeper_saves_away": None,
        "home_blocked_shots": None,
        "away_blocked_shots": None,
        "home_total_shots": None,
        "away_total_shots": None,
    }

    for section in data.get("statistics", []):
        if section.get("period") != "ALL":
            continue  # tylko dane z całego meczu!

        for group in section.get("groups", []):
            for item in group.get("statisticsItems", []):
                name = item.get("name", "").lower().strip()
                home = item.get("home")
                away = item.get("away")

                if name == "corner kicks":
                    stats["home_corner_kicks"] = home
                    stats["away_corner_kicks"] = away
                elif name == "crosses":
                    stats["home_crosses"] = home
                    stats["away_crosses"] = away
                elif name == "ball possession":
                    stats["home_possession"] = home
                    stats["away_possession"] = away
                elif name == "total shots":
                    stats["home_total_shots"] = home
                    stats["away_total_shots"] = away
                elif name == "blocked shots":
                    stats["home_blocked_shots"] = home
                    stats["away_blocked_shots"] = away
                elif name == "goalkeeper saves":
                    stats["goalkeeper_saves_home"] = home
                    stats["goalkeeper_saves_away"] = away
                elif name in ["accurate passes", "pass success rate"]:
                    stats["home_pressing"] = home
                    stats["away_pressing"] = away

    return stats


def run_monthly_scraper():
    all_data = []
    for delta in range(100):
        day = datetime.today() - timedelta(days=delta)
        date_str = day.strftime("%Y-%m-%d")
        print(f"📅 Przetwarzanie dnia: {date_str}")
        matches = get_fixtures_for_date(date_str)
        for match in matches:
            print(f"⚽️  Mecz ID: {match['id']} — {match['home_team']} vs {match['away_team']}")
            stats = get_match_stats(match['id'], match['home_team'], match['away_team'])
            if stats:
                all_data.append(stats)
            else:
                print(f"❌ Brak statystyk dla meczu ID: {match['id']}")
    if all_data:
        df = pd.DataFrame(all_data)
        df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
        print(f"✅ Zapisano {len(df)} rekordów do {OUTPUT_FILE}")
    else:
        print("❌ Nie udało się zebrać danych.")

if __name__ == "__main__":
    run_monthly_scraper()
