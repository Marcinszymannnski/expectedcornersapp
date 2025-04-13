import requests
import pandas as pd
from datetime import datetime, timedelta

OUTPUT_FILE = "sofascore_top5_corners_month.csv"
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
                "away_team": e["awayTeam"]["name"],
                "date": date_str
            }
            for e in events
            if e["tournament"]["uniqueTournament"]["id"] in TOP5_LEAGUE_IDS
        ]
    except:
        return []

def get_corner_stats(match_id, home_team, away_team, date):
    url = f"https://api.sofascore.com/api/v1/event/{match_id}/statistics"
    resp = requests.get(url, headers=HEADERS)
    if resp.status_code != 200:
        return None

    data = resp.json()
    stats = {
        "match_id": match_id,
        "date": date,
        "home_team": home_team,
        "away_team": away_team,
        "home_corner_kicks": None,
        "away_corner_kicks": None,
        "total_corners": None
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
                    try:
                        home = int(home)
                        away = int(away)
                        stats["home_corner_kicks"] = home
                        stats["away_corner_kicks"] = away
                        stats["total_corners"] = home + away
                    except (ValueError, TypeError):
                        pass
                    return stats  # kończymy — tylko corner kicks nas interesuje

    return None



def run_monthly_scraper():
    all_data = []
    for delta in range(400):
        day = datetime.today() - timedelta(days=delta)
        date_str = day.strftime("%Y-%m-%d")
        print(f"📅 Przetwarzanie dnia: {date_str}")
        matches = get_fixtures_for_date(date_str)
        for match in matches:
            print(f"⚽️  Mecz ID: {match['id']} — {match['home_team']} vs {match['away_team']}")
            stats = get_corner_stats(match['id'], match['home_team'], match['away_team'], match['date'])
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
