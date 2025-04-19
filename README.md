# expectedcornersapp
# ⚽ Expected Corners App (XGBoost ML Model)

Aplikacja do przewidywania liczby rzutów rożnych w meczach piłkarskich na podstawie danych statystycznych zebranych z SofaScore (dawniej) oraz terminarza z API-Football. Projekt wykorzystuje model machine learning (XGBoost) i może służyć do wyszukiwania value betów na zakłady over/under corners.

## 🔍 Jak to działa?

- Dla każdego meczu model przewiduje **expected number of corners** (suma home + away).
- Dane wejściowe to zaawansowane statystyki zespołów: m.in. crossy, pressing, strzały, posiadanie piłki, zablokowane strzały, interakcje typu `crosses * pressing`.
- Do trenowania modelu wykorzystano dane z przeszłości zebrane ze scrapera SofaScore (obecnie niedostępne przez zabezpieczenia).
- **Fallback statystyki** zostały wyciągnięte z tych danych i zapisane w pliku `fallback_home_away_total.json`.

## 🧠 Model ML (XGBoost + Optuna)

- Model używa XGBoost Regressora optymalizowanego przez **Optuna** (100 prób z dużym zakresem).
- Używane cechy to m.in.:
  - `crosses`, `shots`, `possession`, `pressing`
  - `crosses_per_shot`, `crosses_per_pressing`
  - `interaction_crosses_pressing`, `avg_corners_conceded`, `team_strength`, `is_home`
- Dane są log-transformowane (log1p) podczas treningu, a predykcje są odwrotnie przekształcane (`expm1`).

## 🧪 Testowanie skuteczności

- Do testowania modelu służy osobny skrypt, który **zalicza zakład**, gdy expected corners przekracza daną linię (granie pod overy):
  - Przykład: jeśli linia to 10.5, a model daje 10.59 → zaliczone jako over. 
  - Działa to na zasadzie zaokrąglania:
    - `8.4` → 8
    - `8.6` → 9

Na dzień 13.04:
✅ Skuteczność: 56.31% (2029/3603)
📉 Średni błąd bezwzględny: 2.78
❌ Ekstremalnych przestrzeleń (>=5): 711


## 💾 Fallback i dane

- **Fallback** zawiera statystyki drużyn uśrednione (home/away/total) i pozwala przewidzieć expected corners nawet bez dostępu do danych live.
- Znajdują się tam statystyki takie jak:
  - `total_crosses`, `total_shots`, `total_pressing`, `blocked_shots`, `gk_saves`, `possession`, `corners_when_losing`, `corners_when_winning`, `team_corner_ratio` i inne.

## 🗂 Struktura folderu `skrypt/`

- `expectedcornersapp.py` — główny skrypt Streamlit do przewidywań i treningu modelu
- `fallback_home_away_total.json` — dane wejściowe statystyk drużyn
- `team_name_map.json` — mapa nazw drużyn (API-Football vs SofaScore)
- `models/` — katalog z modelem XGBoost i cechami
- `data/` — cache terminarza z API-Football

## 🔗 Dane i integracje

- 📅 Terminarz: pobierany z **API-Football**
- 📊 Statystyki: historyczne dane z **SofaScore**
- 🔄 Mapowanie drużyn: użycie `team_name_map.json` do dopasowania nazw z różnych źródeł

## 🛑 Uwaga

Scrapery SofaScore są obecnie **nieaktywne** z powodu blokad po stronie serwisu. Wszystkie predykcje opierają się na fallbackach lub mogą zostać zaktualizowane po integracji z nowym źródłem statystyk.

---

> Projekt edukacyjny w ramach nauki Machine Learning, DevOps i analizy danych sportowych.
> Wykorzystano ogólną wiedzę, oraz pomoc sztucznej inteligencji (Chat GPT)

