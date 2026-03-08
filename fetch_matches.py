"""
Auto-fetch completed T20 international match data from CricketData.org API
and append new matches to the local CSV dataset.

Uses the free CricAPI v1 endpoints:
  - /v1/currentMatches  – recently completed + live matches
  - /v1/match_info      – detail for a single match

Environment variable CRICAPI_KEY must be set to your free API key.
Get one for free at https://cricketdata.org/
"""

import os
import csv
import hashlib
import requests
import pandas as pd
from datetime import datetime

API_KEY = os.environ.get("CRICAPI_KEY", "")
BASE_URL = "https://api.cricapi.com/v1"
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "world_cup_last_30_years.csv")
SEEN_IDS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".fetched_match_ids.txt")


def _load_seen_ids():
    """Load previously fetched match IDs to avoid duplicates."""
    if os.path.exists(SEEN_IDS_PATH):
        with open(SEEN_IDS_PATH, "r") as f:
            return set(line.strip() for line in f if line.strip())
    return set()


def _save_seen_id(match_id):
    with open(SEEN_IDS_PATH, "a") as f:
        f.write(f"{match_id}\n")


def _generate_match_id(row_dict):
    """Deterministic hash for dedup even without API IDs."""
    key = f"{row_dict.get('date','')}-{row_dict.get('team1','')}-{row_dict.get('team2','')}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def fetch_recent_t20i_matches():
    """
    Fetch recently completed T20I matches from the CricAPI free tier.
    Returns a list of dicts ready to append to the CSV.
    """
    if not API_KEY:
        print("⚠️  CRICAPI_KEY not set. Skipping auto-fetch.")
        print("   Get a free key at https://cricketdata.org/ and set:")
        print("   set CRICAPI_KEY=your_key_here  (Windows)")
        return []

    seen_ids = _load_seen_ids()
    new_rows = []

    try:
        # Fetch current/recent matches
        resp = requests.get(
            f"{BASE_URL}/currentMatches",
            params={"apikey": API_KEY, "offset": 0},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") != "success":
            print(f"⚠️  API returned: {data.get('reason', 'unknown error')}")
            return []

        matches = data.get("data", [])

        for m in matches:
            match_id = m.get("id", "")
            match_type = m.get("matchType", "").lower()
            status = m.get("status", "").lower()

            # Only completed T20I men's matches
            if match_type != "t20" and match_type != "t20i":
                continue
            if "won" not in status and "result" not in status:
                continue
            if match_id in seen_ids:
                continue

            # Extract team info
            teams = m.get("teams", [])
            if len(teams) < 2:
                continue

            team1 = teams[0]
            team2 = teams[1]
            venue = m.get("venue", "Unknown")
            date_str = m.get("date", "")
            date_iso = m.get("dateTimeGMT", date_str)

            # Parse scores from the score array
            score_list = m.get("score", [])
            if len(score_list) < 2:
                continue

            # score_list items: {"r": runs, "w": wickets, "o": overs, "inning": "Team Name Inning 1"}
            innings1 = score_list[0]
            innings2 = score_list[1]

            innings1_team = innings1.get("inning", "").replace(" Inning 1", "").replace(" Inning 2", "").strip()
            innings2_team = innings2.get("inning", "").replace(" Inning 1", "").replace(" Inning 2", "").strip()

            # Determine batting first and chasing
            batting_first = innings1_team if innings1_team else team1
            chasing_team = innings2_team if innings2_team else team2

            innings1_runs = innings1.get("r", 0)
            innings1_wkts = innings1.get("w", 0)
            innings1_overs = innings1.get("o", 0)
            innings2_runs = innings2.get("r", 0)
            innings2_wkts = innings2.get("w", 0)
            innings2_overs = innings2.get("o", 0)

            # Determine winner from status text
            winner = ""
            for t in teams:
                if t.lower() in status.lower():
                    winner = t
                    break

            if not winner:
                continue

            # Determine toss info (may not always be available in free tier)
            toss_winner = m.get("tossWinner", team1)
            toss_choice = m.get("tossChoice", "bat")

            # Parse date
            try:
                dt = datetime.fromisoformat(date_iso.replace("Z", "+00:00"))
                date_formatted = dt.strftime("%Y-%m-%d")
                year = dt.year
                month = dt.month
                season = str(year)
            except Exception:
                date_formatted = date_str[:10] if len(date_str) >= 10 else date_str
                year = date_formatted[:4] if len(date_formatted) >= 4 else "2025"
                month = date_formatted[5:7] if len(date_formatted) >= 7 else "1"
                season = str(year)

            # Determine tournament
            tournament = m.get("name", "T20I Series")

            # Build the row matching the CSV schema
            row = {
                "match_id": match_id or _generate_match_id({"date": date_formatted, "team1": team1, "team2": team2}),
                "date": date_formatted,
                "season": season,
                "tournament_name": tournament,
                "is_worldcup": False,
                "match_stage": "Group",
                "team1": team1,
                "team2": team2,
                "venue": venue,
                "city": "",
                "toss_winner": toss_winner,
                "toss_decision": toss_choice,
                "winner": winner,
                "result_type": "completed",
                "format": "T20",
                "innings1_team": batting_first,
                "innings1_runs": innings1_runs,
                "innings1_wkts": innings1_wkts,
                "innings1_overs": innings1_overs,
                "innings2_team": chasing_team,
                "innings2_runs": innings2_runs,
                "innings2_wkts": innings2_wkts,
                "innings2_overs": innings2_overs,
                "year": year,
                "month": month,
                "batting_first": batting_first,
                "chasing_team": chasing_team,
                "first_innings_score": innings1_runs,
                "second_innings_score": innings2_runs,
                "match_result": "completed",
                "elo_team1": 1500.0,
                "elo_team2": 1500.0,
                "elo_diff": 0.0,
                "team1_form_5": 0.5,
                "team2_form_5": 0.5,
                "team1_form_10": 0.5,
                "team2_form_10": 0.5,
                "h2h_win_pct": 0.5,
            }

            new_rows.append(row)
            _save_seen_id(match_id or row["match_id"])
            print(f"   ✅ New match: {team1} vs {team2} ({date_formatted}) → {winner} won")

    except requests.RequestException as e:
        print(f"⚠️  API fetch error: {e}")

    return new_rows


def append_to_csv(new_rows):
    """Append new match rows to the CSV file."""
    if not new_rows:
        return 0

    # Read existing CSV headers
    if os.path.exists(CSV_PATH):
        existing_df = pd.read_csv(CSV_PATH, nrows=0)
        fieldnames = list(existing_df.columns)
    else:
        fieldnames = list(new_rows[0].keys())

    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        for row in new_rows:
            writer.writerow(row)

    print(f"📥 Appended {len(new_rows)} new match(es) to CSV.")
    return len(new_rows)


def fetch_and_update():
    """Main entry point: fetch new matches and append to CSV."""
    print("🔄 Checking for new T20I match data...")
    new_rows = fetch_recent_t20i_matches()
    count = append_to_csv(new_rows)
    if count == 0:
        print("   No new matches found.")
    return count


if __name__ == "__main__":
    fetch_and_update()
