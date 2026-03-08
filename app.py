from flask import Flask, render_template, request, jsonify
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import os
import threading
import time
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ─── Feature config (must match train_model.py) ─────────────────────────────
CATEGORICAL_COLS = ['team1', 'team2', 'venue', 'toss_winner', 'toss_decision']
NUMERIC_COLS = ['elo_team1', 'elo_team2', 'elo_diff',
                'team1_form_5', 'team2_form_5',
                'team1_form_10', 'team2_form_10',
                'h2h_win_pct']

# ─── Country flag mapping (ISO 3166-1 alpha-2) ─────────────────────────────
TEAM_FLAGS = {
    "Afghanistan": "af", "Australia": "au", "Bangladesh": "bd",
    "Bermuda": "bm", "Canada": "ca", "England": "gb-eng",
    "Hong Kong": "hk", "India": "in", "Ireland": "ie",
    "Kenya": "ke", "Malaysia": "my", "Namibia": "na",
    "Nepal": "np", "Netherlands": "nl", "New Zealand": "nz",
    "Nigeria": "ng", "Oman": "om", "Pakistan": "pk",
    "Papua New Guinea": "pg", "Scotland": "gb-sct",
    "Singapore": "sg", "South Africa": "za", "Sri Lanka": "lk",
    "Uganda": "ug", "United Arab Emirates": "ae",
    "United States of America": "us", "West Indies": "jm",
    "Zimbabwe": "zw", "Jersey": "je", "Guernsey": "gg",
    "Denmark": "dk", "Germany": "de", "Italy": "it",
    "Norway": "no", "Qatar": "qa", "Rwanda": "rw",
    "Kuwait": "kw", "Botswana": "bw", "Ghana": "gh",
    "Tanzania": "tz", "Cayman Islands": "ky",
    "ICC World XI": "un",
}

def flag_url(team):
    code = TEAM_FLAGS.get(team, "un")
    return f"https://flagcdn.com/h40/{code}.png"

# ─── PyTorch model definition (must match train_model.py) ───────────────────
class MultiOutputDNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.shared1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.2)
        self.shared2 = nn.Linear(256, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.drop2 = nn.Dropout(0.3)
        self.shared3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.drop3 = nn.Dropout(0.2)
        self.shared4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.drop4 = nn.Dropout(0.15)
        self.out_winner = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())
        self.branch_r1 = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        self.branch_w1 = nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1))
        self.branch_r2 = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        self.branch_w2 = nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1))
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.shared1(x)))
        x = self.drop1(x)
        x = self.relu(self.bn2(self.shared2(x)))
        x = self.drop2(x)
        x = self.relu(self.bn3(self.shared3(x)))
        x = self.drop3(x)
        x = self.relu(self.bn4(self.shared4(x)))
        x = self.drop4(x)
        winner = self.out_winner(x)
        r1 = self.branch_r1(x)
        w1 = self.branch_w1(x)
        r2 = self.branch_r2(x)
        w2 = self.branch_w2(x)
        return torch.cat([winner, r1, w1, r2, w2], dim=1)

# ─── App setup ──────────────────────────────────────────────────────────────
app = Flask(__name__)
model = None
preprocessor = None
match_df = None  # Full dataset for H2H lookup

# Auto-update state
AUTO_UPDATE_INTERVAL = 6 * 60 * 60  # 6 hours in seconds
auto_update_status = {
    'last_fetch': None,
    'last_retrain': None,
    'fetched_count': 0,
    'is_training': False,
    'auto_learning': True,
    'last_error': None,
}

def load_resources():
    global model, preprocessor, match_df
    csv_path = 'world_cup_last_30_years.csv'
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        match_df = df[~df['tournament_name'].str.contains('Women|women', na=False, case=False)]
    
    if os.path.exists('cricket_dl_model.pth') and os.path.exists('preprocessor.pkl'):
        preprocessor = joblib.load('preprocessor.pkl')
        scaler = preprocessor['scaler']
        input_dim = scaler.mean_.shape[0]
        model = MultiOutputDNN(input_dim)
        model.load_state_dict(torch.load('cricket_dl_model.pth', weights_only=True))
        model.eval()
        print(f"✅ Model loaded (input_dim={input_dim}).")
    else:
        print("⚠️  Model files not found. Run train_model.py first.")


def _compute_team_stats(team1, team2):
    """Compute ELO, form, and H2H statistics from the match dataset."""
    defaults = {
        'elo_team1': 1500.0, 'elo_team2': 1500.0, 'elo_diff': 0.0,
        'team1_form_5': 0.5, 'team2_form_5': 0.5,
        'team1_form_10': 0.5, 'team2_form_10': 0.5,
        'h2h_win_pct': 0.5,
    }
    if match_df is None or match_df.empty:
        return defaults

    df = match_df.copy()

    # Latest ELO from the dataset
    t1_rows = df[(df['team1'] == team1) | (df['team2'] == team1)].sort_values('date', ascending=False)
    t2_rows = df[(df['team1'] == team2) | (df['team2'] == team2)].sort_values('date', ascending=False)

    if not t1_rows.empty:
        latest = t1_rows.iloc[0]
        defaults['elo_team1'] = float(latest['elo_team1'] if latest['team1'] == team1 else latest['elo_team2'])
    if not t2_rows.empty:
        latest = t2_rows.iloc[0]
        defaults['elo_team2'] = float(latest['elo_team1'] if latest['team1'] == team2 else latest['elo_team2'])

    defaults['elo_diff'] = defaults['elo_team1'] - defaults['elo_team2']

    # Recent form (last 5 and 10 matches)
    def calc_form(team, n):
        matches = df[(df['team1'] == team) | (df['team2'] == team)].sort_values('date', ascending=False).head(n)
        if matches.empty:
            return 0.5
        wins = (matches['winner'] == team).sum()
        return wins / len(matches)

    defaults['team1_form_5'] = calc_form(team1, 5)
    defaults['team2_form_5'] = calc_form(team2, 5)
    defaults['team1_form_10'] = calc_form(team1, 10)
    defaults['team2_form_10'] = calc_form(team2, 10)

    # H2H win %
    h2h = df[((df['team1'] == team1) & (df['team2'] == team2)) |
             ((df['team1'] == team2) & (df['team2'] == team1))]
    if not h2h.empty:
        t1_wins = (h2h['winner'] == team1).sum()
        defaults['h2h_win_pct'] = t1_wins / len(h2h)
    else:
        defaults['h2h_win_pct'] = 0.5

    return defaults


def retrain_model():
    """Retrain the model on the full (updated) CSV dataset."""
    global model, preprocessor, match_df, auto_update_status

    csv_path = 'world_cup_last_30_years.csv'
    if not os.path.exists(csv_path):
        print("⚠️  CSV not found, cannot retrain.")
        return False

    auto_update_status['is_training'] = True
    print("🧠 Retraining model on updated dataset...")

    try:
        df = pd.read_csv(csv_path)
        df = df[df['result_type'] != 'no result']
        df = df[df['result_type'] != 'tie']
        df = df[~df['tournament_name'].str.contains('Women|women', na=False, case=False)]
        df = df.dropna(subset=['team1', 'team2', 'venue', 'toss_winner', 'toss_decision', 'winner',
                               'innings1_runs', 'innings1_wkts', 'innings2_runs', 'innings2_wkts',
                               'batting_first', 'chasing_team'])

        # Normalize: team1 = batting first, team2 = chasing
        needs_swap = df['team1'] != df['batting_first']
        df.loc[needs_swap, ['team1', 'team2']] = df.loc[needs_swap, ['team2', 'team1']].values
        df.loc[needs_swap, ['elo_team1', 'elo_team2']] = df.loc[needs_swap, ['elo_team2', 'elo_team1']].values
        df.loc[needs_swap, ['team1_form_5', 'team2_form_5']] = df.loc[needs_swap, ['team2_form_5', 'team1_form_5']].values
        df.loc[needs_swap, ['team1_form_10', 'team2_form_10']] = df.loc[needs_swap, ['team2_form_10', 'team1_form_10']].values
        df.loc[needs_swap, 'elo_diff'] = -df.loc[needs_swap, 'elo_diff']
        df.loc[needs_swap, 'h2h_win_pct'] = 1 - df.loc[needs_swap, 'h2h_win_pct']

        X_cat = df[CATEGORICAL_COLS].copy()
        encoders = {}
        for col in CATEGORICAL_COLS:
            le = LabelEncoder()
            X_cat[col] = le.fit_transform(X_cat[col].astype(str))
            encoders[col] = le

        X_num = df[NUMERIC_COLS].copy()
        X_num = X_num.fillna({
            'elo_team1': 1500.0, 'elo_team2': 1500.0, 'elo_diff': 0.0,
            'team1_form_5': 0.5, 'team2_form_5': 0.5,
            'team1_form_10': 0.5, 'team2_form_10': 0.5,
            'h2h_win_pct': 0.5,
        })

        X = pd.concat([X_cat, X_num], axis=1)
        y_runs_1 = df['innings1_runs'].values.astype(np.float32)
        y_wkts_1 = df['innings1_wkts'].values.astype(np.float32)
        y_runs_2 = df['innings2_runs'].values.astype(np.float32)
        y_wkts_2 = df['innings2_wkts'].values.astype(np.float32)
        y_winner = np.where(df['winner'] == df['team2'], 1.0, 0.0).astype(np.float32)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X).astype(np.float32)
        y = np.stack([y_winner, y_runs_1, y_wkts_1, y_runs_2, y_wkts_2], axis=1)

        train_dataset = TensorDataset(torch.tensor(X_scaled), torch.tensor(y))
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        new_model = MultiOutputDNN(input_dim=X_scaled.shape[1])
        optimizer = optim.Adam(new_model.parameters(), lr=0.001)
        bce_loss = nn.BCELoss()
        mse_loss = nn.MSELoss()

        new_model.train()
        epochs = 60
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = new_model(batch_X)
                loss_winner = bce_loss(outputs[:, 0], batch_y[:, 0])
                loss_r1 = mse_loss(outputs[:, 1], batch_y[:, 1])
                loss_w1 = mse_loss(outputs[:, 2], batch_y[:, 2])
                loss_r2 = mse_loss(outputs[:, 3], batch_y[:, 3])
                loss_w2 = mse_loss(outputs[:, 4], batch_y[:, 4])
                loss = loss_winner * 100 + loss_r1 + loss_w1 * 10 + loss_r2 + loss_w2 * 10
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 15 == 0:
                print(f"   Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")

        # Save and hot-swap
        new_model.eval()
        torch.save(new_model.state_dict(), 'cricket_dl_model.pth')
        new_preprocessor = {'encoders': encoders, 'scaler': scaler}
        joblib.dump(new_preprocessor, 'preprocessor.pkl')

        model = new_model
        preprocessor = new_preprocessor
        full_df = pd.read_csv(csv_path)
        match_df = full_df[~full_df['tournament_name'].str.contains('Women|women', na=False, case=False)]

        auto_update_status['last_retrain'] = datetime.now().isoformat()
        auto_update_status['is_training'] = False
        print("✅ Model retrained and hot-swapped successfully.")
        return True

    except Exception as e:
        auto_update_status['is_training'] = False
        auto_update_status['last_error'] = str(e)
        print(f"❌ Retrain error: {e}")
        return False


def auto_update_loop():
    """Background thread: fetch new match data + retrain periodically."""
    while True:
        time.sleep(AUTO_UPDATE_INTERVAL)
        if not auto_update_status['auto_learning']:
            continue

        try:
            from fetch_matches import fetch_and_update
            count = fetch_and_update()
            auto_update_status['last_fetch'] = datetime.now().isoformat()
            auto_update_status['fetched_count'] += count

            if count > 0:
                print(f"📊 {count} new match(es) found. Triggering retrain...")
                retrain_model()
            else:
                print("🔄 No new matches. Skipping retrain.")
        except Exception as e:
            auto_update_status['last_error'] = str(e)
            print(f"⚠️  Auto-update error: {e}")


load_resources()

# Start background auto-update thread
_update_thread = threading.Thread(target=auto_update_loop, daemon=True)
_update_thread.start()
print("🔄 Background auto-update scheduler started (every 6 hours).")


@app.route('/')
def index():
    teams, venues = [], []
    if preprocessor:
        teams = sorted(list(preprocessor['encoders']['team1'].classes_))
        venues = sorted(list(preprocessor['encoders']['venue'].classes_))
    flags = {t: flag_url(t) for t in teams}
    return render_template('index.html', teams=teams, venues=venues, flags=flags)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if not model or not preprocessor:
        return jsonify({'error': 'Model not loaded. Run train_model.py first.'}), 500

    try:
        team1, team2 = data['team1'], data['team2']
        venue = data['venue']
        toss_winner = data['toss_winner']
        toss_decision = data['toss_decision']

        encoders, scaler = preprocessor['encoders'], preprocessor['scaler']

        def encode_safe(col, val):
            return int(encoders[col].transform([val])[0]) if val in encoders[col].classes_ else 0

        # Compute live stats from dataset
        stats = _compute_team_stats(team1, team2)

        # Build 13-feature vector: 5 categorical (encoded) + 8 numeric
        feat = np.array([[
            encode_safe('team1', team1),
            encode_safe('team2', team2),
            encode_safe('venue', venue),
            encode_safe('toss_winner', toss_winner),
            encode_safe('toss_decision', toss_decision),
            stats['elo_team1'],
            stats['elo_team2'],
            stats['elo_diff'],
            stats['team1_form_5'],
            stats['team2_form_5'],
            stats['team1_form_10'],
            stats['team2_form_10'],
            stats['h2h_win_pct'],
        ]], dtype=np.float32)

        feat_scaled = scaler.transform(feat).astype(np.float32)

        with torch.no_grad():
            preds = model(torch.tensor(feat_scaled)).numpy()[0]

        winner_prob = float(preds[0])
        runs_1 = max(50, int(round(float(preds[1]))))
        wkts_1 = min(10, max(0, int(round(float(preds[2])))))
        runs_2 = max(50, int(round(float(preds[3]))))
        wkts_2 = min(10, max(0, int(round(float(preds[4])))))

        winner = team2 if winner_prob > 0.5 else team1
        win_prob_display = winner_prob if winner_prob > 0.5 else (1 - winner_prob)
        loser_prob = round((1 - win_prob_display) * 100, 1)

        # Ensure consistency: winner's score should be >= loser's score
        if winner == team2 and runs_2 < runs_1:
            runs_1, runs_2 = runs_2, runs_1
            wkts_1, wkts_2 = wkts_2, wkts_1
        elif winner == team1 and runs_1 < runs_2:
            runs_1, runs_2 = runs_2, runs_1
            wkts_1, wkts_2 = wkts_2, wkts_1

        return jsonify({
            'winner': winner,
            'loser': team1 if winner == team2 else team2,
            'probability': round(win_prob_display * 100, 1),
            'loser_probability': loser_prob,
            'winner_flag': flag_url(winner),
            'loser_flag': flag_url(team1 if winner == team2 else team2),
            'innings1': {'team': team1, 'runs': runs_1, 'wickets': wkts_1, 'flag': flag_url(team1)},
            'innings2': {'team': team2, 'runs': runs_2, 'wickets': wkts_2, 'flag': flag_url(team2)},
            # Extra data for charts
            'stats': {
                'elo_team1': round(stats['elo_team1'], 1),
                'elo_team2': round(stats['elo_team2'], 1),
                'form5_team1': round(stats['team1_form_5'] * 100, 1),
                'form5_team2': round(stats['team2_form_5'] * 100, 1),
                'form10_team1': round(stats['team1_form_10'] * 100, 1),
                'form10_team2': round(stats['team2_form_10'] * 100, 1),
                'h2h_pct': round(stats['h2h_win_pct'] * 100, 1),
            },
        })

    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/history', methods=['POST'])
def history():
    """Return the last 8 T20 head-to-head matches between two men's teams."""
    if match_df is None:
        return jsonify({'matches': []})

    data = request.json
    team1, team2 = data['team1'], data['team2']

    df = match_df.copy()
    h2h = df[
        ((df['team1'] == team1) & (df['team2'] == team2)) |
        ((df['team1'] == team2) & (df['team2'] == team1))
    ].copy()

    h2h = h2h[h2h['result_type'].isin(['completed', 'unknown'])]
    h2h = h2h.dropna(subset=['winner', 'date'])
    h2h = h2h.sort_values('date', ascending=False).head(8)

    matches = []
    for _, row in h2h.iterrows():
        try:
            batting_first = str(row.get('batting_first', ''))
            chasing = str(row.get('chasing_team', ''))
            score1 = f"{int(row['first_innings_score'])}"
            score2 = f"{int(row['second_innings_score'])}" if str(row['second_innings_score']) not in ['nan', 'NaN', ''] else '—'
            matches.append({
                'date': str(row['date']),
                'venue': str(row['venue']),
                'batting_first': batting_first,
                'batting_first_flag': flag_url(batting_first),
                'chasing_team': chasing,
                'chasing_flag': flag_url(chasing),
                'score1': score1,
                'score2': score2,
                'winner': str(row['winner']),
                'winner_flag': flag_url(str(row['winner'])),
                'tournament': str(row.get('tournament_name', '')),
            })
        except Exception:
            continue

    return jsonify({'matches': matches, 'team1': team1, 'team2': team2})


# ─── Admin API endpoints ────────────────────────────────────────────────────

@app.route('/api/fetch-data', methods=['POST'])
def api_fetch_data():
    """Manually trigger a data fetch from the cricket API."""
    try:
        from fetch_matches import fetch_and_update
        count = fetch_and_update()
        auto_update_status['last_fetch'] = datetime.now().isoformat()
        auto_update_status['fetched_count'] += count
        return jsonify({'success': True, 'new_matches': count})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/retrain', methods=['POST'])
def api_retrain():
    """Manually trigger a model retrain."""
    if auto_update_status['is_training']:
        return jsonify({'success': False, 'error': 'Training already in progress.'}), 409

    def bg_retrain():
        retrain_model()

    threading.Thread(target=bg_retrain, daemon=True).start()
    return jsonify({'success': True, 'message': 'Retrain started in background.'})


@app.route('/api/status')
def api_status():
    """Return current auto-update and model status."""
    csv_path = 'world_cup_last_30_years.csv'
    total_matches = 0
    if os.path.exists(csv_path):
        total_matches = sum(1 for _ in open(csv_path, encoding='utf-8')) - 1

    return jsonify({
        'model_loaded': model is not None,
        'total_matches_in_dataset': total_matches,
        'auto_learning': auto_update_status['auto_learning'],
        'is_training': auto_update_status['is_training'],
        'last_fetch': auto_update_status['last_fetch'],
        'last_retrain': auto_update_status['last_retrain'],
        'fetched_count': auto_update_status['fetched_count'],
        'last_error': auto_update_status['last_error'],
    })


@app.route('/api/toggle-learning', methods=['POST'])
def api_toggle_learning():
    """Toggle auto-learning on/off."""
    auto_update_status['auto_learning'] = not auto_update_status['auto_learning']
    state = 'enabled' if auto_update_status['auto_learning'] else 'disabled'
    print(f"🔄 Auto-learning {state}.")
    return jsonify({'auto_learning': auto_update_status['auto_learning']})


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
