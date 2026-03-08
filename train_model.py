import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

# ─── Feature columns ────────────────────────────────────────────────────────
CATEGORICAL_COLS = ['team1', 'team2', 'venue', 'toss_winner', 'toss_decision']
NUMERIC_COLS = ['elo_team1', 'elo_team2', 'elo_diff',
                'team1_form_5', 'team2_form_5',
                'team1_form_10', 'team2_form_10',
                'h2h_win_pct']


def load_and_preprocess_data(filepath):
    print("Loading data...")
    df = pd.read_csv(filepath)
    df = df[df['result_type'] != 'no result']
    df = df[df['result_type'] != 'tie']
    
    # Filter to men's matches only
    df = df[~df['tournament_name'].str.contains('Women|women', na=False, case=False)]
    
    df = df.dropna(subset=['team1', 'team2', 'venue', 'toss_winner', 'toss_decision', 'winner', 
                           'innings1_runs', 'innings1_wkts', 'innings2_runs', 'innings2_wkts',
                           'batting_first', 'chasing_team'])
    
    # ─────────────────────────────────────────────────────────────────────
    # CRITICAL: Normalize rows so team1 = batting_first, team2 = chasing
    # In the raw CSV, team1 is NOT always the batting first team!
    # This ensures the model learns: team1 → 1st innings, team2 → 2nd innings
    # ─────────────────────────────────────────────────────────────────────
    needs_swap = df['team1'] != df['batting_first']
    swap_count = needs_swap.sum()
    print(f"Normalizing {swap_count}/{len(df)} rows where team1 != batting_first...")
    
    # Swap team1 ↔ team2 for rows where team2 batted first
    df.loc[needs_swap, ['team1', 'team2']] = df.loc[needs_swap, ['team2', 'team1']].values
    
    # Swap ELO/form stats accordingly
    df.loc[needs_swap, ['elo_team1', 'elo_team2']] = df.loc[needs_swap, ['elo_team2', 'elo_team1']].values
    df.loc[needs_swap, ['team1_form_5', 'team2_form_5']] = df.loc[needs_swap, ['team2_form_5', 'team1_form_5']].values
    df.loc[needs_swap, ['team1_form_10', 'team2_form_10']] = df.loc[needs_swap, ['team2_form_10', 'team1_form_10']].values
    df.loc[needs_swap, 'elo_diff'] = -df.loc[needs_swap, 'elo_diff']
    df.loc[needs_swap, 'h2h_win_pct'] = 1 - df.loc[needs_swap, 'h2h_win_pct']
    
    # After swap, verify alignment
    assert (df['team1'] == df['batting_first']).all(), "team1 must equal batting_first after normalization"
    print("✅ All rows normalized: team1 = batting first, team2 = chasing.")
    
    # ── Encode categorical features ─────────────────────────────────────
    X_cat = df[CATEGORICAL_COLS].copy()
    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        X_cat[col] = le.fit_transform(X_cat[col].astype(str))
        encoders[col] = le
    
    # ── Numeric features (fill missing with defaults) ────────────────────
    X_num = df[NUMERIC_COLS].copy()
    X_num = X_num.fillna({
        'elo_team1': 1500.0, 'elo_team2': 1500.0, 'elo_diff': 0.0,
        'team1_form_5': 0.5, 'team2_form_5': 0.5,
        'team1_form_10': 0.5, 'team2_form_10': 0.5,
        'h2h_win_pct': 0.5,
    })
    
    # ── Combine all features ─────────────────────────────────────────────
    X = pd.concat([X_cat, X_num], axis=1)
    
    # ── Targets ──────────────────────────────────────────────────────────
    # Now innings1_runs = team1's (batting first) runs, innings2_runs = team2's (chasing) runs
    y_runs_1 = df['innings1_runs'].values.astype(np.float32)
    y_wkts_1 = df['innings1_wkts'].values.astype(np.float32)
    y_runs_2 = df['innings2_runs'].values.astype(np.float32)
    y_wkts_2 = df['innings2_wkts'].values.astype(np.float32)
    
    # Winner: 0 = team1 (batting first) won, 1 = team2 (chasing) won
    y_winner = np.where(df['winner'] == df['team2'], 1.0, 0.0).astype(np.float32)
    
    # ── Scale ────────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)
    
    y = np.stack([y_winner, y_runs_1, y_wkts_1, y_runs_2, y_wkts_2], axis=1)
    
    feature_names = CATEGORICAL_COLS + NUMERIC_COLS
    print(f"Using {len(feature_names)} features: {feature_names}")
    
    # Print stats
    t1_wins = (y_winner == 0).sum()
    t2_wins = (y_winner == 1).sum()
    print(f"Winner distribution: Batting first wins {t1_wins} ({t1_wins/len(y_winner)*100:.1f}%), "
          f"Chasing wins {t2_wins} ({t2_wins/len(y_winner)*100:.1f}%)")
    print(f"Avg 1st innings: {y_runs_1.mean():.0f}/{y_wkts_1.mean():.1f}, "
          f"Avg 2nd innings: {y_runs_2.mean():.0f}/{y_wkts_2.mean():.1f}")
    
    return X_scaled, y, encoders, scaler


class MultiOutputDNN(nn.Module):
    def __init__(self, input_dim):
        super(MultiOutputDNN, self).__init__()
        # Wider shared layers for more features
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
        
        # Output Branches
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


def main():
    filepath = 'world_cup_last_30_years.csv'
    if not os.path.exists(filepath):
        print(f"Error: dataset {filepath} not found.")
        return
        
    X_scaled, y, encoders, scaler = load_and_preprocess_data(filepath)
    print(f"Dataset compiled. Shape: {X_scaled.shape}")
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    model = MultiOutputDNN(input_dim=X_scaled.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # Loss functions
    bce_loss = nn.BCELoss()
    mse_loss = nn.MSELoss()
    
    print("\nTraining model...")
    epochs = 80
    best_acc = 0
    best_state = None
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            
            loss_winner = bce_loss(outputs[:, 0], batch_y[:, 0])
            loss_r1 = mse_loss(outputs[:, 1], batch_y[:, 1])
            loss_w1 = mse_loss(outputs[:, 2], batch_y[:, 2])
            loss_r2 = mse_loss(outputs[:, 3], batch_y[:, 3])
            loss_w2 = mse_loss(outputs[:, 4], batch_y[:, 4])
            
            loss = loss_winner * 100 + loss_r1 + loss_w1 * 10 + loss_r2 + loss_w2 * 10
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step()
        
        if (epoch + 1) % 5 == 0:
            # Evaluate accuracy on test set
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    out = model(batch_X)
                    preds = (out[:, 0] > 0.5).float()
                    correct += (preds == batch_y[:, 0]).sum().item()
                    total += batch_y.shape[0]
            acc = correct / total * 100
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}, Test Accuracy: {acc:.1f}%")
            if acc > best_acc:
                best_acc = acc
                best_state = model.state_dict().copy()
    
    print(f"\n🏆 Best Test Accuracy: {best_acc:.1f}%")
    
    # Restore best model state
    if best_state:
        model.load_state_dict(best_state)
    
    # Save the best model
    model.eval()
    torch.save(model.state_dict(), 'cricket_dl_model.pth')
    print("Model saved to cricket_dl_model.pth")
    
    # Save preprocessors
    joblib.dump({'encoders': encoders, 'scaler': scaler}, 'preprocessor.pkl')
    print("Preprocessing pipeline saved to preprocessor.pkl")

if __name__ == '__main__':
    main()
