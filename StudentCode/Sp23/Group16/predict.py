import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay

"""
event_stats table:
0 team_id | 1 event_id | 2 opponent_id | 3 is_home | 4 is_winner | 5 moneyline_odds | 6 runs_scored (this event) | 

7 runs_allowed (this event) | 8 ops (this event) | 9 sp_era (pitcher career) | 10 win_pct (up to event) | 

11 home_win_pct (up to event) | 12 away_win_pct (up to event) | 13 prev_event | 14 next_event

Input feature vector for team:
[runs_scored (aggregated avg), runs_allowed (aggregated avg), ops (aggregated avg), sp_era, win_pct, home/away_win_pct]

"""

AGGREGATION_SIZE = 30


def get_team_features(team_stats, aggregation_size, conn):
    prev_events_list = []

    # Loop until aggregation_size is hit or there is not a previous game, skip games with winner=-1
    cur_event_stats = team_stats

    while len(prev_events_list) < aggregation_size:

        prev_event_id = cur_event_stats[13]

        # Make sure there is a previous event in db
        has_prev_event = conn.execute("SELECT EXISTS(SELECT 1 FROM event_stats WHERE team_id = ? AND event_id = ?)",
                                      (team_stats[0], prev_event_id)).fetchone()[0]
        if not has_prev_event:
            break

        # Grab team's previous event stats
        prev_event_stats = conn.execute("SELECT * FROM event_stats WHERE team_id = ? AND event_id = ?",
                                        (team_stats[0], prev_event_id)).fetchone()

        # If game was rained out, continue to next iteration, else add to list of prev events
        cur_event_stats = prev_event_stats
        if prev_event_stats[4] == -1:
            continue
        else:
            prev_events_list.append(prev_event_stats)

    # Create feature vector and aggregate data from past events
    runs_scored = 0
    runs_allowed = 0
    ops = 0
    win_pct = 0
    split_win_pct = 0

    for event in prev_events_list:
        runs_scored += event[6]
        runs_allowed += event[7]
        ops += event[8]

    if len(prev_events_list) > 0:
        win_pct = prev_events_list[0][10]
        if prev_events_list[0][3] == 1:
            split_win_pct = prev_events_list[0][11]
        else:
            split_win_pct = prev_events_list[0][12]

        runs_scored = runs_scored / len(prev_events_list)
        runs_allowed = runs_allowed / len(prev_events_list)
        ops = ops / len(prev_events_list)

    return [runs_scored, runs_allowed, ops, team_stats[9], win_pct, split_win_pct]


# Simulate betting $100 on the moneyline of each predicted winner, return final balance
def simulate_betting(y_pred, y_test, odds):
    odds_test = odds[-len(y_pred):]
    balance = 0
    balance_over_time = []
    num_bets_hit = 0
    total_bets_taken = 0

    for i in range(len(y_pred)):
        if odds_test[i][0] == 0 or odds_test[i][1] == 0:
            continue

        total_bets_taken += 1

        # Correctly predicted outcome
        if y_pred[i] == y_test[i]:
            ml_odds = odds_test[i][y_pred[i]]
            num_bets_hit += 1

            # Simulate payout
            if ml_odds > 0:
                balance += ml_odds
            else:
                balance += (100 / ((ml_odds * -1) / 100))

        else:
            balance -= 100

        balance_over_time.append(balance)

    print("$100 unit size balance over", len(odds_test), "games: $", balance)
    print("Bet win percentage:", num_bets_hit / total_bets_taken * 100, "%")
    x_axis = list(range(len(balance_over_time)))
    plt.plot(x_axis, balance_over_time)
    plt.title("Balance Over Time")
    plt.xlabel("Number of games bet on")
    plt.ylabel("Balance ($)")
    plt.show()


def main():

    connection = sqlite3.connect('mlb2022')

    # Get list of each event_id
    event_ids = []
    cursor = connection.execute("SELECT DISTINCT event_id FROM event_stats WHERE is_winner != -1 ORDER BY event_id ASC;")
    for row in cursor:
        event_ids.append(row[0])

    # Iterate through each event and create feature vector, outcome vector, and odds vector
    X = []
    y = []
    odds = []

    for event_id in event_ids:
        # Grab both teams' stats for current event
        teams_stats = connection.execute("SELECT * FROM event_stats WHERE event_id = ? ORDER BY is_home DESC;",
                                         (event_id,)).fetchall()

        home_stats = teams_stats[0]
        away_stats = teams_stats[1]

        # If starting pitcher era is none, skip game (only one in db)
        if not home_stats[9] or not away_stats[9]:
            continue

        # Get aggregated features for both teams and append input feature vector to X
        home_features = get_team_features(home_stats, AGGREGATION_SIZE, connection)
        away_features = get_team_features(away_stats, AGGREGATION_SIZE, connection)

        event_features = home_features + away_features
        X.append(event_features)

        # Append outcome to y, home team won = 0, away team won = 1
        if home_stats[4] == 1:
            y.append(0)
        else:
            y.append(1)

        # Append odds data
        odds.append((home_stats[5], away_stats[5]))

    connection.close()

    # Trim and scale datasets
    X_pd = pd.DataFrame(np.array(X))
    X_train, X_test, y_train, y_test = train_test_split(X_pd, y, shuffle=False, test_size=0.25)
    sc_X = StandardScaler()
    X_trainscaled = sc_X.fit_transform(X_train)
    X_testscaled = sc_X.transform(X_test)

    # Train network
    clf = MLPClassifier(hidden_layer_sizes=(3,), activation="relu", random_state=1, solver="lbfgs", max_iter=1000)\
        .fit(X_trainscaled, y_train)

    # Predict test data
    y_pred = clf.predict(X_testscaled)

    print("Accuracy of model:", clf.score(X_testscaled, y_test) * 100, "%")

    fig = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["Home win", "Away win"])
    fig.figure_.suptitle("Confusion Matrix for MLB Dataset")
    plt.show()

    simulate_betting(y_pred, y_test, odds)


if __name__ == "__main__":
    main()
