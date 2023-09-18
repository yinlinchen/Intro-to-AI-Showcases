import requests
import json
import sqlite3

TEAM_IDS = {'BAL': 1, 'BOS': 2, 'LAA': 3, 'CHW': 4, 'CLE': 5, 'DET': 6, 'KC': 7, 'MIL': 8, 'MIN': 9, 'NYY': 10,
            'OAK': 11, 'SEA': 12, 'TEX': 13, 'TOR': 14, 'ATL': 15, 'CHC': 16, 'CIN': 17, 'HOU': 18, 'LAD': 19, 'WSH': 20,
            'NYM': 21, 'PHI': 22, 'PIT': 23, 'STL': 24, 'SD': 25, 'SF': 26, 'COL': 27, 'MIA': 28, 'ARI': 29, 'TB': 30}

URL = 'https://sports.core.api.espn.com/v2/sports/baseball/leagues/mlb/seasons/2022/types/2/'


def print_json(data):
    print(json.dumps(data, indent=4))


def create_team_table(conn):
    conn.execute("""CREATE TABLE team (
                id INT PRIMARY KEY NOT NULL,
                abbr VARCHAR(3) NOT NULL)""")
    for abbr, team_id in TEAM_IDS.items():
        conn.execute("INSERT INTO team (id, abbr) VALUES (?, ?)", (team_id, abbr))
    conn.commit()


def create_event_table(conn):
    conn.execute("""CREATE TABLE event_stats (
                    team_id INT NOT NULL,
                    event_id INT NOT NULL,
                    opponent_id INT NOT NULL,
                    is_home INT NOT NULL,
                    is_winner INT NOT NULL,
                    moneyline_odds INT,
                    runs_scored INT NOT NULL,
                    runs_allowed INT NOT NULL,
                    ops INT NOT NULL,
                    sp_era INT,
                    win_pct INT NOT NULL,
                    home_win_pct INT NOT NULL,
                    away_win_pct INT NOT NULL,
                    prev_event INT,
                    next_event INT,
                    PRIMARY KEY (team_id, event_id),
                    FOREIGN KEY (team_id) REFERENCES team(id),
                    FOREIGN KEY (opponent_id) REFERENCES team(id))""")


def reset_db(conn):
    conn.execute("DROP TABLE team")
    conn.execute("DROP TABLE event_stats")
    create_team_table(conn)
    create_event_table(conn)


def process_team_event_stats(team_json):
    team_stats = {'id': int(team_json['id']),
                  'is_home': 1 if team_json['homeAway'] == "home" else 0,
                  'is_winner': 1 if team_json['winner'] else 0}

    event_stats_page = requests.get(team_json['statistics']['$ref']).json()
    team_stats['runs_scored'] = event_stats_page['splits']['categories'][0]['stats'][11]['value']
    team_stats['runs_allowed'] = event_stats_page['splits']['categories'][1]['stats'][11]['value']
    team_stats['ops'] = event_stats_page['splits']['categories'][0]['stats'][41]['value']

    record_stats_page = requests.get(team_json['record']['$ref']).json()
    team_stats['win_pct'] = record_stats_page['items'][0]['stats'][16]['value']
    team_stats['home_win_pct'] = record_stats_page['items'][1]['stats'][3]['value']
    team_stats['away_win_pct'] = record_stats_page['items'][2]['stats'][3]['value']

    if 'probables' in team_json:
        if 'statistics' in team_json['probables'][0]:
            if '$ref' in team_json['probables'][0]['statistics']:
                starting_pitcher_page = requests.get(team_json['probables'][0]['statistics']['$ref']).json()
                team_stats['sp_era'] = starting_pitcher_page['splits']['categories'][0]['stats'][52]['value']

    if 'sp_era' not in team_stats.keys():
        team_stats['sp_era'] = None

    if 'previousCompetition' in team_json and '$ref' in team_json['previousCompetition']:
        prev_link_string = team_json['previousCompetition']['$ref']
        team_stats['prev_event'] = int(prev_link_string[:-18].split('/')[-1])
    else:
        team_stats['prev_event'] = 0

    if 'nextCompetition' in team_json and '$ref' in team_json['nextCompetition']:
        next_link_string = team_json['nextCompetition']['$ref']
        team_stats['next_event'] = int(next_link_string[:-18].split('/')[-1])
    else:
        team_stats['next_event'] = 0

    return team_stats


def get_api_data(conn):

    event_count = 0

    # Get list of event links on each page
    for page_index in range(1, 101):
        cur_page_data = requests.get(URL + 'events?page=' + str(page_index)).json()
        cur_event_links = cur_page_data['items']

        # Iterate through each event in current page
        for event_link in cur_event_links:
            cur_event_data = requests.get(event_link['$ref']).json()

            # Make sure game has a winner (was not postponed)
            if 'winner' in cur_event_data['competitions'][0]['competitors'][0]:
                event_id = int(cur_event_data['id'])

                home_stats = process_team_event_stats(cur_event_data['competitions'][0]['competitors'][0])
                away_stats = process_team_event_stats(cur_event_data['competitions'][0]['competitors'][1])

                # Get moneyline odds for each team
                if 'odds' in cur_event_data['competitions'][0]:
                    odds_data = requests.get(cur_event_data['competitions'][0]['odds']['$ref']).json()
                    home_odds = odds_data['items'][0]['homeTeamOdds']['moneyLine']
                    away_odds = odds_data['items'][0]['awayTeamOdds']['moneyLine']
                else:
                    home_odds = 0
                    away_odds = 0

                # Insert each team's stats for this event into event_stats table
                conn.execute("INSERT INTO event_stats (team_id, event_id, opponent_id, is_home, is_winner, \
                            moneyline_odds, runs_scored, runs_allowed, ops, sp_era, win_pct, home_win_pct, \
                            away_win_pct, prev_event, next_event) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                             (home_stats['id'], event_id, away_stats['id'], home_stats['is_home'],
                              home_stats['is_winner'], home_odds, home_stats['runs_scored'], home_stats['runs_allowed'],
                              home_stats['ops'], home_stats['sp_era'], home_stats['win_pct'],
                              home_stats['home_win_pct'], home_stats['away_win_pct'], home_stats['prev_event'],
                              home_stats['next_event']))

                conn.execute("INSERT INTO event_stats (team_id, event_id, opponent_id, is_home, is_winner, \
                            moneyline_odds, runs_scored, runs_allowed, ops, sp_era, win_pct, home_win_pct, \
                            away_win_pct, prev_event, next_event) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                             (away_stats['id'], event_id, home_stats['id'], away_stats['is_home'],
                              away_stats['is_winner'], away_odds, away_stats['runs_scored'], away_stats['runs_allowed'],
                              away_stats['ops'], away_stats['sp_era'], away_stats['win_pct'],
                              away_stats['home_win_pct'], away_stats['away_win_pct'], away_stats['prev_event'],
                              away_stats['next_event']))

                event_count += 1
                print("Event", event_count, "inserted")

    conn.commit()

"""
conn = sqlite3.connect("mlb2022")


event_count = 0
for page_index in range(1, 101):
    cur_page_data = requests.get(URL + 'events?page=' + str(page_index)).json()
    cur_event_links = cur_page_data['items']

    # Iterate through each event in current page
    for event_link in cur_event_links:
        cur_event_data = requests.get(event_link['$ref']).json()

        # Make sure game did not have a winner
        if 'winner' not in cur_event_data['competitions'][0]['competitors'][0]:
            # Get team_ids and event_id and insert into event_stats table
            home_team = int(cur_event_data['competitions'][0]['competitors'][0]['id'])
            away_team = int(cur_event_data['competitions'][0]['competitors'][1]['id'])
            event_id = int(cur_event_data['id'])
            home_prev_event_str = cur_event_data['competitions'][0]['competitors'][0]['previousCompetition']['$ref']
            home_prev_event = int(home_prev_event_str[:-18].split('/')[-1])
            away_prev_event_str = cur_event_data['competitions'][0]['competitors'][1]['previousCompetition']['$ref']
            away_prev_event = int(away_prev_event_str[:-18].split('/')[-1])

            home_next_event_str = cur_event_data['competitions'][0]['competitors'][0]['nextCompetition']['$ref']
            home_next_event = int(home_next_event_str[:-18].split('/')[-1])
            away_next_event_str = cur_event_data['competitions'][0]['competitors'][1]['nextCompetition']['$ref']
            away_next_event = int(away_next_event_str[:-18].split('/')[-1])

            conn.execute("INSERT INTO event_stats (team_id, event_id, opponent_id, is_home, is_winner, \
                            moneyline_odds, runs_scored, runs_allowed, ops, sp_era, win_pct, home_win_pct, \
                            away_win_pct, prev_event, next_event) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                             (home_team, event_id, away_team, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, home_prev_event, home_next_event))

            conn.execute("INSERT INTO event_stats (team_id, event_id, opponent_id, is_home, is_winner, \
                                        moneyline_odds, runs_scored, runs_allowed, ops, sp_era, win_pct, home_win_pct, \
                                        away_win_pct, prev_event, next_event) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                           (away_team, event_id, home_team, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, away_prev_event,
                            away_next_event))

            event_count += 1
            print("event", event_count, "inserted")

print(event_count)
conn.commit()
conn.close()
"""
