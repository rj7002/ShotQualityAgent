from datetime import datetime
import requests
from data import calculate_tracking_features, fetch_all_player_locations, fetch_pbp_data, headers
import pandas as pd

def get_nba_season_f(subtract):
    today = datetime.now()
    year = today.year
    month = today.month

    if month >= 10:  # October–December
        start_year = year
        end_year = year + 1
    else:  # January–September
        start_year = year - 1
        end_year = year
    start_year = start_year-subtract
    end_year = end_year-subtract
    return f"{start_year}-{str(end_year)[-2:]}"

def get_player_id_f(name):
    url = f"https://api.shotquality.com/players/?player_name={name}"
    response = requests.get(url, headers=headers)
    data = response.json()
    df = pd.DataFrame(data['players'])
    
    # Keep all columns initially to check for completeness
    if not df.empty:
        # Score each player by data completeness (prefer players with more information)
        # Players with height, weight, and position are more likely to be the main entry
        df['_completeness_score'] = (
            df['height_inches'].notna().astype(int) + 
            df['weight_pounds'].notna().astype(int) + 
            df['position'].notna().astype(int)
        )
        
        # Sort by completeness (descending) and creation date (ascending - older entries first)
        df['created_at'] = pd.to_datetime(df['created_at'])
        df = df.sort_values(by=['_completeness_score', 'created_at'], ascending=[False, True])
        
        # Reset index so the best match is always at index 0
        df = df.reset_index(drop=True)
        
        # Drop the helper column
        df = df.drop(columns=['_completeness_score'])
    
    # Now drop the extra columns as before
    df = df.drop(columns=['created_at','updated_at','first_name','last_name','height_inches','weight_pounds','position'], errors="ignore")
    return df

def get_team_id_f(name):
    url = f"https://api.shotquality.com/teams/?team_name={name}"
    response = requests.get(url, headers=headers)
    data = response.json()
    df = pd.DataFrame(data['teams'])
    df = df.drop(columns=['created_at','updated_at','gender'], errors="ignore")
    return df

def get_competition_seasons_f(name, year):
    url = f"https://api.shotquality.com/competition-seasons/?competition_name={name}&season_start_year={year}"
    response = requests.get(url, headers=headers)
    data = response.json()
    df = pd.DataFrame(data['competition_seasons'])
    dropcols = ['created_at', 'updated_at', 'competition_id',
        'regulation_periods',
       'period_length_minutes', 'overtime_length_minutes',
       'elimination_fouls']
    df = df.drop(columns=dropcols, errors="ignore")
    return df

teams = {
    89631: 'Atlanta Hawks',
    92036: 'Boston Celtics',
    92265: 'Brooklyn Nets',
    93897: 'Charlotte Hornets',
    93962: 'Chicago Bulls',
    94296: 'Cleveland Cavaliers',
    95600: 'Dallas Mavericks',
    95919: 'Denver Nuggets',
    95959: 'Detroit Pistons',
    98838: 'Golden State Warriors',
    103534: 'Houston Rockets',
    104287: 'Indiana Pacers',
    107248: 'Los Angeles Clippers',
    107256: 'Los Angeles Lakers',
    108536: 'Memphis Grizzlies',
    108701: 'Miami Heat',
    109034: 'Milwaukee Bucks',
    109159: 'Minnesota Timberwolves',
    110366: 'New Orleans Pelicans',
    110398: 'New York Knicks',
    111557: 'Oklahoma City Thunder',
    111836: 'Orlando Magic',
    112766: 'Philadelphia 76ers',
    112853: 'Phoenix Suns',
    113174: 'Portland Trail Blazers',
    115371: 'Sacramento Kings',
    115613: 'San Antonio Spurs',
    119792: 'Toronto Raptors',
    121054: 'Utah Jazz',
    121997: 'Washington Wizards'
}

def get_games_f(compid, teamid):
    url = f"https://api.shotquality.com/games/{compid}?team_id={teamid}&limit=5000"
    response = requests.get(url, headers=headers)
    data = response.json()
    games = pd.DataFrame(data['games'])
    dropcols = ['created_at', 'updated_at', 
       'competition_id',  'home_score',
       'away_score', 'home_dynamic_sq_score', 'away_dynamic_sq_score',
       'home_initial_sq_score', 'away_initial_sq_score', 'is_neutral',
       'game_status', 'pbp_updated_at','game_descriptors']
    games = games.drop(columns=dropcols, errors="ignore")
    games['away_team'] = games['away_team_id'].map(teams)
    games['home_team'] = games['home_team_id'].map(teams)
    games['date'] = pd.to_datetime(games['game_datetime_utc']).dt.strftime('%m-%d-%Y')
    return games

def get_full_tracking_data_f(gameid, playerid):
    try:
        # Fetch data
        pbpdata = fetch_pbp_data(gameid)
        locsdata = fetch_all_player_locations(gameid)
        
        # Validate pbpdata exists
        if pbpdata is None or pbpdata.empty:
            print(f"Warning: No play-by-play data found for game {gameid}. Cannot proceed.")
            return pd.DataFrame()
        
        # Validate 'play_id'
        if 'play_id' not in pbpdata.columns:
            print(f"Warning: Play-by-play data missing 'play_id' for game {gameid}. Proceeding with limited data.")
        
        # Filter for player
        pbpdata = pbpdata[pbpdata['player_id'] == playerid]
        if pbpdata.empty:
            print(f"Warning: No shot data found for player {playerid} in game {gameid}.")
            return pd.DataFrame()
        
        # Keep only 2pt and 3pt shots
        pbpdata = pbpdata[pbpdata['action_type'].isin(['2pt', '3pt'])]
        if pbpdata.empty:
            print(f"Warning: Player {playerid} had no 2pt or 3pt shots in game {gameid}.")
            return pd.DataFrame()
        
        final_df = pbpdata.copy()
        
        # Calculate tracking features if location data is available
        tracking_features_df = None
        if locsdata is not None and not locsdata.empty:
            try:
                tracking_features_df = calculate_tracking_features(pbpdata, locsdata)
                if tracking_features_df is not None and not tracking_features_df.empty and 'play_id' in tracking_features_df.columns:
                    final_df = pbpdata.merge(tracking_features_df, on='play_id', how='left')
                else:
                    print(f"Note: Tracking features missing or incomplete for game {gameid}. Proceeding with PBP data only.")
            except Exception as e:
                print(f"Note: Failed to calculate tracking features: {str(e)}. Proceeding with PBP data only.")
        else:
            print(f"Note: No player location data available for game {gameid}. Proceeding with PBP data only.")
        
        # Drop unnecessary columns
        dropcols = ['created_at', 'updated_at', 'parent_play_id', 'play_id',
                    'feature_store_top_5', 'shot_taking',
                    'shooting_bench_direction', 'initial_updated_at']
        final_df = final_df.drop(columns=dropcols, errors="ignore")
        
        # Filter out free throws
        final_df = final_df[final_df['action_type'] != 'freethrow']
        
        if final_df.empty:
            print(f"Warning: No valid shot data after processing for game {gameid}.")
            return pd.DataFrame()
        
        return final_df
    
    except KeyError as e:
        print(f"Error: Missing required column in data: {str(e)}. Returning PBP data if available.")
        return pbpdata if pbpdata is not None else pd.DataFrame()
    except Exception as e:
        print(f"Error retrieving tracking data: {str(e)}. Returning PBP data if available.")
        return pbpdata if pbpdata is not None else pd.DataFrame()
