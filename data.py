import requests 
import pandas as pd
import numpy as np
import dotenv
import os
dotenv.load_dotenv()
headers = {
    "accept": "application/json",
    "authorization": f"Bearer {os.getenv('SHOTQUALITY_API_KEY')}"
}


def make_get_request(url, params=None):  
    """  
    Helper function to make a GET request to the API.
    :param url: Endpoint URL.
    :param token: API token for authorization.
    :param params: Dictionary of query parameters.
    :return: Parsed JSON response or error message.
    """

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()  # Raise an error for bad responses (4xx, 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None
    
def fetch_all_player_locations(game_id):
    base_url = "https://api.shotquality.com"
    endpoint = "/player-locations"
    all_rows = []
    offset = 0
    limit = 1000

    while True:
        url = f"{base_url}{endpoint}/{game_id}"
        params = {"limit": limit, "offset": offset}
        response = make_get_request(url, params=params)
        
        # If response is None (error occurred), return what we have collected so far
        if not response:
            break
        
        # If no player_locations key or empty array, we've reached the end
        if 'player_locations' not in response or not response['player_locations']:
            break

        all_rows.extend(response['player_locations'])
        offset += limit

    return pd.DataFrame(all_rows)

def fetch_pbp_data(game_id):
    base_url = "https://api.shotquality.com"
    endpoint = "/play-by-play"
    url = f"{base_url}{endpoint}/{game_id}"
    params = {"limit":5000}
    response = make_get_request(url,params=params)
    
    # Handle empty or malformed response
    if not response:
        return pd.DataFrame()
    
    if 'plays' not in response:
        return pd.DataFrame()
    
    if not response['plays']:
        return pd.DataFrame()
    
    return pd.DataFrame(response['plays'])

def calculate_tracking_features(pbp_df, locs_df):
    """
    Advanced geometric tracking features for LLM analysis.
    Focuses on angles, triangles, and spatial relationships between shooter, defenders, teammates, and basket.
    """
    
    import numpy as np
    
    # Handle empty dataframes
    if pbp_df is None or pbp_df.empty:
        return pd.DataFrame()
    
    if locs_df is None or locs_df.empty:
        return pd.DataFrame()
    
    # Validate required columns
    required_pbp_cols = ['play_id', 'shot_x', 'shot_y']
    if not all(col in pbp_df.columns for col in required_pbp_cols):
        return pd.DataFrame()

    BASKET_LEFT = (5.25, 25)
    BASKET_RIGHT = (88.75, 25)

    def euclidean_distance(x1, y1, x2, y2):
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def angle_between_three_points(ax, ay, bx, by, cx, cy):
        ba = np.array([ax - bx, ay - by])
        bc = np.array([cx - bx, cy - by])
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-10)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))
    
    def triangle_area(x1, y1, x2, y2, x3, y3):
        return abs((x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)) / 2.0)
    
    features = []

    for _, play in pbp_df.iterrows():
        play_id = play['play_id']
        shot_x = play['shot_x']
        shot_y = play['shot_y']
        
        play_locs = locs_df[locs_df['play_id'] == play_id]
        basket_x, basket_y = BASKET_LEFT if shot_x < 47 else BASKET_RIGHT
        
        shooter = play_locs[play_locs['player_location_type'] == 'shooter']
        defenders = play_locs[play_locs['player_location_type'] == 'defender']
        teammates = play_locs[play_locs['player_location_type'] == 'teammate']
        
        feat_dict = {
            'play_id': play_id,
            'shot_x': shot_x,
            'shot_y': shot_y,
        }
        
        # --- Shooter ---
        if len(shooter) > 0:
            sx, sy = shooter.iloc[0]['court_x'], shooter.iloc[0]['court_y']
        else:
            sx, sy = shot_x, shot_y
        
        feat_dict['shooter_distance_to_basket'] = euclidean_distance(sx, sy, basket_x, basket_y)
        feat_dict['shooter_angle_to_basket'] = np.degrees(np.arctan2(abs(sy - basket_y), abs(sx - basket_x) + 1e-10))
        
        # --- Defenders ---
        num_def = len(defenders)
        feat_dict['num_defenders_tracked'] = num_def
        
        if num_def > 0:
            def_coords = defenders[['court_x', 'court_y']].values
            def_distances = [euclidean_distance(sx, sy, x, y) for x, y in def_coords]
            def_angles_to_shooter = [angle_between_three_points(x, y, sx, sy, basket_x, basket_y) for x, y in def_coords]
            
            sorted_idx = np.argsort(def_distances)
            
            # Closest defenders
            feat_dict['closest_defender_distance'] = def_distances[sorted_idx[0]]
            feat_dict['second_closest_defender_distance'] = def_distances[sorted_idx[1]] if num_def > 1 else None
            feat_dict['closest_defender_angle_to_shooter'] = def_angles_to_shooter[sorted_idx[0]]
            feat_dict['second_closest_defender_angle_to_shooter'] = def_angles_to_shooter[sorted_idx[1]] if num_def > 1 else None
            
            if num_def > 1:
                feat_dict['avg_defender_dist_from_shooter'] = np.mean(def_distances)
            # Triangles and angles between shooter and defenders
            if num_def >= 2:
                idx1, idx2 = sorted_idx[0], sorted_idx[1]
                # Area of triangle formed by shooter + two closest defenders
                feat_dict['shooter_2def_triangle_area'] = triangle_area(
                    sx, sy,
                    def_coords[idx1, 0], def_coords[idx1, 1],
                    def_coords[idx2, 0], def_coords[idx2, 1]
                )
                # Angle at shooter formed by two closest defenders
                feat_dict['angle_at_shooter_between_two_closest_defenders'] = angle_between_three_points(
                    def_coords[idx1, 0], def_coords[idx1, 1],
                    sx, sy,
                    def_coords[idx2, 0], def_coords[idx2, 1]
                )
            
            # Relative angles between closest defenders
            if num_def > 2:
                idx3 = sorted_idx[2]
                feat_dict['angle_between_three_closest_defenders'] = angle_between_three_points(
                    def_coords[idx1, 0], def_coords[idx1, 1],
                    def_coords[idx2, 0], def_coords[idx2, 1],
                    def_coords[idx3, 0], def_coords[idx3, 1]
                )
        
        # --- Teammates ---
        num_tm = len(teammates)
        feat_dict['num_teammates_tracked'] = num_tm
        
        if num_tm > 0:
            tm_coords = teammates[['court_x', 'court_y']].values
            tm_distances = [euclidean_distance(sx, sy, x, y) for x, y in tm_coords]
            feat_dict['closest_teammate_distance'] = min(tm_distances)
            if num_tm > 1:
                feat_dict['avg_teammate_distance_from_shooter'] = np.mean(tm_distances)
            
            # Angles between shooter and two closest teammates
            if num_tm >= 2:
                sorted_tm_idx = np.argsort(tm_distances)
                idx1, idx2 = sorted_tm_idx[0], sorted_tm_idx[1]
                feat_dict['shooter_2tm_triangle_area'] = triangle_area(
                    sx, sy,
                    tm_coords[idx1, 0], tm_coords[idx1, 1],
                    tm_coords[idx2, 0], tm_coords[idx2, 1]
                )
                feat_dict['angle_at_shooter_between_two_closest_teammates'] = angle_between_three_points(
                    tm_coords[idx1, 0], tm_coords[idx1, 1],
                    sx, sy,
                    tm_coords[idx2, 0], tm_coords[idx2, 1]
                )
        
        # --- Combined geometric relationships ---
        feat_dict['total_players_tracked'] = num_def + num_tm + 1
        features.append(feat_dict)
    
    return pd.DataFrame(features)