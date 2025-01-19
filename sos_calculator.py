import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

class SOSCalculator:
    """
    Calculates Strength of Schedule (SoS) factors for NBA teams.
    
    Features:
    - Rolling SoS calculation based on opponent quality
    - Weighted opponent ratings
    - Home/away split consideration
    - Recent games weighted more heavily
    """
    
    def __init__(self):
        self.sos_factors = {}
        self.opponent_ratings = {}
        self.weights = {
            'opponent_win_rate': 0.3,
            'opponent_point_diff': 0.3,
            'opponent_net_rating': 0.4
        }
        
        # Exponential decay factor for game recency
        self.decay_factor = 0.95
    
    def calculate_opponent_ratings(self, games_df):
        """Calculate base ratings for each team."""
        current_date = datetime.now()
        
        # Group by team and calculate basic metrics
        team_stats = {}
        
        for team_type in ['Home', 'Away']:
            team_games = games_df[games_df[f'{team_type}_Team'].notna()]
            
            for team in team_games[f'{team_type}_Team'].unique():
                if team not in team_stats:
                    team_stats[team] = {
                        'games_played': 0,
                        'wins': 0,
                        'point_diff_sum': 0,
                        'net_rating_sum': 0
                    }
                
                team_matches = team_games[team_games[f'{team_type}_Team'] == team]
                
                for _, game in team_matches.iterrows():
                    # Skip future games
                    if pd.to_datetime(game['Date']) > current_date:
                        continue
                    
                    team_stats[team]['games_played'] += 1
                    
                    # Calculate win
                    if team_type == 'Home':
                        won = game['Home_Points'] > game['Away_Points']
                        point_diff = game['Home_Points'] - game['Away_Points']
                    else:
                        won = game['Away_Points'] > game['Home_Points']
                        point_diff = game['Away_Points'] - game['Home_Points']
                    
                    if won:
                        team_stats[team]['wins'] += 1
                    
                    # Update point differential
                    team_stats[team]['point_diff_sum'] += point_diff
                    
                    # Update net rating if available
                    if f'{team_type}_Net_Rating_Roll5' in game:
                        team_stats[team]['net_rating_sum'] += game[f'{team_type}_Net_Rating_Roll5']
        
        # Calculate final ratings
        for team, stats in team_stats.items():
            if stats['games_played'] > 0:
                win_rate = stats['wins'] / stats['games_played']
                avg_point_diff = stats['point_diff_sum'] / stats['games_played']
                avg_net_rating = stats['net_rating_sum'] / stats['games_played']
                
                self.opponent_ratings[team] = {
                    'win_rate': win_rate,
                    'point_diff': avg_point_diff,
                    'net_rating': avg_net_rating
                }
    
    def calculate_sos(self, games_df):
        """Calculate SoS for each team based on opponent quality."""
        self.calculate_opponent_ratings(games_df)
        current_date = datetime.now()
        
        # Initialize SoS tracking
        sos_tracking = {}
        
        # Calculate SoS for each team
        for team_type in ['Home', 'Away']:
            team_games = games_df[games_df[f'{team_type}_Team'].notna()]
            
            for team in team_games[f'{team_type}_Team'].unique():
                if team not in sos_tracking:
                    sos_tracking[team] = []
                
                team_matches = team_games[team_games[f'{team_type}_Team'] == team]
                
                for _, game in team_matches.iterrows():
                    game_date = pd.to_datetime(game['Date'])
                    
                    # Skip future games
                    if game_date > current_date:
                        continue
                    
                    # Get opponent
                    opponent = game['Away_Team'] if team_type == 'Home' else game['Home_Team']
                    
                    if opponent in self.opponent_ratings:
                        opp_rating = self.opponent_ratings[opponent]
                        
                        # Calculate game SoS
                        game_sos = (
                            self.weights['opponent_win_rate'] * opp_rating['win_rate'] +
                            self.weights['opponent_point_diff'] * (opp_rating['point_diff'] / 10) +  # Normalize point diff
                            self.weights['opponent_net_rating'] * (opp_rating['net_rating'] / 10)  # Normalize net rating
                        )
                        
                        # Apply home/away adjustment
                        if team_type == 'Away':
                            game_sos *= 1.1  # 10% boost for away games
                        
                        # Calculate days since game
                        days_since_game = (current_date - game_date).days
                        
                        # Apply exponential decay based on recency
                        recency_weight = self.decay_factor ** (days_since_game / 30)  # Decay per month
                        
                        sos_tracking[team].append({
                            'date': game_date,
                            'sos': game_sos,
                            'weight': recency_weight
                        })
        
        # Calculate final weighted SoS for each team
        for team, games in sos_tracking.items():
            if games:
                total_weight = sum(game['weight'] for game in games)
                weighted_sos = sum(game['sos'] * game['weight'] for game in games) / total_weight
                self.sos_factors[team] = weighted_sos
            else:
                self.sos_factors[team] = 1.0  # Neutral SoS if no games
        
        # Normalize SoS factors around 1.0
        mean_sos = np.mean(list(self.sos_factors.values()))
        for team in self.sos_factors:
            self.sos_factors[team] /= mean_sos
        
        return self.sos_factors
    
    def get_sos_factor(self, team):
        """Get the SoS factor for a specific team."""
        return self.sos_factors.get(team, 1.0)  # Return 1.0 if not found (neutral) 