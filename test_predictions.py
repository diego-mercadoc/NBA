import pandas as pd
from nba_predictor import NBAPredictor
from datetime import datetime, timedelta

def test_injury_tracking():
    """Test just the injury tracking functionality"""
    predictor = NBAPredictor()
    
    # Force an injury data update
    predictor.injury_tracker.update_injuries()
    
    # Test teams
    test_teams = [
        'Denver Nuggets',
        'Los Angeles Lakers',
        'Boston Celtics',
        'Phoenix Suns'
    ]
    
    print("\nInjury Reports by Team:")
    print("=" * 50)
    
    for team in test_teams:
        print(f"\n{team} Injury Report:")
        print("-" * 30)
        
        # Get injury impact
        impact = predictor.injury_tracker.get_team_injury_impact(team)
        print(f"Team Injury Impact Score: {impact:.3f}")
        
        # Get detailed report
        report = predictor.injury_tracker.get_injury_report(team)
        if not report.empty:
            for _, injury in report.iterrows():
                print(f"\nPlayer: {injury['player']}")
                print(f"Status: {injury['status']}")
                print(f"Details: {injury['details']}")
                print(f"Impact Value: {injury['impact']:.3f}")
                print(f"Source: {injury['source']}")
                print(f"Last Update: {injury['last_update']}")
        else:
            print("No injuries reported")
        
        print("-" * 30)

def create_sample_game_data():
    # Create sample game data with all required features
    data = {
        'Date': [pd.Timestamp('2025-01-19')],
        'Home_Team': ['Denver Nuggets'],
        'Away_Team': ['Los Angeles Lakers'],
        'Season': [2025],
        'Home_Points': [None],  # Future game
        'Away_Points': [None],  # Future game
        'Home_Win_Pct': [0.650],
        'Away_Win_Pct': [0.600],
        'Home_Point_Diff': [5.2],
        'Away_Point_Diff': [3.8],
        'Home_Rest_Days': [2],
        'Away_Rest_Days': [1],
        'Home_Games_In_7': [3],
        'Away_Games_In_7': [4],
        'Home_Win_Streak': [3],
        'Away_Win_Streak': [2],
        'Home_Streak': [3],  # Same as Home_Win_Streak for compatibility
        'Away_Streak': [2],  # Same as Away_Win_Streak for compatibility
        'Home_Points_Roll3': [115.5],
        'Away_Points_Roll3': [112.3],
        'Home_Points_Roll10': [114.2],
        'Away_Points_Roll10': [110.8],
        'Home_Points_Against_Roll3': [105.5],
        'Away_Points_Against_Roll3': [108.2],
        'Home_Points_Against_Roll10': [106.8],
        'Away_Points_Against_Roll10': [109.5],
        'Month': ['January'],
        'Is_Future': [True],
        'Is_Scheduled': [True],
        'Is_Played': [False],
        'Home_Off_Rating': [115.2],
        'Away_Off_Rating': [112.8],
        'Home_Def_Rating': [108.4],
        'Away_Def_Rating': [110.6],
        'Home_Net_Rating': [6.8],
        'Away_Net_Rating': [2.2],
        'Home_Pace': [98.5],
        'Away_Pace': [99.2]
    }
    return pd.DataFrame(data)

def main():
    print("\nTesting Injury Tracking System...")
    predictor = NBAPredictor()
    
    # Test injury tracking
    test_injury_tracking()
    
    print("\nTesting Full Prediction System...")
    try:
        # Create sample game data
        games_df = create_sample_game_data()
        
        # Load or train models
        predictor.load_models()
        
        # Get predictions
        predictions = predictor.predict_games(games_df)
        
        # Print predictions
        print("\nPredictions for upcoming games:")
        print("=" * 50)
        for _, row in predictions.iterrows():
            print(f"\n{row['Away_Team']} @ {row['Home_Team']}")
            print(f"Win Probability: {row['Win_Probability']:.3f}")
            print(f"Predicted Spread: {row['Predicted_Spread']:.1f}")
            print(f"Home Team Injury Impact: {row['Home_Injury_Impact']:.3f}")
            print(f"Away Team Injury Impact: {row['Away_Injury_Impact']:.3f}")
            if 'Value_Rating' in row:
                print(f"Value Rating: {row['Value_Rating']:.2f}")
            
    except NotImplementedError as e:
        print(f"Error: {str(e)}")
        print("Please train the models with historical data before making predictions.")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        print("Please check the error message and try again.")

if __name__ == "__main__":
    main() 