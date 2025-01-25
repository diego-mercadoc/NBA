import pandas as pd
import numpy as np
from nba_predictor import NBAPredictor
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def generate_correlated_features(n_samples=100, base_strength_std=0.1):
    """Generate correlated features that meet validation requirements"""
    # Generate base team strengths (will influence all other metrics)
    team_strength = np.random.normal(1.0, base_strength_std, n_samples)
    
    # Generate win rates strongly correlated with team strength (reduced noise, increased coefficient)
    noise = np.random.normal(0, 0.02, n_samples)  # Reduced noise from 0.05 to 0.02
    win_rate = 0.5 + 0.4 * (team_strength - 1.0) + noise  # Increased coefficient from 0.3 to 0.4
    win_rate = np.clip(win_rate, 0.2, 0.8)  # Realistic bounds
    
    # Generate points based on team strength
    base_points = 110
    points = base_points + 15 * (team_strength - 1.0) + np.random.normal(0, 3, n_samples)
    
    # Generate offensive rating correlated with points
    noise = np.random.normal(0, 2, n_samples)
    offensive_rating = points + noise
    
    # Generate recent form and streak with correlation
    form = 5 * (team_strength - 1.0) + np.random.normal(0, 1, n_samples)
    streak = np.sign(form) * np.ceil(np.abs(form))
    streak = np.clip(streak, -5, 5)  # Realistic streak bounds
    
    return {
        'team_strength': team_strength,
        'win_rate': win_rate,
        'points': points,
        'offensive_rating': offensive_rating,
        'form': form,
        'streak': streak
    }

def main():
    """Generate validation dataset for dry-run testing"""
    try:
        predictor = NBAPredictor()
        n_samples = 100
        
        # Generate home team features
        home_features = generate_correlated_features(n_samples)
        away_features = generate_correlated_features(n_samples)
        
        # Create validation dataset
        validation_data = pd.DataFrame({
            'Date': pd.date_range(end=pd.Timestamp.now(), periods=n_samples),
            'Home_Team': np.random.choice(['BOS', 'LAL', 'GSW', 'MIA', 'PHX', 'NYK', 'CHI', 'DAL', 'DEN', 'MIL'], n_samples),
            'Away_Team': np.random.choice(['BOS', 'LAL', 'GSW', 'MIA', 'PHX', 'NYK', 'CHI', 'DAL', 'DEN', 'MIL'], n_samples),
            'Home_Points': home_features['points'],
            'Away_Points': away_features['points'],
            'Home_Team_Strength': home_features['team_strength'],
            'Away_Team_Strength': away_features['team_strength'],
            'Home_Win_Rate': home_features['win_rate'],
            'Away_Win_Rate': away_features['win_rate'],
            'Home_Recent_Form': home_features['form'],
            'Away_Recent_Form': away_features['form'],
            'Home_Streak': home_features['streak'],
            'Away_Streak': away_features['streak'],
            'Home_ORtg': home_features['offensive_rating'],
            'Away_ORtg': away_features['offensive_rating']
        })
        
        # Ensure teams don't play against themselves
        mask = validation_data['Home_Team'] == validation_data['Away_Team']
        validation_data.loc[mask, 'Away_Team'] = validation_data.loc[mask, 'Away_Team'].apply(
            lambda x: np.random.choice([t for t in ['BOS', 'LAL', 'GSW', 'MIA', 'PHX', 'NYK', 'CHI', 'DAL', 'DEN', 'MIL'] if t != x])
        )
        
        # Add required rolling statistics
        validation_data['Home_Points_Roll15'] = validation_data['Home_Points'].rolling(window=15, min_periods=1).mean()
        validation_data['Away_Points_Roll15'] = validation_data['Away_Points'].rolling(window=15, min_periods=1).mean()
        validation_data['Home_Points_Allowed_Roll15'] = validation_data['Away_Points'].rolling(window=15, min_periods=1).mean()
        validation_data['Away_Points_Allowed_Roll15'] = validation_data['Home_Points'].rolling(window=15, min_periods=1).mean()
        
        # Add required features from .cursorrules
        validation_data['3pt_volatility'] = np.random.uniform(-1, 1, n_samples)
        validation_data['pace_adjusted_offense'] = np.random.uniform(50, 150, n_samples)
        validation_data['pace_adjusted_defense'] = np.random.uniform(50, 150, n_samples)
        
        # Add required game stats
        validation_data['Home_FGA'] = validation_data['Home_Points'] * 0.85
        validation_data['Away_FGA'] = validation_data['Away_Points'] * 0.85
        validation_data['Home_FGM'] = validation_data['Home_Points'] * 0.4
        validation_data['Away_FGM'] = validation_data['Away_Points'] * 0.4
        validation_data['Home_FTA'] = validation_data['Home_Points'] * 0.2
        validation_data['Away_FTA'] = validation_data['Away_Points'] * 0.2
        validation_data['Home_OREB'] = np.random.randint(8, 15, n_samples)
        validation_data['Away_OREB'] = np.random.randint(8, 15, n_samples)
        validation_data['Home_DREB'] = np.random.randint(30, 40, n_samples)
        validation_data['Away_DREB'] = np.random.randint(30, 40, n_samples)
        validation_data['Home_TOV'] = np.random.randint(10, 18, n_samples)
        validation_data['Away_TOV'] = np.random.randint(10, 18, n_samples)
        validation_data['Home_3PA'] = validation_data['Home_FGA'] * 0.4
        validation_data['Away_3PA'] = validation_data['Away_FGA'] * 0.4
        validation_data['Home_3PM'] = validation_data['Home_3PA'] * 0.36
        validation_data['Away_3PM'] = validation_data['Away_3PA'] * 0.36
        validation_data['Home_DRtg'] = 200 - validation_data['Home_ORtg']
        validation_data['Away_DRtg'] = 200 - validation_data['Away_ORtg']
        validation_data['Home_Pace'] = np.random.uniform(95, 105, n_samples)
        validation_data['Away_Pace'] = np.random.uniform(95, 105, n_samples)
        
        # Save validation dataset
        validation_data.to_csv('validation_set.csv', index=False)
        logging.info(f"Validation dataset created with {len(validation_data)} samples")
        
        # Verify correlations
        correlations = {
            'team_strength_win_rate': np.corrcoef(validation_data['Home_Team_Strength'], validation_data['Home_Win_Rate'])[0, 1],
            'points_scored_ortg': np.corrcoef(validation_data['Home_Points'], validation_data['Home_ORtg'])[0, 1],
            'recent_form_streak': np.corrcoef(validation_data['Home_Recent_Form'], validation_data['Home_Streak'])[0, 1]
        }
        
        logging.info("\nCorrelation Verification:")
        for metric, corr in correlations.items():
            logging.info(f"{metric}: {corr:.3f}")
        
        # Generate validation metrics using the generated dataset
        metrics = predictor.validate_predictions(validation_data=validation_data)
        
        # Log validation metrics
        logging.info("\nValidation Metrics:")
        for metric, value in metrics.items():
            if isinstance(value, dict):
                logging.info(f"\n{metric}:")
                for submetric, subvalue in value.items():
                    logging.info(f"  {submetric}: {subvalue:.3f}")
            else:
                logging.info(f"{metric}: {value:.3f}")
        
    except Exception as e:
        logging.error(f"Error generating validation dataset: {str(e)}")
        raise

if __name__ == "__main__":
    main() 