import pytest
import pandas as pd
import numpy as np
from nba_predictor import NBAPredictor
import logging
import psutil
import os

# Set up memory monitoring
def check_memory_usage():
    """Check if memory usage is within limits (4GB)"""
    process = psutil.Process(os.getpid())
    memory_gb = process.memory_info().rss / 1024 / 1024 / 1024
    if memory_gb > 4:
        raise MemoryError(f"Memory usage ({memory_gb:.2f}GB) exceeds 4GB limit")

@pytest.fixture(autouse=True)
def monitor_memory():
    """Monitor memory usage before and after each test"""
    check_memory_usage()
    yield
    check_memory_usage()

@pytest.fixture
def sample_data():
    """Generate sample data for testing"""
    n_samples = 1000
    np.random.seed(42)
    
    # Create team strength ratings (will influence all other metrics)
    teams = ['BOS', 'LAL', 'GSW', 'MIA', 'PHX', 'NYK', 'CHI', 'DAL', 'DEN', 'MIL']
    team_ratings = {team: np.random.normal(100, 10) for team in teams}
    
    # Base data
    data = {
        'Date': pd.date_range(start='2024-01-01', periods=n_samples),
        'Home_Team': np.random.choice(teams, n_samples),
        'Away_Team': np.random.choice(teams, n_samples),
        'Season': 2024
    }
    
    df = pd.DataFrame(data)
    
    # Ensure teams don't play against themselves
    mask = df['Home_Team'] == df['Away_Team']
    df.loc[mask, 'Away_Team'] = df.loc[mask, 'Away_Team'].apply(
        lambda x: np.random.choice([t for t in teams if t != x])
    )
    
    # Calculate base team strengths
    df['Home_Rating'] = df['Home_Team'].map(team_ratings)
    df['Away_Rating'] = df['Away_Team'].map(team_ratings)
    
    # Add home court advantage
    home_advantage = 3.5
    df['Home_Rating'] += home_advantage
    
    # Generate correlated points based on team ratings
    base_points = 110
    rating_to_points_factor = 0.5
    point_variance = 8
    
    df['Home_Points'] = (
        base_points + 
        (df['Home_Rating'] - 100) * rating_to_points_factor + 
        np.random.normal(0, point_variance, n_samples)
    ).round()
    
    df['Away_Points'] = (
        base_points + 
        (df['Away_Rating'] - 100) * rating_to_points_factor + 
        np.random.normal(0, point_variance, n_samples)
    ).round()
    
    # Calculate team stats based on their ratings and actual points
    for team_type in ['Home', 'Away']:
        # Points scored and allowed (with proper rolling calculations)
        df[f'{team_type}_Points_Scored'] = df[f'{team_type}_Points']
        df[f'{team_type}_Points_Allowed'] = df[f'{"Away" if team_type=="Home" else "Home"}_Points']
        
        df = df.sort_values('Date')
        df[f'{team_type}_Points_Scored_Roll5'] = df.groupby(f'{team_type}_Team')[f'{team_type}_Points_Scored'].transform(
            lambda x: x.rolling(5, min_periods=1).mean()
        )
        df[f'{team_type}_Points_Allowed_Roll5'] = df.groupby(f'{team_type}_Team')[f'{team_type}_Points_Allowed'].transform(
            lambda x: x.rolling(5, min_periods=1).mean()
        )
        df[f'{team_type}_Point_Diff_Roll5'] = df[f'{team_type}_Points_Scored_Roll5'] - df[f'{team_type}_Points_Allowed_Roll5']
        
        # Form and streaks (based on actual game outcomes)
        df[f'{team_type}_Win'] = df[f'{team_type}_Points'] > df[f'{"Away" if team_type=="Home" else "Home"}_Points']
        df[f'{team_type}_Streak'] = df.groupby(f'{team_type}_Team')[f'{team_type}_Win'].transform(
            lambda x: x.rolling(5, min_periods=1).sum() - (1 - x).rolling(5, min_periods=1).sum()
        )
        
        df[f'{team_type}_Form'] = df.groupby(f'{team_type}_Team')[f'{team_type}_Win'].transform(
            lambda x: x.rolling(10, min_periods=1).mean()
        )
        
        df[f'{team_type}_Rest_Days'] = np.random.randint(1, 5, n_samples)
        df[f'{team_type}_Win_Rate'] = df.groupby(f'{team_type}_Team')[f'{team_type}_Win'].transform('mean')
        
        # Shooting efficiency (correlated with team rating)
        rating_factor = (df[f'{team_type}_Rating'] - 100) / 20
        df[f'{team_type}_eFG_Pct'] = (0.52 + rating_factor * 0.02 + np.random.normal(0, 0.02, n_samples)).clip(0.45, 0.60)
        df[f'{team_type}_3P_Pct'] = (0.36 + rating_factor * 0.015 + np.random.normal(0, 0.015, n_samples)).clip(0.30, 0.45)
        df[f'{team_type}_3pt_variance'] = np.random.uniform(0.02, 0.08, n_samples)
        
        # Advanced metrics (derived from actual performance)
        possessions = df[f'{team_type}_Points'].mean() * 100 / 110  # Estimate possessions from points
        df[f'{team_type}_ORtg'] = (df[f'{team_type}_Points'] / possessions * 100).clip(95, 125)
        df[f'{team_type}_DRtg'] = (df[f'{"Away" if team_type=="Home" else "Home"}_Points'] / possessions * 100).clip(95, 125)
        df[f'{team_type}_Pace'] = (possessions + np.random.normal(0, 2, n_samples)).clip(90, 110)
    
    # Calculate derived features
    df['Win_Rate_Diff'] = df['Home_Win_Rate'] - df['Away_Win_Rate']
    df['Point_Diff_Ratio'] = df['Home_Point_Diff_Roll5'] / (df['Away_Point_Diff_Roll5'].abs() + 1e-6)
    df['3pt_volatility'] = (df['Home_3pt_variance'] - df['Away_3pt_variance']).clip(-1, 1)
    
    # Calculate pace-adjusted metrics within expected ranges
    df['pace_adjusted_offense'] = (df['Home_ORtg'] * df['Home_Pace'] / 100).clip(50, 150)
    df['pace_adjusted_defense'] = (df['Away_DRtg'] * df['Away_Pace'] / 100).clip(50, 150)
    
    return df.sort_values('Date').reset_index(drop=True)

def test_feature_preparation(sample_data):
    """Test feature preparation with validation"""
    predictor = NBAPredictor()
    
    # Prepare features
    X = predictor.prepare_features(sample_data)
    
    # Validate feature presence
    for feature in predictor.feature_columns:
        assert feature in X.columns, f"Missing feature: {feature}"
    
    # Validate no NaN values
    assert not X.isna().any().any(), "Features contain NaN values"
    
    # Validate value ranges
    assert X['3pt_volatility'].between(-1, 1).all(), "Invalid 3pt_volatility values"
    assert X['pace_adjusted_offense'].between(50, 150).all(), "Invalid pace_adjusted_offense values"
    assert X['pace_adjusted_defense'].between(50, 150).all(), "Invalid pace_adjusted_defense values"

def test_model_training(sample_data):
    """Test model training with validation"""
    predictor = NBAPredictor()
    
    try:
        # Convert DataFrame to numpy array before training
        X = predictor.prepare_features(sample_data)
        y = (sample_data['Home_Points'] > sample_data['Away_Points']).astype(int)
        
        # Convert to numpy arrays
        X_np = X.to_numpy()
        y_np = y.to_numpy()
        
        metrics = predictor.train_models(sample_data, test_size=0.2)
        
        # Validate metric presence
        required_metrics = [
            'moneyline_accuracy', 'moneyline_brier', 'moneyline_log_loss',
            'spread_rmse', 'totals_rmse'
        ]
        for metric in required_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
        
        # Validate metric ranges
        assert 0 <= metrics['moneyline_accuracy'] <= 1, "Invalid accuracy range"
        assert 0 <= metrics['moneyline_brier'] <= 1, "Invalid Brier score range"
        assert metrics['moneyline_log_loss'] >= 0, "Invalid log loss range"
        assert metrics['spread_rmse'] >= 0, "Invalid spread RMSE range"
        assert metrics['totals_rmse'] >= 0, "Invalid totals RMSE range"
        
        # Validate minimum performance
        assert metrics['moneyline_accuracy'] >= 0.78, "Accuracy below minimum threshold"
        assert metrics['moneyline_brier'] <= 0.20, "Brier score above maximum threshold"
        assert metrics['moneyline_log_loss'] <= 0.55, "Log loss above maximum threshold"
        assert metrics['spread_rmse'] <= 16.0, "Spread RMSE above maximum threshold"
        assert metrics['totals_rmse'] <= 16.0, "Totals RMSE above maximum threshold"
        
    except Exception as e:
        logging.error(f"Model training failed: {str(e)}")
        raise

def test_prediction_stability(sample_data):
    """Test prediction stability across multiple runs"""
    predictor = NBAPredictor()
    
    # Train models with numpy arrays
    X = predictor.prepare_features(sample_data)
    y = (sample_data['Home_Points'] > sample_data['Away_Points']).astype(int)
    
    # Convert to numpy arrays
    X_np = X.to_numpy()
    y_np = y.to_numpy()
    
    # Train models
    predictor.train_models(sample_data, test_size=0.2)
    
    # Generate predictions multiple times
    predictions = []
    for _ in range(3):
        pred = predictor.predict_games(sample_data)
        predictions.append(pred['Moneyline_Pick'].values)
    
    # Validate prediction stability
    for i in range(1, len(predictions)):
        agreement = np.mean(predictions[0] == predictions[i])
        assert agreement >= 0.95, f"Predictions not stable: {agreement:.2f} agreement" 