import pytest
import pandas as pd
import numpy as np
from nba_predictor import NBAPredictor

@pytest.fixture
def sample_data():
    # Create sample data for testing
    np.random.seed(42)
    n_samples = 1000
    
    # Generate base data
    data = {
        'Date': pd.date_range(start='2024-01-01', periods=n_samples),
        'Home_Team': ['Team A'] * n_samples,
        'Away_Team': ['Team B'] * n_samples,
        'Home_Points': np.random.normal(110, 10, n_samples),
        'Away_Points': np.random.normal(105, 10, n_samples),
        'Home_Points_Scored_Roll5': np.random.normal(110, 5, n_samples),
        'Home_Points_Allowed_Roll5': np.random.normal(105, 5, n_samples),
        'Away_Points_Scored_Roll5': np.random.normal(105, 5, n_samples),
        'Away_Points_Allowed_Roll5': np.random.normal(110, 5, n_samples),
        'Home_Streak': np.random.randint(-5, 6, n_samples),
        'Away_Streak': np.random.randint(-5, 6, n_samples),
        'Home_Rest_Days': np.random.randint(1, 5, n_samples),
        'Away_Rest_Days': np.random.randint(1, 5, n_samples),
        'Home_Win_Rate': np.random.uniform(0.4, 0.6, n_samples),
        'Away_Win_Rate': np.random.uniform(0.4, 0.6, n_samples),
        'Home_eFG_Pct': np.random.uniform(0.45, 0.55, n_samples),
        'Away_eFG_Pct': np.random.uniform(0.45, 0.55, n_samples),
        'Home_3pt_variance': np.random.uniform(0.02, 0.08, n_samples),
        'Away_3pt_variance': np.random.uniform(0.02, 0.08, n_samples),
        'Home_ORtg': np.random.normal(110, 5, n_samples),
        'Away_ORtg': np.random.normal(110, 5, n_samples),
        'Home_DRtg': np.random.normal(110, 5, n_samples),
        'Away_DRtg': np.random.normal(110, 5, n_samples),
        'Home_Pace': np.random.normal(100, 3, n_samples),
        'Away_Pace': np.random.normal(100, 3, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Calculate point differentials
    df['Home_Point_Diff_Roll5'] = df['Home_Points_Scored_Roll5'] - df['Home_Points_Allowed_Roll5']
    df['Away_Point_Diff_Roll5'] = df['Away_Points_Scored_Roll5'] - df['Away_Points_Allowed_Roll5']
    
    return df

def test_feature_preparation(sample_data):
    predictor = NBAPredictor()
    X = predictor.prepare_features(sample_data)
    
    # Check that all expected features are present
    expected_features = set(predictor.feature_columns)
    actual_features = set(X.columns)
    assert expected_features.issubset(actual_features), f"Missing features: {expected_features - actual_features}"
    
    # Check for NaN values
    assert not X.isna().any().any(), "NaN values found in prepared features"
    
    # Check feature ranges
    assert (X['3pt_volatility'] >= -1).all() and (X['3pt_volatility'] <= 1).all(), "3pt_volatility out of range"
    assert (X['pace_adjusted_offense'] > 0).all(), "Invalid pace_adjusted_offense values"
    assert (X['pace_adjusted_defense'] > 0).all(), "Invalid pace_adjusted_defense values"

def test_model_training(sample_data):
    predictor = NBAPredictor()
    metrics = predictor.train_models(sample_data, test_size=0.2)
    
    # Check that training completes and returns metrics
    assert isinstance(metrics, dict), "Training should return metrics dictionary"
    assert 'moneyline_accuracy' in metrics, "Missing moneyline accuracy metric"
    assert 'spread_rmse' in metrics, "Missing spread RMSE metric"
    assert 'totals_rmse' in metrics, "Missing totals RMSE metric"
    
    # Check metric ranges
    assert 0 <= metrics['moneyline_accuracy'] <= 1, "Invalid moneyline accuracy"
    assert metrics['moneyline_accuracy'] > 0.5, "Model performing worse than random"
    assert metrics['spread_rmse'] > 0, "Invalid spread RMSE"
    assert metrics['totals_rmse'] > 0, "Invalid totals RMSE"

def test_prediction_stability(sample_data):
    predictor = NBAPredictor()
    predictor.train_models(sample_data, test_size=0.2)
    
    # Make predictions on same data multiple times
    pred1 = predictor.predict_games(sample_data)
    pred2 = predictor.predict_games(sample_data)
    
    # Check prediction consistency
    pd.testing.assert_frame_equal(pred1, pred2, "Predictions not consistent between runs")
    
    # Check prediction ranges
    assert (pred1['Home_Win_Prob'] >= 0).all() and (pred1['Home_Win_Prob'] <= 1).all(), "Invalid probability range"
    assert (pred1['Away_Win_Prob'] >= 0).all() and (pred1['Away_Win_Prob'] <= 1).all(), "Invalid probability range"
    assert (pred1['Home_Win_Prob'] + pred1['Away_Win_Prob']).between(0.99, 1.01).all(), "Probabilities don't sum to 1" 