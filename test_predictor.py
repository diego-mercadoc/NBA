import pandas as pd
import numpy as np
import logging
from nba_predictor import NBAPredictor

def test_hyperparameter_optimization():
    """Test the enhanced hyperparameter optimization implementation"""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Load test data
        logging.info("Loading test data...")
        games_df = pd.read_csv('nba_games_all.csv')
        games_df['Date'] = pd.to_datetime(games_df['Date'])
        
        # Initialize predictor
        predictor = NBAPredictor()
        
        # Prepare features and labels
        logging.info("Preparing features and labels...")
        X = predictor.prepare_features(games_df)
        y_moneyline, y_spread, y_totals = predictor.prepare_labels(games_df)
        
        # Convert to numpy arrays
        X = X.to_numpy()
        y_moneyline = y_moneyline.to_numpy()
        y_spread = y_spread.to_numpy()
        y_totals = y_totals.to_numpy()
        
        # Split data for testing
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_ml_train, y_ml_val = train_test_split(
            X, y_moneyline, test_size=0.2, stratify=y_moneyline, random_state=42
        )
        _, _, y_spread_train, y_spread_val = train_test_split(
            X, y_spread, test_size=0.2, random_state=42
        )
        _, _, y_totals_train, y_totals_val = train_test_split(
            X, y_totals, test_size=0.2, random_state=42
        )
        
        # Test moneyline optimization
        logging.info("\nTesting moneyline optimization...")
        try:
            ml_params = predictor.optimize_hyperparameters(
                X_train, y_ml_train, X_val, y_ml_val, model_type='moneyline'
            )
            logging.info("✓ Moneyline optimization completed successfully")
        except Exception as e:
            logging.error(f"✗ Moneyline optimization failed: {str(e)}")
            raise
        
        # Test spread optimization
        logging.info("\nTesting spread optimization...")
        try:
            spread_params = predictor.optimize_hyperparameters(
                X_train, y_spread_train, X_val, y_spread_val, model_type='spread'
            )
            logging.info("✓ Spread optimization completed successfully")
        except Exception as e:
            logging.error(f"✗ Spread optimization failed: {str(e)}")
            raise
        
        # Test totals optimization
        logging.info("\nTesting totals optimization...")
        try:
            totals_params = predictor.optimize_hyperparameters(
                X_train, y_totals_train, X_val, y_totals_val, model_type='totals'
            )
            logging.info("✓ Totals optimization completed successfully")
        except Exception as e:
            logging.error(f"✗ Totals optimization failed: {str(e)}")
            raise
        
        logging.info("\nAll optimization tests completed successfully!")
        
        # Verify parameter ranges
        def verify_param_ranges(params, model_type):
            logging.info(f"\nVerifying {model_type} parameter ranges:")
            if model_type == 'moneyline':
                # Check RF params
                assert 100 <= params.get('rf_n_estimators', 0) <= 500, "RF n_estimators out of range"
                assert 3 <= params.get('rf_max_depth', 0) <= 15, "RF max_depth out of range"
                
                # Check XGB params
                assert 100 <= params.get('xgb_n_estimators', 0) <= 500, "XGB n_estimators out of range"
                assert 3 <= params.get('xgb_max_depth', 0) <= 10, "XGB max_depth out of range"
                assert 0.01 <= params.get('xgb_learning_rate', 0) <= 0.1, "XGB learning_rate out of range"
                
                # Check LGB params
                assert 100 <= params.get('lgb_n_estimators', 0) <= 500, "LGB n_estimators out of range"
                assert 20 <= params.get('lgb_num_leaves', 0) <= 50, "LGB num_leaves out of range"
                
                logging.info("✓ All moneyline parameters within expected ranges")
            else:
                # Check regression params
                assert 100 <= params.get('n_estimators', 0) <= 500, "n_estimators out of range"
                assert 3 <= params.get('max_depth', 0) <= 10, "max_depth out of range"
                assert 0.01 <= params.get('learning_rate', 0) <= 0.1, "learning_rate out of range"
                
                logging.info(f"✓ All {model_type} parameters within expected ranges")
        
        verify_param_ranges(ml_params, 'moneyline')
        verify_param_ranges(spread_params, 'spread')
        verify_param_ranges(totals_params, 'totals')
        
        return True
        
    except Exception as e:
        logging.error(f"Test failed: {str(e)}")
        raise

def test_efg_feature_implementation():
    """Test the implementation of the eFG% differential feature"""
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Load test data
        logging.info("Loading test data...")
        games_df = pd.read_csv('nba_games_all.csv')
        team_stats_df = pd.read_csv('nba_team_stats_all.csv')
        
        # Verify team stats columns
        required_cols = ['Team', 'Season', 'eFG_Pct', 'eFG_Pct_Allowed']
        missing_cols = [col for col in required_cols if col not in team_stats_df.columns]
        assert len(missing_cols) == 0, f"Missing columns in team stats: {missing_cols}"
        
        # Initialize predictor
        predictor = NBAPredictor()
        
        # Prepare features
        logging.info("Preparing features...")
        X = predictor.prepare_features(games_df)
        
        # Verify new features are present
        required_features = ['efg_mismatch', 'eFG_Rest_Interaction']
        missing_features = [feat for feat in required_features if feat not in X.columns]
        assert len(missing_features) == 0, f"Missing features: {missing_features}"
        
        # Verify no NaN values in new features
        nan_counts = X[required_features].isna().sum()
        assert nan_counts.sum() == 0, f"Found NaN values in features: {nan_counts}"
        
        # Verify feature ranges
        assert X['efg_mismatch'].abs().max() <= 1.0, "eFG% mismatch outside expected range"
        
        logging.info("✓ All eFG% feature tests passed successfully")
        return True
        
    except Exception as e:
        logging.error(f"Test failed: {str(e)}")
        raise

if __name__ == "__main__":
    test_hyperparameter_optimization()
    test_efg_feature_implementation() 