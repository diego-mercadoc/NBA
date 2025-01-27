import json
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import joblib
import os
from datetime import datetime
import pytz
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nba_scraper.log'),
        logging.StreamHandler()
    ]
)

class ModelTuner:
    """
    Automated model tuning using RandomizedSearchCV.
    Uses existing configuration from .cursorrules.
    Does not modify existing code or models unless performance improves.
    """
    
    def __init__(self):
        """Initialize tuner with configuration from .cursorrules"""
        with open('.cursorrules', 'r') as f:
            self.config = json.load(f)
        
        self.tuning_config = self.config['model_validation']['hyperparameter_tuning']
        self.validation_periods = self.tuning_config['validation_periods']
        self.optimization_metrics = self.tuning_config['optimization_metrics']
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Ensure models directory exists
        if not os.path.exists('models'):
            os.makedirs('models')
        if not os.path.exists('models/backup'):
            os.makedirs('models/backup')
    
    def load_data(self):
        """Load and preprocess data using existing format"""
        try:
            df = pd.read_csv('nba_games_all.csv')
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Get configuration
            data_validation = self.config.get('data_validation', {})
            model_retraining = data_validation.get('model_retraining', {})
            history = model_retraining.get('history', {})
            
            # Apply data cutoff using model retraining history
            cutoff_date = history.get('data_coverage', {}).get('start_date')
            if cutoff_date:
                df = df[df['Date'] >= cutoff_date]
                logging.info(f"Applied data cutoff from {cutoff_date}")
            
            # Convert boolean columns
            bool_columns = ['Is_Future', 'Is_Scheduled', 'Is_Played']
            for col in bool_columns:
                if col in df.columns:
                    df[col] = df[col].astype(bool)
            
            # Handle points for played games
            df.loc[df['Is_Played'], 'Home_Points'] = df.loc[df['Is_Played'], 'Home_Points'].fillna(0)
            df.loc[df['Is_Played'], 'Away_Points'] = df.loc[df['Is_Played'], 'Away_Points'].fillna(0)
            
            # Log data quality metrics
            total_games = len(df)
            played_games = df['Is_Played'].sum()
            future_games = df['Is_Future'].sum()
            scheduled_games = df['Is_Scheduled'].sum()
            
            logging.info(f"Data summary:")
            logging.info(f"Total games: {total_games}")
            logging.info(f"Played games: {played_games}")
            logging.info(f"Future games: {future_games}")
            logging.info(f"Scheduled games: {scheduled_games}")
            
            # Check data coverage
            expected_coverage = history.get('data_coverage', {})
            expected_games = expected_coverage.get('total_games', 0)
            
            if total_games < expected_games * 0.9:  # Allow 10% tolerance
                logging.warning(f"Found {total_games} games, expected at least {expected_games}")
            
            # Validate season breakdown
            season_breakdown = expected_coverage.get('season_breakdown', {})
            for season, expected_count in season_breakdown.items():
                season_games = len(df[df['Season'] == season])
                if season_games > 0:  # Only log seasons we have data for
                    logging.info(f"Season {season}: {season_games} games (Expected: {expected_count})")
            
            return df
            
        except KeyError as e:
            logging.error(f"Missing configuration key: {str(e)}")
            return None
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            return None
    
    def prepare_features(self, df):
        """Prepare features with proper naming."""
        try:
            df = df.copy()
            
            # Convert season format (e.g., '2024-25' to '2025')
            df['Season'] = df['Season'].apply(lambda x: str(x).split('-')[1] if '-' in str(x) else str(x))
            
            # Calculate win/loss columns
            df['Home_Win'] = (df['Home_Points'] > df['Away_Points']).astype(int)
            df['Away_Win'] = (df['Away_Points'] > df['Home_Points']).astype(int)
            
            # Calculate win rates with exponential weighting
            for team_type in ['Home', 'Away']:
                if team_type == 'Home':
                    team_stats = df.groupby(f'{team_type}_Team')['Home_Win'].agg(['count', 'mean'])
                    df[f'{team_type}_Win_Rate'] = df[f'{team_type}_Team'].map(team_stats['mean'])
                else:
                    team_stats = df.groupby(f'{team_type}_Team')['Home_Win'].agg(['count', 'mean'])
                    df[f'{team_type}_Win_Rate'] = df[f'{team_type}_Team'].map(1 - team_stats['mean'])
            
            # Fill NaN values in win rates with 0.5 (neutral)
            df['Home_Win_Rate'] = df['Home_Win_Rate'].fillna(0.5)
            df['Away_Win_Rate'] = df['Away_Win_Rate'].fillna(0.5)
            
            # Calculate win rate difference
            df['Win_Rate_Diff'] = df['Home_Win_Rate'] - df['Away_Win_Rate']
            
            # Compute rolling statistics for different windows
            windows = [3, 5, 10]  # Fixed windows for stability
            for window in windows:
                df = self.compute_rolling_stats(df, window=window)
                if df is None:
                    return None, None, None
            
            # Calculate point differential ratio (using 5-game window as base)
            df['Point_Diff_Ratio'] = (df['Home_Point_Diff_Roll5'] - df['Away_Point_Diff_Roll5']) / 10
            
            # Fill NaN values in rest days with median
            df['Home_Rest_Days'] = df['Home_Rest_Days'].fillna(df['Home_Rest_Days'].median())
            df['Away_Rest_Days'] = df['Away_Rest_Days'].fillna(df['Away_Rest_Days'].median())
            df['Rest_Advantage'] = df['Home_Rest_Days'] - df['Away_Rest_Days']
            
            # Apply rest impact factors
            rest_config = self.config['model_validation']['feature_weights']['rest_impact']
            df['Rest_Advantage'] = df['Rest_Advantage'] * rest_config['base_factor']
            
            # Fill NaN values in streaks with 0
            df['Home_Streak'] = df['Home_Streak'].fillna(0)
            df['Away_Streak'] = df['Away_Streak'].fillna(0)
            df['Streak_Advantage'] = df['Home_Streak'] - df['Away_Streak']
            
            # Apply streak importance factors
            streak_config = self.config['model_validation']['feature_weights']['streak_importance']
            df['Streak_Advantage'] = df['Streak_Advantage'] * streak_config['base_weight']
            
            # Calculate form ratio (using 3-game window for recent form)
            df['Recent_Form_Ratio'] = (df['Home_Point_Diff_Roll3'] - df['Away_Point_Diff_Roll3']) / 10
            
            # Interaction features
            df['Win_Rate_Rest_Interaction'] = df['Win_Rate_Diff'] * df['Rest_Advantage']
            df['Streak_Form_Interaction'] = df['Streak_Advantage'] * df['Recent_Form_Ratio']
            
            # Apply season weights
            season_weights = self.config['model_validation']['season_weights']
            current_season = self.config['season_handling']['current_season'].split('-')[1]  # Get end year
            df['Season_Weight'] = df['Season'].map({
                current_season: season_weights['current'],
                str(int(current_season) - 1): season_weights['previous'],
                str(int(current_season) - 2): season_weights['historical']
            }).fillna(season_weights['historical'])
            
            # Prepare feature matrix
            feature_cols = [
                'Win_Rate_Diff', 'Point_Diff_Ratio', 'Rest_Advantage',
                'Streak_Advantage', 'Recent_Form_Ratio', 'Win_Rate_Rest_Interaction',
                'Streak_Form_Interaction', 'Season_Weight',
                'Home_Point_Diff_Roll3', 'Away_Point_Diff_Roll3',
                'Home_Point_Diff_Roll5', 'Away_Point_Diff_Roll5',
                'Home_Point_Diff_Roll10', 'Away_Point_Diff_Roll10'
            ]
            
            # Filter to only played games for training
            df_train = df[df['Is_Played'] & ~df['Is_Future']].copy()
            
            # Prepare X and y
            X = df_train[feature_cols].fillna(0)
            y = df_train['Home_Win'].values
            
            # Scale features using tanh for final normalization
            X_scaled = X.copy()
            for col in X_scaled.columns:
                if col != 'Season_Weight':  # Don't normalize season weights
                    X_scaled[col] = np.tanh(X_scaled[col])
            
            logging.info(f"Prepared {len(feature_cols)} features for {len(X)} samples")
            
            return X_scaled.values, y, feature_cols
            
        except Exception as e:
            logging.error(f"Error in feature preparation: {str(e)}")
            return None, None, None
    
    def compute_rolling_stats(self, games_df, window=5):
        """Compute rolling statistics for teams"""
        try:
            df = games_df.copy()
            
            # Initialize columns in original dataframe as float64
            df[f'Home_Point_Diff_Roll{window}'] = 0.0
            df[f'Away_Point_Diff_Roll{window}'] = 0.0
            
            # Sort by date and filter out future games for calculation
            df_hist = df[~df['Is_Future']].sort_values('Date')
            
            if len(df_hist) == 0:
                logging.warning("No historical games found for rolling stats")
                return df
            
            # Calculate points scored and allowed for historical games
            df_hist['Home_Points_Scored'] = df_hist['Home_Points'].fillna(0).astype(float)
            df_hist['Home_Points_Allowed'] = df_hist['Away_Points'].fillna(0).astype(float)
            df_hist['Away_Points_Scored'] = df_hist['Away_Points'].fillna(0).astype(float)
            df_hist['Away_Points_Allowed'] = df_hist['Home_Points'].fillna(0).astype(float)
            
            # Calculate rolling stats for home teams
            for team in df_hist['Home_Team'].unique():
                # Get all games where team played (home or away)
                team_games = pd.concat([
                    df_hist[df_hist['Home_Team'] == team][['Date', 'Home_Points_Scored', 'Home_Points_Allowed']].rename(
                        columns={'Home_Points_Scored': 'Points_Scored', 'Home_Points_Allowed': 'Points_Allowed'}
                    ),
                    df_hist[df_hist['Away_Team'] == team][['Date', 'Away_Points_Scored', 'Away_Points_Allowed']].rename(
                        columns={'Away_Points_Scored': 'Points_Scored', 'Away_Points_Allowed': 'Points_Allowed'}
                    )
                ]).sort_values('Date')
                
                if not team_games.empty:
                    # Calculate point differential
                    team_games['Point_Diff'] = team_games['Points_Scored'] - team_games['Points_Allowed']
                    # Calculate rolling average
                    team_games[f'Point_Diff_Roll{window}'] = team_games['Point_Diff'].rolling(
                        window=window, min_periods=1
                    ).mean()
                    
                    # Map back to original dataframe
                    latest_diff = float(team_games[f'Point_Diff_Roll{window}'].iloc[-1] if len(team_games) > 0 else 0)
                    
                    # Update home games
                    df.loc[df['Home_Team'] == team, f'Home_Point_Diff_Roll{window}'] = latest_diff
                    # Update away games
                    df.loc[df['Away_Team'] == team, f'Away_Point_Diff_Roll{window}'] = latest_diff
            
            return df
            
        except Exception as e:
            logging.error(f"Error in compute_rolling_stats: {str(e)}")
            return None
    
    def tune_random_forest(self, X, y, feature_names=None):
        """Tune Random Forest model with progress tracking."""
        try:
            logging.info("\nTuning random_forest...")
            
            # Get parameters from configuration
            param_grid = self.tuning_config['safe_ranges']['random_forest']
            n_iter = self.tuning_config.get('n_iterations', 50)
            
            # Create parameter distributions
            param_dist = {
                'n_estimators': param_grid['n_estimators'],
                'max_depth': param_grid['max_depth'],
                'min_samples_split': param_grid['min_samples_split'],
                'min_samples_leaf': param_grid['min_samples_leaf']
            }
            
            # Initialize model
            rf = RandomForestClassifier(random_state=42)
            
            # Initialize RandomizedSearchCV with progress callback
            search = RandomizedSearchCV(
                rf, param_dist,
                n_iter=n_iter,
                cv=5,
                scoring='accuracy',
                n_jobs=-1,
                verbose=2,
                random_state=42
            )
            
            # Track start time
            start_time = time.time()
            
            # Fit the model
            search.fit(X, y)
            
            # Log results
            duration = time.time() - start_time
            logging.info(f"\nTuning completed in {duration:.2f} seconds")
            logging.info(f"Best parameters: {search.best_params_}")
            logging.info(f"Best cross-validation score: {search.best_score_:.4f}")
            
            # Early stopping check
            if search.best_score_ < self.config['model_validation']['minimum_accuracy']['moneyline']:
                logging.warning("Model performance below minimum accuracy threshold")
                return None
            
            return search.best_estimator_
            
        except Exception as e:
            logging.error(f"Error in random forest tuning: {str(e)}")
            return None
    
    def tune_lightgbm(self, X, y):
        """Tune LightGBM model"""
        try:
            param_dist = self.tuning_config['safe_ranges']['lightgbm']
            lgbm = lgb.LGBMClassifier(random_state=42)
            
            search = RandomizedSearchCV(
                lgbm,
                param_distributions=param_dist,
                n_iter=self.tuning_config['n_iterations'],
                cv=self.tuning_config['cross_validation']['folds'],
                scoring=self.optimization_metrics['moneyline'],
                n_jobs=-1,
                random_state=42
            )
            
            search.fit(X, y)
            logging.info(f"Best LightGBM parameters: {search.best_params_}")
            logging.info(f"Best score: {search.best_score_:.3f}")
            
            return search.best_estimator_, search.best_score_
        except Exception as e:
            logging.error(f"Error tuning LightGBM: {str(e)}")
            return None, None
    
    def tune_xgboost(self, X, y):
        """Tune XGBoost model"""
        try:
            param_dist = self.tuning_config['safe_ranges']['xgboost']
            xgb_model = xgb.XGBClassifier(random_state=42)
            
            search = RandomizedSearchCV(
                xgb_model,
                param_distributions=param_dist,
                n_iter=self.tuning_config['n_iterations'],
                cv=self.tuning_config['cross_validation']['folds'],
                scoring=self.optimization_metrics['moneyline'],
                n_jobs=-1,
                random_state=42
            )
            
            search.fit(X, y)
            logging.info(f"Best XGBoost parameters: {search.best_params_}")
            logging.info(f"Best score: {search.best_score_:.3f}")
            
            return search.best_estimator_, search.best_score_
        except Exception as e:
            logging.error(f"Error tuning XGBoost: {str(e)}")
            return None, None
    
    def validate_performance(self, model, X, y, previous_score):
        """Validate model performance against thresholds"""
        try:
            scores = cross_val_score(
                model,
                X,
                y,
                cv=self.tuning_config['cross_validation']['folds'],
                scoring=self.optimization_metrics['moneyline']
            )
            
            current_score = scores.mean()
            score_diff = current_score - previous_score
            
            if score_diff < -self.tuning_config['backup']['performance_threshold']:
                logging.warning(f"Performance regression detected: {score_diff:.3f}")
                return False
            
            logging.info(f"Performance improvement: {score_diff:.3f}")
            return True
        except Exception as e:
            logging.error(f"Error validating performance: {str(e)}")
            return False
    
    def save_model(self, model, model_type):
        """Save model with versioning"""
        try:
            # Backup existing model
            model_path = f'models/{model_type}_model.joblib'
            if os.path.exists(model_path):
                backup_path = f'models/backup/{model_type}_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.joblib'
                os.rename(model_path, backup_path)
            
            # Save new model
            joblib.dump(model, model_path)
            logging.info(f"Saved {model_type} model: {model_path}")
            return True
        except Exception as e:
            logging.error(f"Error saving model: {str(e)}")
            return False
    
    def validate_data(self, df):
        """Validate data integrity and format"""
        try:
            required_cols = [
                'Date', 'Home_Team', 'Away_Team', 'Home_Points', 'Away_Points',
                'Season', 'Home_Rest_Days', 'Away_Rest_Days', 'Home_Streak', 'Away_Streak',
                'Is_Future', 'Is_Scheduled', 'Is_Played'
            ]
            
            # Check required columns
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logging.error(f"Missing required columns: {missing_cols}")
                return False
            
            # Filter to only played games for training
            df_train = df[df['Is_Played'] & ~df['Is_Future']].copy()
            
            # Log data filtering results
            logging.info(f"Training data summary:")
            logging.info(f"Total games in dataset: {len(df)}")
            logging.info(f"Games available for training: {len(df_train)}")
            
            # Validate data types
            if not pd.api.types.is_datetime64_any_dtype(df_train['Date']):
                logging.error("Date column is not datetime type")
                return False
            
            # Check for invalid values in played games
            nan_points = df_train[['Home_Points', 'Away_Points']].isnull().any(axis=1)
            if nan_points.any():
                problem_games = df_train[nan_points]
                logging.error(f"Found {len(problem_games)} games with NaN points:")
                for _, game in problem_games.head().iterrows():
                    logging.error(f"Game on {game['Date']}: {game['Away_Team']} @ {game['Home_Team']}")
                return False
            
            # Validate season format
            if not df_train['Season'].astype(str).str.match(r'^\d{4}$').all():
                logging.error("Invalid season format found")
                return False
            
            # Validate team names
            teams_per_season = df_train.groupby('Season').agg({
                'Home_Team': 'nunique',
                'Away_Team': 'nunique'
            })
            
            for season in teams_per_season.index:
                home_teams = teams_per_season.loc[season, 'Home_Team']
                away_teams = teams_per_season.loc[season, 'Away_Team']
                if home_teams != 30 or away_teams != 30:
                    logging.warning(f"Season {season} has unexpected number of teams: {home_teams} home, {away_teams} away")
            
            # Log validation results
            logging.info("Data validation passed successfully")
            logging.info(f"Training data spans from {df_train['Date'].min()} to {df_train['Date'].max()}")
            logging.info(f"Number of seasons: {len(df_train['Season'].unique())}")
            
            return True
            
        except Exception as e:
            logging.error(f"Error in data validation: {str(e)}")
            return False
    
    def validate_features(self, X, feature_names=None):
        """Validate the prepared features."""
        try:
            X_df = pd.DataFrame(X, columns=feature_names if feature_names else [f'Feature_{i}' for i in range(X.shape[1])])
            
            # Check for NaN values
            if np.isnan(X).any():
                logging.error("NaN values found in features")
                return False
            
            # Check for infinite values
            if np.isinf(X).any():
                logging.error("Infinite values found in features")
                return False
            
            # Log feature statistics
            logging.info("\nFeature Statistics:")
            for col in X_df.columns:
                logging.info(f"\n{col}:")
                logging.info(f"  Mean: {X_df[col].mean():.3f}")
                logging.info(f"  Std: {X_df[col].std():.3f}")
                logging.info(f"  Min: {X_df[col].min():.3f}")
                logging.info(f"  Max: {X_df[col].max():.3f}")
                
                # Value range validation for non-weight features
                if col != 'Season_Weight':
                    if not (-1 <= X_df[col].min() <= X_df[col].max() <= 1):
                        logging.warning(f"Feature {col} contains values outside [-1, 1] range")
                        # Don't fail validation, just warn
            
            logging.info("Feature validation passed successfully")
            return True
            
        except Exception as e:
            logging.error(f"Error in feature validation: {str(e)}")
            return False
    
    def validate_model_performance(self, model, X, y, model_type):
        """Validate model performance metrics"""
        try:
            # Get minimum accuracy requirements
            min_accuracy = self.config['model_validation']['minimum_accuracy']['moneyline']
            
            # Perform cross-validation
            scores = cross_val_score(
                model,
                X,
                y,
                cv=self.tuning_config['cross_validation']['folds'],
                scoring=self.optimization_metrics['moneyline']
            )
            
            mean_score = scores.mean()
            std_score = scores.std()
            
            logging.info(f"{model_type} CV Score: {mean_score:.3f} (+/- {std_score:.3f})")
            
            # Check if performance meets minimum requirements
            if mean_score < min_accuracy:
                logging.warning(f"{model_type} performance below minimum requirement: {mean_score:.3f} < {min_accuracy}")
                return False
            
            # Check for overfitting
            train_score = model.score(X, y)
            if train_score - mean_score > 0.1:  # More than 10% difference indicates overfitting
                logging.warning(f"{model_type} shows signs of overfitting: Train={train_score:.3f}, CV={mean_score:.3f}")
                return False
            
            logging.info(f"{model_type} performance validation passed")
            return True
            
        except Exception as e:
            logging.error(f"Error in performance validation: {str(e)}")
            return False
    
    def run_tuning(self):
        """Run complete tuning process with validation"""
        try:
            logging.info("Starting automated model tuning...")
            
            # Load and validate data
            df = self.load_data()
            if df is None or not self.validate_data(df):
                return False
            
            # Prepare and validate features
            X, y, feature_names = self.prepare_features(df)
            if X is None or y is None:
                return False
            
            # Validate features
            if not self.validate_features(X, feature_names):
                return False
            
            # Tune each model
            models = {
                'random_forest': self.tune_random_forest,
                'lightgbm': self.tune_lightgbm,
                'xgboost': self.tune_xgboost
            }
            
            for model_type, tune_func in models.items():
                logging.info(f"\nTuning {model_type}...")
                
                # Load previous model score
                try:
                    previous_model = joblib.load(f'models/{model_type}_model.joblib')
                    previous_score = cross_val_score(
                        previous_model,
                        X,
                        y,
                        cv=self.tuning_config['cross_validation']['folds'],
                        scoring=self.optimization_metrics['moneyline']
                    ).mean()
                except:
                    previous_score = 0
                
                # Tune model
                model = tune_func(X, y)
                if model is None:
                    continue
                
                # Validate performance
                if not self.validate_model_performance(model, X, y, model_type):
                    logging.warning(f"Skipping {model_type} due to validation failure")
                    continue
                
                # Save if improved
                if self.validate_performance(model, X, y, previous_score):
                    self.save_model(model, model_type)
                else:
                    logging.info(f"Keeping previous {model_type} model")
            
            logging.info("Model tuning completed successfully")
            return True
            
        except Exception as e:
            logging.error(f"Error in tuning process: {str(e)}")
            return False
    
    def test_tuner(self, sample_size=1000):
        """Test tuner implementation with a small data subset"""
        try:
            logging.info(f"Testing tuner with {sample_size} samples...")
            
            # Load data
            df = self.load_data()
            if df is None:
                return False
                
            # Take a sample
            df = df.sort_values('Date').tail(sample_size)
            
            # Validate data
            if not self.validate_data(df):
                return False
            
            # Prepare features
            X, y, feature_names = self.prepare_features(df)
            if X is None or y is None:
                return False
            
            # Validate features
            if not self.validate_features(X, feature_names):
                return False
            
            # Test Random Forest tuning without saving
            rf_model = self.tune_random_forest(X, y)
            if rf_model is None:
                return False
            
            # Validate performance
            if not self.validate_model_performance(rf_model, X, y, 'random_forest'):
                return False
            
            logging.info("Tuner test completed successfully")
            return True
            
        except Exception as e:
            logging.error(f"Error in tuner test: {str(e)}")
            return False

if __name__ == "__main__":
    # Configure logging first
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('model_tuner.log'),
            logging.StreamHandler()
        ]
    )
    
    logging.info("Starting model tuning with full dataset...")
    tuner = ModelTuner()
    
    # Run full tuning process
    success = tuner.run_tuning()
    
    if success:
        logging.info("Model tuning completed successfully")
    else:
        logging.error("Model tuning failed") 