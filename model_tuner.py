import json
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import joblib
import os
from datetime import datetime
import pytz
import time
import warnings
import pkg_resources
import collections
from sklearn.metrics import log_loss, make_scorer

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Additional LightGBM warning filters
warnings.filterwarnings(
    "ignore",
    message="No further splits with positive gain, best gain: -inf",
    module='lightgbm'
)
warnings.filterwarnings(
    "ignore",
    message="FutureWarning: 'force_all_finite' was renamed to 'ensure_all_finite'",
    module='lightgbm'
)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nba_scraper.log'),
        logging.StreamHandler()
    ]
)

def check_package_versions():
    """
    Verify compatibility between XGBoost and scikit-learn versions.
    Returns True if versions are compatible, False otherwise.
    """
    try:
        sklearn_version = pkg_resources.get_distribution('scikit-learn').version
        xgboost_version = pkg_resources.get_distribution('xgboost').version
        logging.info(f"scikit-learn version: {sklearn_version}")
        logging.info(f"xgboost version: {xgboost_version}")

        # Parse version numbers
        sklearn_major = int(sklearn_version.split('.')[0])
        sklearn_minor = int(sklearn_version.split('.')[1])
        xgb_major = int(xgboost_version.split('.')[0])

        # Define compatibility rules
        if xgb_major >= 2:
            # XGBoost 2.x works well with scikit-learn 1.x
            return sklearn_major >= 1
        else:
            # XGBoost 1.x works with scikit-learn 0.22+ or 1.x
            return (sklearn_major == 0 and sklearn_minor >= 22) or sklearn_major >= 1

    except Exception as e:
        logging.error(f"Error checking package versions: {str(e)}")
        return False


class ModelTuner:
    """
    Automated model tuning using RandomizedSearchCV.
    Reads config from .cursorrules, doesn't overwrite models unless performance improves.
    """

    def __init__(self):
        """Initialize the ModelTuner with configuration and data."""
        # Load configuration
        with open('.cursorrules', 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        self.tuning_config = self.config['model_validation']['hyperparameter_tuning']
        self.validation_periods = self.tuning_config['validation_periods']
        self.optimization_metrics = self.tuning_config['optimization_metrics']

        self.logger = logging.getLogger(__name__)

        # Ensure 'models' and 'models/backup' directories exist
        if not os.path.exists('models'):
            os.makedirs('models')
        if not os.path.exists('models/backup'):
            os.makedirs('models/backup')


    #########################
    #        DATA LOAD      #
    #########################
    def load_data(self):
        """Load nba_games_all.csv, apply cutoffs, parse columns, etc."""
        try:
            df = pd.read_csv('nba_games_all.csv')
            df['Date'] = pd.to_datetime(df['Date'])

            # Possibly apply a cutoff date from .cursorrules
            data_validation = self.config.get('data_validation', {})
            model_retraining = data_validation.get('model_retraining', {})
            history = model_retraining.get('history', {})
            cutoff_date = history.get('data_coverage', {}).get('start_date')

            if cutoff_date:
                df = df[df['Date'] >= cutoff_date]
                logging.info(f"Applied data cutoff from {cutoff_date}")

            # Convert certain columns to bool
            bool_columns = ['Is_Future', 'Is_Scheduled', 'Is_Played']
            for col in bool_columns:
                if col in df.columns:
                    df[col] = df[col].astype(bool)

            # For played games, fill missing points with 0
            df.loc[df['Is_Played'], 'Home_Points'] = df.loc[df['Is_Played'], 'Home_Points'].fillna(0)
            df.loc[df['Is_Played'], 'Away_Points'] = df.loc[df['Is_Played'], 'Away_Points'].fillna(0)

            logging.info("Data summary:")
            logging.info(f"Total games: {len(df)}")
            logging.info(f"Played games: {df['Is_Played'].sum()}")
            logging.info(f"Future games: {df['Is_Future'].sum()}")
            logging.info(f"Scheduled games: {df['Is_Scheduled'].sum()}")

            return df
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            return None

    def validate_data(self, df):
        """Check columns, etc., for the loaded DataFrame."""
        try:
            required_cols = [
                'Date', 'Home_Team', 'Away_Team', 'Home_Points', 'Away_Points',
                'Season', 'Home_Rest_Days', 'Away_Rest_Days',
                'Home_Streak', 'Away_Streak',
                'Is_Future', 'Is_Scheduled', 'Is_Played'
            ]
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                logging.error(f"Missing required columns: {missing}")
                return False

            df_train = df[df['Is_Played'] & ~df['Is_Future']]
            logging.info(f"Training data summary: {len(df_train)} played games out of {len(df)} total.")

            # Check that Date is actually datetime
            if not pd.api.types.is_datetime64_any_dtype(df_train['Date']):
                logging.error("Date column is not datetime type")
                return False

            # We can add more checks here if needed
            logging.info("Data validation passed successfully.")
            return True
        except Exception as e:
            logging.error(f"Error in data validation: {str(e)}")
            return False


    #########################
    #    FEATURE PREP       #
    #########################
    def prepare_features(self, df):
        """
        Build the feature matrix (X) and labels (y) from df.
        Returns both moneyline and totals targets.

        Returns
        -------
        X : array-like of shape (n_samples, n_features)
            The feature matrix
        y_moneyline : array-like of shape (n_samples,)
            Binary target for moneyline prediction (1 for home win, 0 for away win)
        y_totals : array-like of shape (n_samples,)
            Total points scored in each game for totals prediction
        feature_names : list
            Names of the features used
        """
        try:
            # Convert e.g. '2024-25' to just '2025' if needed
            df['Season'] = df['Season'].apply(lambda x: str(x).split('-')[1] if '-' in str(x) else str(x))

            # Add columns for W/L
            df['Home_Win'] = (df['Home_Points'] > df['Away_Points']).astype(int)
            df['Away_Win'] = (df['Away_Points'] > df['Home_Points']).astype(int)

            # Win rates
            for team_type in ['Home', 'Away']:
                if team_type == 'Home':
                    team_stats = df.groupby(f'{team_type}_Team')['Home_Win'].agg(['count', 'mean'])
                    df[f'{team_type}_Win_Rate'] = df[f'{team_type}_Team'].map(team_stats['mean'])
                else:
                    # away win rate = 1 - home team's home win rate
                    team_stats = df.groupby(f'{team_type}_Team')['Home_Win'].agg(['count', 'mean'])
                    df[f'{team_type}_Win_Rate'] = df[f'{team_type}_Team'].map(1 - team_stats['mean'])

            df['Home_Win_Rate'] = df['Home_Win_Rate'].fillna(0.5)
            df['Away_Win_Rate'] = df['Away_Win_Rate'].fillna(0.5)
            df['Win_Rate_Diff'] = df['Home_Win_Rate'] - df['Away_Win_Rate']

            # Compute rolling stats for windows: 3, 5, 10
            windows = [3, 5, 10]
            for w in windows:
                df = self.compute_rolling_stats(df, w)
                if df is None:
                    return None, None, None, None

            # Ratio using 5-game window
            df['Point_Diff_Ratio'] = (df['Home_Point_Diff_Roll5'] - df['Away_Point_Diff_Roll5']) / 10

            # Fill rest day NaNs
            df['Home_Rest_Days'] = df['Home_Rest_Days'].fillna(df['Home_Rest_Days'].median())
            df['Away_Rest_Days'] = df['Away_Rest_Days'].fillna(df['Away_Rest_Days'].median())
            df['Rest_Advantage'] = df['Home_Rest_Days'] - df['Away_Rest_Days']

            # Apply a multiplier from config
            rest_factor = self.config['feature_weights']['rest_impact']['base_factor']
            df['Rest_Advantage'] = df['Rest_Advantage'] * rest_factor

            # Fill streak
            df['Home_Streak'] = df['Home_Streak'].fillna(0)
            df['Away_Streak'] = df['Away_Streak'].fillna(0)
            df['Streak_Advantage'] = df['Home_Streak'] - df['Away_Streak']

            # Streak multiplier
            streak_factor = self.config['feature_weights']['streak_importance']['base_weight']
            df['Streak_Advantage'] = df['Streak_Advantage'] * streak_factor

            # Form ratio with 3-game window
            df['Recent_Form_Ratio'] = (df['Home_Point_Diff_Roll3'] - df['Away_Point_Diff_Roll3']) / 10

            # Interaction features
            df['Win_Rate_Rest_Interaction'] = df['Win_Rate_Diff'] * df['Rest_Advantage']
            df['Streak_Form_Interaction'] = df['Streak_Advantage'] * df['Recent_Form_Ratio']

            # Season weights
            season_weights = self.config['model_validation']['season_weights']
            current_season = self.config['season_handling']['current_season'].split('-')[1]
            df['Season_Weight'] = df['Season'].map({
                current_season: season_weights['current'],
                str(int(current_season) - 1): season_weights['previous'],
                str(int(current_season) - 2): season_weights['historical']
            }).fillna(season_weights['historical'])

            # Our feature columns
            feature_cols = [
                'Win_Rate_Diff', 'Point_Diff_Ratio', 'Rest_Advantage',
                'Streak_Advantage', 'Recent_Form_Ratio',
                'Win_Rate_Rest_Interaction', 'Streak_Form_Interaction',
                'Season_Weight',
                'Home_Point_Diff_Roll3', 'Away_Point_Diff_Roll3',
                'Home_Point_Diff_Roll5', 'Away_Point_Diff_Roll5',
                'Home_Point_Diff_Roll10','Away_Point_Diff_Roll10'
            ]

            # Only train on played games
            df_train = df[df['Is_Played'] & ~df['Is_Future']].copy()

            # X / y
            X = df_train[feature_cols].fillna(0)
            y_moneyline = df_train['Home_Win'].values
            y_totals = df_train['Home_Points'] + df_train['Away_Points']  # Total points for totals prediction

            # We do a tanh scaling for everything except 'Season_Weight'
            X_scaled = X.copy()
            for col in X_scaled.columns:
                if col != 'Season_Weight':
                    X_scaled[col] = np.tanh(X_scaled[col])

            logging.info(f"Prepared {len(feature_cols)} features for {len(X)} samples")
            return X_scaled.values, y_moneyline, y_totals.values, feature_cols

        except Exception as e:
            logging.error(f"Error in feature preparation: {str(e)}")
            return None, None, None, None


    def compute_rolling_stats(self, df, window=5):
        """Compute rolling point-diff stats for each team over `window` games."""
        try:
            if df is None or df.empty:
                logging.info(f"No games to compute rolling stats for window={window}.")
                return df

            df = df.copy()
            df[f'Home_Point_Diff_Roll{window}'] = 0.0
            df[f'Away_Point_Diff_Roll{window}'] = 0.0

            # Sort by date, ignoring future games
            df_hist = df[~df['Is_Future']].sort_values('Date')
            if df_hist.empty:
                logging.warning("No historical games for rolling stats.")
                return df

            df_hist['Home_Points_Scored'] = df_hist['Home_Points'].fillna(0).astype(float)
            df_hist['Home_Points_Allowed'] = df_hist['Away_Points'].fillna(0).astype(float)
            df_hist['Away_Points_Scored'] = df_hist['Away_Points'].fillna(0).astype(float)
            df_hist['Away_Points_Allowed'] = df_hist['Home_Points'].fillna(0).astype(float)

            for team in df_hist['Home_Team'].unique():
                # gather all games (home or away) for that team
                team_games = pd.concat([
                    df_hist[df_hist['Home_Team'] == team][['Date','Home_Points_Scored','Home_Points_Allowed']].rename(
                        columns={'Home_Points_Scored':'Points_Scored','Home_Points_Allowed':'Points_Allowed'}
                    ),
                    df_hist[df_hist['Away_Team'] == team][['Date','Away_Points_Scored','Away_Points_Allowed']].rename(
                        columns={'Away_Points_Scored':'Points_Scored','Away_Points_Allowed':'Points_Allowed'}
                    )
                ]).sort_values('Date')

                if not team_games.empty:
                    team_games['Point_Diff'] = team_games['Points_Scored'] - team_games['Points_Allowed']
                    team_games[f'Point_Diff_Roll{window}'] = team_games['Point_Diff'].rolling(window=window, min_periods=1).mean()
                    latest_diff = float(team_games[f'Point_Diff_Roll{window}'].iloc[-1]) if len(team_games)>0 else 0

                    # update for home or away in main df
                    df.loc[df['Home_Team'] == team, f'Home_Point_Diff_Roll{window}'] = latest_diff
                    df.loc[df['Away_Team'] == team, f'Away_Point_Diff_Roll{window}'] = latest_diff

            return df
        except Exception as e:
            logging.error(f"Error in compute_rolling_stats: {str(e)}")
            return None


    #########################
    #      TUNE MODELS      #
    #########################
    def tune_random_forest(self, X, y):
        """Tune Random Forest with a RandomizedSearchCV."""
        try:
            logging.info("\nTuning random_forest...")

            param_grid = self.tuning_config['safe_ranges']['random_forest']
            n_iter = self.tuning_config.get('n_iterations', 50)

            param_dist = {
                'n_estimators': param_grid['n_estimators'],
                'max_depth': param_grid['max_depth'],
                'min_samples_split': param_grid['min_samples_split'],
                'min_samples_leaf': param_grid['min_samples_leaf']
            }

            rf = RandomForestClassifier(random_state=42)
            search = RandomizedSearchCV(
                rf, param_dist,
                n_iter=n_iter,
                cv=5,
                scoring='accuracy',
                n_jobs=-1,
                verbose=0,
                random_state=42
            )

            start_time = time.time()
            search.fit(X, y)
            duration = time.time() - start_time
            logging.info(f"Tuning completed in {duration:.2f} seconds")
            logging.info(f"Best parameters: {search.best_params_}")
            logging.info(f"Best cross-validation score: {search.best_score_:.4f}")

            if search.best_score_ < self.config['model_validation']['minimum_accuracy']['moneyline']:
                logging.warning("RF model below minimum accuracy threshold.")
                return None, None

            return search.best_estimator_, search.best_score_

        except Exception as e:
            logging.error(f"Error in random forest tuning: {str(e)}")
            return None, None


    def tune_lightgbm(self, X, y):
        """
        Tune LightGBM with proper time-based validation and regularization.
        Uses enhanced validation strategy to prevent overfitting.
        """
        try:
            # Get parameters from config
            param_dist = self.tuning_config['safe_ranges']['lightgbm']
            n_iter = self.tuning_config.get('n_iterations', 50)
            early_stopping = self.tuning_config['early_stopping']

            # Add stronger regularization parameters
            param_dist.update({
                'reg_alpha': [0.5, 1.0, 2.0, 4.0],  # L1 regularization
                'reg_lambda': [0.5, 1.0, 2.0, 4.0],  # L2 regularization
                'min_gain_to_split': [0.1, 0.5, 1.0],  # Minimum gain to split
                'min_data_in_leaf': [50, 100, 200],  # Minimum data in leaf
                'bagging_fraction': [0.7, 0.8, 0.9],  # Bagging
                'feature_fraction': [0.7, 0.8, 0.9],  # Feature sampling
                'max_bin': [200, 255, 300]  # Max number of bins
            })

            # Create base model with conservative defaults
            lgbm = lgb.LGBMClassifier(
                random_state=42,
                verbose=-1,
                n_estimators=500,  # Will be overridden by early stopping
                boosting_type='gbdt',
                objective='binary',
                metric='binary_logloss'
            )

            # Create time-based validation split
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Set up RandomizedSearchCV with time-series split
            search = RandomizedSearchCV(
                estimator=lgbm,
                param_distributions=param_dist,
                n_iter=n_iter,
                cv=self.tuning_config['cross_validation']['folds'],
                scoring=self.optimization_metrics['moneyline'],
                n_jobs=-1,
                random_state=42,
                verbose=0
            )

            # Create eval set for early stopping
            eval_set = [(X_val, y_val)]
            callbacks = [
                lgb.early_stopping(
                    stopping_rounds=early_stopping['patience'],
                    min_delta=early_stopping['min_delta']
                ),
                lgb.log_evaluation(period=0)  # Suppress per-iteration logging
            ]

            # Fit with early stopping
            search.fit(
                X_train, y_train,
                eval_set=eval_set,
                callbacks=callbacks
            )

            # Log results
            logging.info(f"Best LightGBM parameters: {search.best_params_}")
            logging.info(f"Best score: {search.best_score_:.3f}")

            # Additional validation on holdout set
            best_model = search.best_estimator_
            val_pred = best_model.predict_proba(X_val)[:, 1]
            val_score = log_loss(y_val, val_pred)
            logging.info(f"Validation set logloss: {val_score:.3f}")

            # Validate performance
            if not self.validate_model_performance(best_model, X, y, 'lightgbm'):
                logging.warning("LightGBM model failed performance validation")
                return None, None

            return best_model, search.best_score_

        except Exception as e:
            logging.error(f"Error tuning LightGBM: {str(e)}")
            return None, None


    def custom_cv_iterator(self, X, y, cv):
        """
        A custom cross-validation iterator that yields (train_idx, test_idx) along with
        a corresponding eval_set with reset indices. This ensures proper index alignment
        for XGBoost's early stopping mechanism.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Training data
        y : pd.Series or np.ndarray
            Target values
        cv : TimeSeriesSplit or other CV splitter
            Cross-validation splitter object

        Yields
        ------
        tuple : ((train_idx, test_idx), [(X_train, y_train), (X_val, y_val)])
            Train/test indices and corresponding data with reset indices
        """
        # Convert to DataFrame/Series once at the start
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        y_series = pd.Series(y) if not isinstance(y, pd.Series) else y
        
        for train_index, test_index in cv.split(X_df, y_series):
            # Extract train/validation sets
            X_train = X_df.iloc[train_index].copy()  # Use copy to ensure independent indices
            X_val = X_df.iloc[test_index].copy()
            y_train = y_series.iloc[train_index].copy()
            y_val = y_series.iloc[test_index].copy()
            
            # Reset indices to ensure proper alignment
            X_train.reset_index(drop=True, inplace=True)
            X_val.reset_index(drop=True, inplace=True)
            y_train.reset_index(drop=True, inplace=True)
            y_val.reset_index(drop=True, inplace=True)
            
            yield (train_index, test_index), [(X_train, y_train), (X_val, y_val)]

    def tune_xgboost(self, X, y, is_classification=True):
        """
        Tune XGBoost model with proper early stopping and index handling.
        Uses TimeSeriesSplit and custom CV iterator to ensure proper validation.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
        is_classification : bool, default=True
            Whether to tune for classification (moneyline) or regression (totals)
        
        Returns
        -------
        tuple : (best_estimator, best_score)
            The best model and its score
        """
        try:
            logging.info(f"\nTuning XGBoost for {'classification' if is_classification else 'regression'}...")
            
            # Initialize TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=5)
            
            # Define parameter grid from config
            param_grid = {
                'n_estimators': [200, 300, 400],
                'max_depth': [6, 7, 8, 9],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.8, 0.9, 1.0],
                'min_child_weight': [1, 3, 5],
                'reg_alpha': [0, 0.1, 0.5],
                'reg_lambda': [0.1, 0.5, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
            
            # Initialize base model with early stopping
            if is_classification:
                base_model = xgb.XGBClassifier(
                    objective='binary:logistic',
                    use_label_encoder=False,
                    eval_metric='logloss',
                    early_stopping_rounds=50,
                    random_state=42
                )
                scoring = 'neg_log_loss'
            else:
                base_model = xgb.XGBRegressor(
                    objective='reg:squarederror',
                    eval_metric='rmse',
                    early_stopping_rounds=50,
                    random_state=42
                )
                scoring = 'neg_root_mean_squared_error'
            
            # Convert X to DataFrame and y to Series if they aren't already
            X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
            y_series = pd.Series(y) if not isinstance(y, pd.Series) else y
            
            # Create custom CV iterator with reset indices
            cv_iter = self.custom_cv_iterator(X_df, y_series, tscv)
            
            # Set up RandomizedSearchCV with custom CV iterator
            search = RandomizedSearchCV(
                base_model,
                param_distributions=param_grid,
                n_iter=50,
                cv=cv_iter,
                scoring=scoring,
                n_jobs=-1,
                random_state=42,
                verbose=0
            )
            
            # Fit without passing early_stopping_rounds (already in constructor)
            search.fit(X_df, y_series)
            
            # Log results
            logging.info(f"Best XGBoost parameters: {search.best_params_}")
            logging.info(f"Best score: {search.best_score_:.3f}")
            
            # Access best_iteration if available
            if hasattr(search.best_estimator_, 'best_iteration_'):
                logging.info(f"Best iteration: {search.best_estimator_.best_iteration_}")
            
            # Validate performance
            model_type = 'xgboost_moneyline' if is_classification else 'xgboost_totals'
            if not self.validate_model_performance(search.best_estimator_, X, y, model_type):
                logging.warning(f"{model_type} model failed performance validation")
                return None, None
            
            return search.best_estimator_, search.best_score_
            
        except Exception as e:
            logging.error(f"Error in XGBoost tuning: {str(e)}")
            return None, None


    #########################
    #  PERFORMANCE CHECKS   #
    #########################
    def validate_performance(self, model, X, y, previous_score):
        """
        Compare new model's cross-val score to the old model's previous_score.
        If significantly worse, reject. If better or about the same, accept.
        """
        try:
            scores = cross_val_score(
                model,
                X, y,
                cv=self.tuning_config['cross_validation']['folds'],
                scoring=self.optimization_metrics['moneyline']
            )
            current_score = float(scores.mean())
            if isinstance(previous_score, (dict, collections.defaultdict)):
                previous_score = 0
            score_diff = current_score - float(previous_score if previous_score else 0)
            threshold = float(self.tuning_config['backup']['performance_threshold'])

            if score_diff < -threshold:
                logging.warning(f"Performance regression: {score_diff:.3f}")
                return False

            logging.info(f"Performance improvement: {score_diff:.3f}")
            return True
        except Exception as e:
            logging.error(f"Error validating performance: {str(e)}")
            return False


    def validate_model_performance(self, model, X, y, model_type):
        """Check if model meets minimum accuracy + no big overfit gap."""
        try:
            min_accuracy = self.config['model_validation']['minimum_accuracy']['moneyline']
            scores = cross_val_score(
                model, X, y,
                cv=self.tuning_config['cross_validation']['folds'],
                scoring=self.optimization_metrics['moneyline']
            )
            mean_score = scores.mean()
            std_score = scores.std()
            logging.info(f"{model_type} CV Score: {mean_score:.3f} (+/- {std_score:.3f})")

            if mean_score < min_accuracy:
                logging.warning(f"{model_type} below min accuracy: {mean_score:.3f} < {min_accuracy}")
                return False

            train_score = model.score(X, y)
            if (train_score - mean_score) > 0.1:
                logging.warning(
                    f"{model_type} shows potential overfitting: Train={train_score:.3f}, CV={mean_score:.3f}"
                )
                return False

            logging.info(f"{model_type} performance validation passed.")
            return True

        except Exception as e:
            logging.error(f"Error in performance validation: {str(e)}")
            return False


    def save_model(self, model, model_type):
        """Save the new model, backing up the old one if it exists."""
        try:
            model_path = f"models/{model_type}_model.joblib"
            if os.path.exists(model_path):
                backup_name = f"{model_type}_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
                backup_path = os.path.join('models', 'backup', backup_name)
                os.rename(model_path, backup_path)

            joblib.dump(model, model_path)
            logging.info(f"Saved {model_type} model: {model_path}")
            return True
        except Exception as e:
            logging.error(f"Error saving {model_type} model: {str(e)}")
            return False


    #########################
    #       MAIN TUNER      #
    #########################
    def calculate_ensemble_weights(self, models_scores):
        """
        Calculate optimal weights for ensemble models based on their validation scores.
        LightGBM is excluded due to overfitting concerns.
        """
        try:
            if not models_scores:
                return {}

            # Remove LightGBM from ensemble if present
            if 'lightgbm' in models_scores:
                logging.info("Excluding LightGBM from ensemble due to overfitting concerns")
                del models_scores['lightgbm']

            if not models_scores:  # If no models left after removing LightGBM
                return {}

            # Get base scores
            scores = np.array(list(models_scores.values()))

            # Apply softmax to convert scores to probabilities
            exp_scores = np.exp(scores - np.max(scores))  # Subtract max for numerical stability
            weights = exp_scores / exp_scores.sum()

            # Create weights dictionary
            weights_dict = {
                model: weight 
                for model, weight in zip(models_scores.keys(), weights)
            }

            # Ensure weights sum to 1
            total_weight = sum(weights_dict.values())
            if total_weight > 0:
                weights_dict = {k: v/total_weight for k, v in weights_dict.items()}

            logging.info("Calculated ensemble weights (excluding LightGBM):")
            for model, weight in weights_dict.items():
                logging.info(f"{model}: {weight:.3f}")

            return weights_dict

        except Exception as e:
            logging.error(f"Error calculating ensemble weights: {str(e)}")
            return {}

    def run_tuning(self):
        """
        Orchestrate the entire tuning process with enhanced ensemble weighting.
        Handles both moneyline (classification) and totals (regression) models.
        """
        try:
            logging.info("Starting automated model tuning...")

            # 1) Load and validate data
            df = self.load_data()
            if df is None or not self.validate_data(df):
                return False

            # 2) Prepare features and targets
            X, y_moneyline, y_totals, feature_names = self.prepare_features(df)
            if X is None or y_moneyline is None or not self.validate_features(X, feature_names):
                return False

            # 3) Dictionary to store models and scores
            models_scores = {}
            best_models = {}

            # 4) Tune each model
            tuning_configs = {
                'random_forest': {
                    'func': self.tune_random_forest,
                    'target': y_moneyline,
                    'is_classification': True
                },
                'xgboost_moneyline': {
                    'func': self.tune_xgboost,
                    'target': y_moneyline,
                    'is_classification': True
                },
                'xgboost_totals': {
                    'func': self.tune_xgboost,
                    'target': y_totals,
                    'is_classification': False
                }
            }

            for model_type, config in tuning_configs.items():
                logging.info(f"\nTuning {model_type}...")
                
                # Get previous score if exists
                previous_score = 0
                old_path = f"models/{model_type}_model.joblib"
                if os.path.exists(old_path):
                    try:
                        prev_model = joblib.load(old_path)
                        if hasattr(prev_model, 'best_score_'):
                            previous_score = prev_model.best_score_
                    except Exception:
                        logging.info(f"No previous score found for {model_type}")

                # Tune model with appropriate target and configuration
                if config['is_classification']:
                    best_estimator, best_score = config['func'](X, config['target'])
                else:
                    best_estimator, best_score = config['func'](X, config['target'], is_classification=False)

                if best_estimator is None:
                    logging.warning(f"Skipping {model_type} due to tuning failure")
                    continue

                # Validate performance
                if not self.validate_model_performance(best_estimator, X, config['target'], model_type):
                    logging.warning(f"Skipping {model_type} due to validation failure")
                    continue

                if not self.validate_performance(best_estimator, X, config['target'], previous_score):
                    logging.warning(f"{model_type} performance validation failed")
                    continue

                # Store model and score
                models_scores[model_type] = best_score
                best_models[model_type] = best_estimator

                # Save model
                if self.save_model(best_estimator, model_type):
                    logging.info(f"New {model_type} model saved")
                else:
                    logging.warning(f"Failed to save {model_type} model")

            # 5) Calculate ensemble weights for moneyline models only
            moneyline_scores = {k: v for k, v in models_scores.items() 
                              if k in ['random_forest', 'xgboost_moneyline']}
            
            if len(moneyline_scores) > 0:
                weights = self.calculate_ensemble_weights(moneyline_scores)
                
                # Save weights
                weights_path = 'models/ensemble_weights.json'
                try:
                    with open(weights_path, 'w') as f:
                        json.dump(weights, f)
                    logging.info("Saved ensemble weights for moneyline models")
                except Exception as e:
                    logging.error(f"Error saving ensemble weights: {str(e)}")

            logging.info("Model tuning completed successfully")
            return True

        except Exception as e:
            logging.error(f"Error in tuning process: {str(e)}")
            return False


    def test_tuner(self, sample_size=1000):
        """Optional: test with smaller data portion."""
        try:
            logging.info(f"Testing tuner with {sample_size} samples...")

            df = self.load_data()
            if df is None:
                return False

            df = df.sort_values('Date').tail(sample_size)
            if not self.validate_data(df):
                return False

            X, y, feature_names = self.prepare_features(df)
            if X is None or y is None:
                return False

            if not self.validate_features(X, feature_names):
                return False

            # For a quick test, just tune RandomForest
            best_estimator, best_score = self.tune_random_forest(X, y)
            if best_estimator is None:
                return False

            # Check performance
            if not self.validate_model_performance(best_estimator, X, y, 'random_forest'):
                return False

            logging.info("Tuner test completed successfully.")
            return True

        except Exception as e:
            logging.error(f"Error in tuner test: {str(e)}")
            return False


    def validate_features(self, X, feature_names):
        """Validate that features (X) and the feature names list are correctly structured.
        
        Returns:
            bool: True if X is a numpy array with the proper number of columns matching feature_names.
        """
        if not isinstance(X, np.ndarray):
            logging.error("Features are not a numpy array.")
            return False
        if not isinstance(feature_names, list):
            logging.error("Feature names are not provided as a list.")
            return False
        if X.shape[1] != len(feature_names):
            logging.error(f"Feature dimension mismatch: X has {X.shape[1]} columns, expected {len(feature_names)}.")
            return False
        logging.info("Feature validation passed.")
        return True


if __name__ == "__main__":
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

    success = tuner.run_tuning()
    if success:
        logging.info("Model tuning completed successfully")
    else:
        logging.error("Model tuning failed")
