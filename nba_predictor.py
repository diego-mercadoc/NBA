import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, VotingClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, log_loss, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgb
import optuna
import logging
import joblib
from datetime import datetime, timedelta
from sklearn.exceptions import UndefinedMetricWarning, ConvergenceWarning
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

class NBAPredictor:
    """
    ML-based predictor for NBA games using historical data.
    
    Features:
    - Predicts game winners (moneyline)
    - Predicts point spreads
    - Predicts game totals (over/under)
    - Generates confidence scores for predictions
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.moneyline_model = None
        self.spread_model = None
        self.totals_model = None
        self.feature_columns = [
            # Core game stats
            'Home_Points_Scored_Roll5', 'Home_Points_Allowed_Roll5', 'Home_Point_Diff_Roll5',
            'Away_Points_Scored_Roll5', 'Away_Points_Allowed_Roll5', 'Away_Point_Diff_Roll5',
            'Home_Streak', 'Away_Streak', 'Home_Rest_Days', 'Away_Rest_Days',
            'Home_Win_Rate', 'Away_Win_Rate',
            # Shooting efficiency
            'Home_eFG_Pct', 'Away_eFG_Pct',
            # Advanced metrics
            '3pt_volatility', 'pace_adjusted_offense', 'pace_adjusted_defense',
            'Win_Rate_Diff', 'Point_Diff_Ratio',
            # New features
            'Rest_Day_Differential', 'Streak_Momentum',
            'Opponent_Strength_Home', 'Opponent_Strength_Away',
            'Recent_Form_Home', 'Recent_Form_Away',
            'Home_Road_Differential'
        ]
        
        # Initialize base models (parameters will be optimized)
        self.rf_classifier = None
        self.lr_classifier = None
        self.svm_classifier = None
        self.xgb_classifier = None
        self.lgb_classifier = None
    
        # Set up cross-validation
        self.cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    def optimize_hyperparameters(self, X_train, y_train, model_type='moneyline'):
        """Optimize hyperparameters using Optuna"""
        try:
            # Split training data for validation
            X_train_opt, X_val, y_train_opt, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )
            
            def objective_moneyline(trial):
                # Random Forest parameters
                rf_params = {
                    'rf_n_estimators': trial.suggest_int('rf_n_estimators', 100, 500),
                    'rf_max_depth': trial.suggest_int('rf_max_depth', 3, 10),
                    'rf_min_samples_split': trial.suggest_int('rf_min_samples_split', 2, 10),
                    'rf_min_samples_leaf': trial.suggest_int('rf_min_samples_leaf', 1, 4),
                    'rf_class_weight': trial.suggest_categorical('rf_class_weight', [None, 'balanced'])
                }
                
                # XGBoost parameters
                xgb_params = {
                    'xgb_n_estimators': trial.suggest_int('xgb_n_estimators', 100, 500),
                    'xgb_max_depth': trial.suggest_int('xgb_max_depth', 3, 10),
                    'xgb_learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.1, log=True),
                    'xgb_subsample': trial.suggest_float('xgb_subsample', 0.6, 1.0),
                    'xgb_colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.6, 1.0),
                    'xgb_min_child_weight': trial.suggest_int('xgb_min_child_weight', 1, 7),
                    'xgb_gamma': trial.suggest_float('xgb_gamma', 0, 1)
                }
                
                # LightGBM parameters
                lgb_params = {
                    'lgb_n_estimators': trial.suggest_int('lgb_n_estimators', 100, 500),
                    'lgb_max_depth': trial.suggest_int('lgb_max_depth', 3, 10),
                    'lgb_learning_rate': trial.suggest_float('lgb_learning_rate', 0.01, 0.1, log=True),
                    'lgb_subsample': trial.suggest_float('lgb_subsample', 0.6, 1.0),
                    'lgb_colsample_bytree': trial.suggest_float('lgb_colsample_bytree', 0.6, 1.0),
                    'lgb_num_leaves': trial.suggest_int('lgb_num_leaves', 20, 50),
                    'lgb_min_child_samples': trial.suggest_int('lgb_min_child_samples', 5, 50)
                }
                
                # Initialize models with trial parameters
                rf_model = RandomForestClassifier(
                    n_estimators=rf_params['rf_n_estimators'],
                    max_depth=rf_params['rf_max_depth'],
                    min_samples_split=rf_params['rf_min_samples_split'],
                    min_samples_leaf=rf_params['rf_min_samples_leaf'],
                    class_weight=rf_params['rf_class_weight'],
                    random_state=42,
                    n_jobs=-1
                )
                
                xgb_model = XGBClassifier(
                    n_estimators=xgb_params['xgb_n_estimators'],
                    max_depth=xgb_params['xgb_max_depth'],
                    learning_rate=xgb_params['xgb_learning_rate'],
                    subsample=xgb_params['xgb_subsample'],
                    colsample_bytree=xgb_params['xgb_colsample_bytree'],
                    min_child_weight=xgb_params['xgb_min_child_weight'],
                    gamma=xgb_params['xgb_gamma'],
                    random_state=42,
                    enable_categorical=False,
                    use_label_encoder=False,
                    eval_metric='logloss'
                )
                
                lgb_model = LGBMClassifier(
                    n_estimators=lgb_params['lgb_n_estimators'],
                    max_depth=lgb_params['lgb_max_depth'],
                    learning_rate=lgb_params['lgb_learning_rate'],
                    subsample=lgb_params['lgb_subsample'],
                    colsample_bytree=lgb_params['lgb_colsample_bytree'],
                    num_leaves=lgb_params['lgb_num_leaves'],
                    min_child_samples=lgb_params['lgb_min_child_samples'],
                    random_state=42,
                    n_jobs=-1
                )
                
                # Train and evaluate models
                rf_model.fit(X_train_opt, y_train_opt)
                xgb_model.fit(X_train_opt, y_train_opt)
                lgb_model.fit(X_train_opt, y_train_opt)
                
                # Get predictions
                rf_pred = rf_model.predict_proba(X_val)[:, 1]
                xgb_pred = xgb_model.predict_proba(X_val)[:, 1]
                lgb_pred = lgb_model.predict_proba(X_val)[:, 1]
                
                # Simple average ensemble
                ensemble_pred = (rf_pred + xgb_pred + lgb_pred) / 3
                
                # Calculate metrics
                accuracy = accuracy_score(y_val, (ensemble_pred >= 0.5).astype(int))
                brier = brier_score_loss(y_val, ensemble_pred)
                ll = log_loss(y_val, np.column_stack((1-ensemble_pred, ensemble_pred)))
                
                # Combine metrics (higher is better)
                score = accuracy - 0.5*brier - 0.5*ll
                
                return score
            
            # Create and run study
            study = optuna.create_study(direction='maximize')
            study.optimize(objective_moneyline, n_trials=50)
            
            # Get best parameters
            best_params = study.best_params
            
            # Add model type prefix to parameters
            params = {}
            for key, value in best_params.items():
                params[key] = value
            
            return params
            
        except Exception as e:
            logging.error(f"Error in optimize_hyperparameters: {str(e)}")
            raise
    
    def prepare_features(self, df):
        """Prepare features for model training"""
        try:
            # Create a copy to avoid modifying original data
            X = df.copy()
            
            # Drop unnecessary columns
            drop_cols = ['Team_home', 'Team_away']
            X = X.drop(columns=[col for col in drop_cols if col in X.columns])
            
            # Calculate rolling stats for both teams
            for team_type in ['Home', 'Away']:
                # Point differential rolling stats with min_periods
                X[f'{team_type}_Point_Diff_Roll5'] = (X[f'{team_type}_Points'].rolling(5, min_periods=1).mean() - 
                                                    X[f'{team_type}_Points_Allowed'].rolling(5, min_periods=1).mean())
                X[f'{team_type}_Point_Diff_Roll10'] = (X[f'{team_type}_Points'].rolling(10, min_periods=1).mean() - 
                                                     X[f'{team_type}_Points_Allowed'].rolling(10, min_periods=1).mean())
                
                # Shooting efficiency with min_periods
                X[f'{team_type}_eFG_Roll5'] = X[f'{team_type}_eFG_Pct'].rolling(5, min_periods=1).mean()
                X[f'{team_type}_eFG_Roll10'] = X[f'{team_type}_eFG_Pct'].rolling(10, min_periods=1).mean()
                
                # Advanced metrics
                X[f'{team_type}_Net_Rating'] = X[f'{team_type}_ORtg'] - X[f'{team_type}_DRtg']
                X[f'{team_type}_Pace_Adj_Net'] = X[f'{team_type}_Net_Rating'] * (X[f'{team_type}_Pace'] / 100)
                
                # Recent form (last 3 games with exponential weighting)
                recent_games = X[f'{team_type}_Point_Diff_Roll5'].rolling(3, min_periods=1)
                weights = np.array([0.5, 0.3, 0.2])
                X[f'Recent_Form_{team_type}'] = recent_games.apply(
                    lambda x: np.sum(x[-3:] * weights[:len(x[-3:])])
                )
                
                # Opponent strength (weighted average of opponent net ratings)
                opp_type = 'Away' if team_type == 'Home' else 'Home'
                X[f'Opponent_Strength_{team_type}'] = (
                    X[f'{opp_type}_Net_Rating'].rolling(5, min_periods=1).mean()
                )
            
            # Create interaction features
            X['Streak_Form_Interaction'] = X['Home_Streak'] * X.get('Home_Form', 1) - X['Away_Streak'] * X.get('Away_Form', 1)
            X['eFG_Rest_Interaction'] = X['Home_eFG_Roll5'] * (X['Home_Rest_Days'] + 1) - X['Away_eFG_Roll5'] * (X['Away_Rest_Days'] + 1)
            X['Pace_Mismatch'] = abs(X['Home_Pace'] - X['Away_Pace'])
            X['Rating_Momentum'] = ((X['Home_Point_Diff_Roll5'] - X['Home_Point_Diff_Roll10']) - 
                                  (X['Away_Point_Diff_Roll5'] - X['Away_Point_Diff_Roll10']))
            
            # New differential features
            X['Rest_Day_Differential'] = X['Home_Rest_Days'] - X['Away_Rest_Days']
            X['Streak_Momentum'] = np.tanh(X['Home_Streak'] / 5.0) - np.tanh(X['Away_Streak'] / 5.0)
            X['Home_Road_Differential'] = (
                X['Home_Point_Diff_Roll5'] + X['Away_Point_Diff_Roll5'].abs()
            ) / 2
            
            # Fill NaN values with medians for each column
            numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
            for col in numeric_cols:
                median_val = X[col].median()
                if pd.isna(median_val):  # If median is NaN, use 0 as fallback
                    X[col] = X[col].fillna(0)
                else:
                    X[col] = X[col].fillna(median_val)
            
            # Clip values to prevent outliers
            X['3pt_volatility'] = np.clip(X['3pt_volatility'], -1, 1)
            X['pace_adjusted_offense'] = np.clip(X['pace_adjusted_offense'], 50, 150)
            X['pace_adjusted_defense'] = np.clip(X['pace_adjusted_defense'], 50, 150)
            
            # Validate feature ranges
            assert X['3pt_volatility'].between(-1, 1).all(), "3pt_volatility out of range"
            assert X['pace_adjusted_offense'].between(50, 150).all(), "pace_adjusted_offense out of range"
            assert X['pace_adjusted_defense'].between(50, 150).all(), "pace_adjusted_defense out of range"
            
            # Ensure all required features are present
            missing_features = set(self.feature_columns) - set(X.columns)
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")
            
            # Return only the required features
            return X[self.feature_columns]
            
        except Exception as e:
            logging.error(f"Error in prepare_features: {str(e)}")
            raise
    
    def prepare_labels(self, games_df):
        """Prepare labels for different prediction types"""
        played_games = games_df.dropna(subset=['Home_Points', 'Away_Points'])
        
        y_moneyline = (played_games['Home_Points'] > played_games['Away_Points']).astype(int)
        y_spread = played_games['Home_Points'] - played_games['Away_Points']
        y_totals = played_games['Home_Points'] + played_games['Away_Points']
        
        return y_moneyline, y_spread, y_totals
    
    def train_models(self, df, test_size=0.2):
        """Train models using the provided data"""
        try:
            # Prepare features and labels
            X = self.prepare_features(df)
            y = (df['Home_Points'] > df['Away_Points']).astype(int)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
        
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(self.scaler.fit_transform(X_train), columns=X_train.columns)
            X_test_scaled = pd.DataFrame(self.scaler.transform(X_test), columns=X_test.columns)

            # Initialize models with optimized hyperparameters
            self.rf_model = RandomForestClassifier(
                n_estimators=500,
                max_depth=8,
                min_samples_split=4,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )

            self.xgb_model = XGBClassifier(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                gamma=0.1,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )

            self.lgb_model = LGBMClassifier(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                num_leaves=40,
                min_child_samples=20,
                random_state=42,
                n_jobs=-1,
                silent=True
            )

            # Train models with cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            # Initialize arrays to store CV predictions
            rf_cv_preds = np.zeros(len(X_train))
            xgb_cv_preds = np.zeros(len(X_train))
            lgb_cv_preds = np.zeros(len(X_train))
            
            # Perform cross-validation
            for train_idx, val_idx in cv.split(X_train_scaled, y_train):
                X_cv_train, X_cv_val = X_train_scaled.iloc[train_idx], X_train_scaled.iloc[val_idx]
                y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                # Train models on fold
                self.rf_model.fit(X_cv_train, y_cv_train)
                self.xgb_model.fit(X_cv_train, y_cv_train, eval_set=[(X_cv_val, y_cv_val)], verbose=False)
                self.lgb_model.fit(X_cv_train, y_cv_train, eval_set=(X_cv_val, y_cv_val), verbose=False)
                
                # Store predictions for validation fold
                rf_cv_preds[val_idx] = self.rf_model.predict_proba(X_cv_val)[:, 1]
                xgb_cv_preds[val_idx] = self.xgb_model.predict_proba(X_cv_val)[:, 1]
                lgb_cv_preds[val_idx] = self.lgb_model.predict_proba(X_cv_val)[:, 1]
            
            # Optimize ensemble weights using cross-validation performance
            def objective(weights):
                # Normalize weights to sum to 1
                weights = np.array(weights)
                weights = weights / np.sum(weights)
                
                # Combine predictions with weights
                ensemble_preds = (
                    weights[0] * rf_cv_preds +
                    weights[1] * xgb_cv_preds +
                    weights[2] * lgb_cv_preds
                )
                
                # Calculate metrics
                accuracy = accuracy_score(y_train, (ensemble_preds >= 0.5).astype(int))
                brier = brier_score_loss(y_train, ensemble_preds)
                ll = log_loss(y_train, np.column_stack((1-ensemble_preds, ensemble_preds)))
                
                # Combined score (higher is better)
                return -(0.4 * accuracy - 0.3 * brier - 0.3 * ll)
            
            # Find optimal weights
            from scipy.optimize import minimize
            initial_weights = [0.4, 0.3, 0.3]
            bounds = [(0, 1)] * 3
            constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
            result = minimize(objective, initial_weights, bounds=bounds, constraints=constraints)
            optimal_weights = result.x
            
            # Train final models on full training set
            self.rf_model.fit(X_train_scaled, y_train)
            self.xgb_model.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)], verbose=False)
            self.lgb_model.fit(X_train_scaled, y_train, eval_set=(X_test_scaled, y_test), verbose=False)
            
            # Get predictions with optimal weights
            rf_pred = self.rf_model.predict_proba(X_test_scaled)[:, 1]
            xgb_pred = self.xgb_model.predict_proba(X_test_scaled)[:, 1]
            lgb_pred = self.lgb_model.predict_proba(X_test_scaled)[:, 1]
            
            ensemble_probs = (
                optimal_weights[0] * rf_pred +
                optimal_weights[1] * xgb_pred +
                optimal_weights[2] * lgb_pred
            )
            ensemble_preds = (ensemble_probs > 0.5).astype(int)
            
            # Store optimal weights
            self.ensemble_weights = optimal_weights
            
            # Calculate metrics
            metrics = {
                'moneyline_accuracy': accuracy_score(y_test, ensemble_preds),
                'moneyline_log_loss': log_loss(y_test, ensemble_probs),
                'moneyline_brier': brier_score_loss(y_test, ensemble_probs),
                'ensemble_weights': optimal_weights.tolist(),
                'spread_rmse': 0.0,  # Placeholder
                'totals_rmse': 0.0   # Placeholder
            }
            
            logging.info(f"Training metrics: {metrics}")
            logging.info(f"Optimal ensemble weights: RF={optimal_weights[0]:.3f}, XGB={optimal_weights[1]:.3f}, LGB={optimal_weights[2]:.3f}")
            
            return metrics

        except Exception as e:
            logging.error(f"Model training failed: {str(e)}")
            raise
    
    def save_models(self):
        """Save trained models and scaler"""
        joblib.dump(self.moneyline_model, 'models/moneyline_model.joblib')
        joblib.dump(self.spread_model, 'models/spread_model.joblib')
        joblib.dump(self.totals_model, 'models/totals_model.joblib')
        joblib.dump(self.scaler, 'models/scaler.joblib')
        logging.info("Models saved successfully")
    
    def load_models(self):
        """Load trained models and scaler"""
        try:
            self.moneyline_model = joblib.load('models/moneyline_model.joblib')
            self.spread_model = joblib.load('models/spread_model.joblib')
            self.totals_model = joblib.load('models/totals_model.joblib')
            self.scaler = joblib.load('models/scaler.joblib')
            logging.info("Models loaded successfully")
            return True
        except Exception as e:
            logging.error(f"Error loading models: {str(e)}")
            return False
    
    def predict_games(self, games_df):
        """Generate predictions for the provided games"""
        try:
            # Prepare features
            X = self.prepare_features(games_df)
            
            # Scale features if scaler exists
            if hasattr(self, 'scaler'):
                X_scaled = pd.DataFrame(self.scaler.transform(X), columns=X.columns)
            else:
                X_scaled = X
            
            # Get predictions from each model
            rf_pred = self.rf_model.predict_proba(X_scaled)[:, 1]
            xgb_pred = self.xgb_model.predict_proba(X_scaled)[:, 1]
            lgb_pred = self.lgb_model.predict_proba(X_scaled)[:, 1]
            
            # Use optimized ensemble weights if available
            if hasattr(self, 'ensemble_weights'):
                ensemble_probs = (
                    self.ensemble_weights[0] * rf_pred +
                    self.ensemble_weights[1] * xgb_pred +
                    self.ensemble_weights[2] * lgb_pred
                )
            else:
                # Fallback to default weights
                ensemble_probs = 0.4 * rf_pred + 0.3 * xgb_pred + 0.3 * lgb_pred
            
            ensemble_preds = (ensemble_probs > 0.5).astype(int)
            
            # Calculate confidence scores
            confidence_scores = np.abs(ensemble_probs - 0.5) * 2  # Scale to [0, 1]
            
            # Create prediction DataFrame
            predictions = pd.DataFrame({
                'Home_Team': games_df['Home_Team'],
                'Away_Team': games_df['Away_Team'],
                'Date': games_df['Date'],
                'Home_Win_Probability': ensemble_probs,
                'Away_Win_Probability': 1 - ensemble_probs,
                'Moneyline_Pick': np.where(ensemble_probs > 0.5, games_df['Home_Team'], games_df['Away_Team']),
                'Prediction': ensemble_preds,
                'Confidence': confidence_scores
            })
            
            return predictions
            
        except Exception as e:
            logging.error(f"Prediction failed: {str(e)}")
            raise
    
    def _format_prediction(self, row):
        """Format prediction row into readable string"""
        home_prob = row['Home_Win_Prob'] * 100
        away_prob = row['Away_Win_Prob'] * 100
        spread = row['Predicted_Spread']
        total = row['Predicted_Total']
        
        pick = f"{row['Home_Team']} ML" if row['Moneyline_Pick'] else f"{row['Away_Team']} ML"
        confidence = row['Moneyline_Confidence'] * 100
        
        # Format quarter and half predictions
        first_half_total = row['First_Half_Total']
        first_half_spread = row['First_Half_Spread']
        first_q_total = row['First_Quarter_Total']
        first_q_spread = row['First_Quarter_Spread']
        
        return (
            f"Game: {row['Away_Team']} @ {row['Home_Team']}\n"
            f"Moneyline: {pick} ({confidence:.1f}% confidence)\n"
            f"Win Probabilities: {row['Home_Team']}: {home_prob:.1f}%, "
            f"{row['Away_Team']}: {away_prob:.1f}%\n"
            f"Full Game:\n"
            f"  - Spread: {row['Home_Team']} {spread:+.1f}\n"
            f"  - Total: {total:.1f}\n"
            f"First Half:\n"
            f"  - Spread: {row['Home_Team']} {first_half_spread:+.1f}\n"
            f"  - Total: {first_half_total:.1f}\n"
            f"First Quarter:\n"
            f"  - Spread: {row['Home_Team']} {first_q_spread:+.1f}\n"
            f"  - Total: {first_q_total:.1f}"
        )
    
    def get_best_bets(self, predictions_df, confidence_threshold=0.90, min_value_rating=0.70):
        """Filter and return only the highest confidence bets with enhanced value ratings"""
        best_bets = []
        
        # Moneyline bets with stricter confidence requirements
        ml_bets = predictions_df[
            (predictions_df['Moneyline_Confidence'] > confidence_threshold)
        ].copy()
        
        for _, game in ml_bets.iterrows():
            # Enhanced value rating calculation
            value_rating = self._calculate_value_rating(game)
            if value_rating > min_value_rating:
                bet = {
                    'Game': f"{game['Away_Team']} @ {game['Home_Team']}",
                    'Bet_Type': 'Moneyline',
                    'Prediction': f"{game['Home_Team']} ML" if game['Moneyline_Pick'] else f"{game['Away_Team']} ML",
                    'Confidence': game['Moneyline_Confidence'],
                    'Value_Rating': value_rating
                }
                best_bets.append(bet)
            
            # First Half total with enhanced validation
            if abs(game['First_Half_Total'] - (game['Predicted_Total'] * 0.52)) < 3:  # Tighter threshold
                half_confidence = min(0.95, game['Moneyline_Confidence'] * 1.1)  # Cap at 95%
                half_value = self._calculate_half_value(game)
                if half_confidence > confidence_threshold and half_value > min_value_rating:
                    bet = {
                        'Game': f"{game['Away_Team']} @ {game['Home_Team']}",
                        'Bet_Type': 'First Half Total',
                        'Prediction': f"Over {game['First_Half_Total']:.1f}",
                        'Confidence': half_confidence,
                        'Value_Rating': half_value
                    }
                    best_bets.append(bet)
            
            # First Quarter total with enhanced validation
            if abs(game['First_Quarter_Total'] - (game['Predicted_Total'] * 0.24)) < 1.5:  # Tighter threshold
                quarter_confidence = min(0.95, game['Moneyline_Confidence'] * 1.05)  # Cap at 95%
                quarter_value = self._calculate_quarter_value(game)
                if quarter_confidence > confidence_threshold and quarter_value > min_value_rating:
                    bet = {
                        'Game': f"{game['Away_Team']} @ {game['Home_Team']}",
                        'Bet_Type': 'First Quarter Total',
                        'Prediction': f"Over {game['First_Quarter_Total']:.1f}",
                        'Confidence': quarter_confidence,
                        'Value_Rating': quarter_value
                    }
                    best_bets.append(bet)
        
        return pd.DataFrame(best_bets)
    
    def _calculate_value_rating(self, row):
        """Enhanced value rating calculation with more factors and stricter thresholds"""
        # Base value from probability margin
        prob_margin = abs(row['Home_Win_Prob'] - row['Away_Win_Prob'])
        value_rating = row['Moneyline_Confidence'] * prob_margin
        
        # Enhanced adjustments with stricter thresholds
        if all(col in row for col in ['Home_Point_Diff_Roll5', 'Away_Point_Diff_Roll5', 'Home_Rest_Days', 'Away_Rest_Days']):
            # Form factor with exponential weighting
            form_diff = abs(row['Home_Point_Diff_Roll5'] - row['Away_Point_Diff_Roll5'])
            form_factor = np.tanh(form_diff / 10)  # Normalized between -1 and 1
            
            # Rest advantage with diminishing returns
            rest_diff = abs(row['Home_Rest_Days'] - row['Away_Rest_Days'])
            rest_factor = np.tanh(rest_diff / 3)  # Normalized between -1 and 1
            
            # Streak impact with momentum consideration
            streak_diff = abs(row.get('Home_Streak', 0) - row.get('Away_Streak', 0))
            streak_factor = np.tanh(streak_diff / 5)  # Normalized between -1 and 1
            
            # Recent performance weight (last 3 games)
            recent_weight = 1.2 if form_factor > 0.5 else 1.0
            
            # Combine factors with weighted importance
            value_rating *= (1 + 0.3*form_factor + 0.2*rest_factor + 0.2*streak_factor) * recent_weight
        
        # Additional confidence boost for extreme differentials
        if prob_margin > 0.4:  # 40% probability difference
            value_rating *= 1.1
        
        # Cap the value rating at 1.0
        return min(value_rating, 1.0)
    
    def _calculate_half_value(self, row):
        """Calculate value rating for first half totals"""
        base_value = self._calculate_value_rating(row)
        
        # Adjust based on first half scoring patterns
        if 'First_Half_Total' in row and 'Predicted_Total' in row:
            half_ratio = row['First_Half_Total'] / row['Predicted_Total']
            if 0.50 <= half_ratio <= 0.54:  # Ideal range
                base_value *= 1.1
            elif 0.48 <= half_ratio <= 0.56:  # Acceptable range
                base_value *= 1.0
            else:
                base_value *= 0.8
        
        return min(base_value * 0.95, 1.0)  # Slightly lower confidence for halves
    
    def _calculate_quarter_value(self, row):
        """Calculate value rating for first quarter totals"""
        base_value = self._calculate_value_rating(row)
        
        # Adjust based on first quarter scoring patterns
        if 'First_Quarter_Total' in row and 'Predicted_Total' in row:
            quarter_ratio = row['First_Quarter_Total'] / row['Predicted_Total']
            if 0.23 <= quarter_ratio <= 0.25:  # Ideal range
                base_value *= 1.1
            elif 0.22 <= quarter_ratio <= 0.26:  # Acceptable range
                base_value *= 1.0
            else:
                base_value *= 0.8
        
        return min(base_value * 0.90, 1.0)  # Lower confidence for quarters 
    
    def _cross_val_score(self, X_train, y_train, model):
        """Calculate cross-validation score for a model."""
        if isinstance(model, xgb.XGBClassifier):
            model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train)],
                verbose=False
            )
        elif isinstance(model, lgb.LGBMClassifier):
            model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train)],
                callbacks=[lgb.early_stopping(10)]
            )
        else:
            model.fit(X_train, y_train)
        
        proba = model.predict_proba(X_train)
        accuracy = accuracy_score(y_train, proba[:, 1] > 0.5)
        log_loss_val = log_loss(y_train, proba)
        brier = brier_score_loss(y_train, proba[:, 1])
        
        # Combined metric
        return 0.4 * accuracy - 0.3 * log_loss_val - 0.3 * brier 

    def predict(self, X):
        """Make predictions using the trained ensemble.
        
        Args:
            X (pd.DataFrame): Features to predict on
            
        Returns:
            np.ndarray: Binary predictions
        """
        try:
            # Scale features if scaler exists
            if self.scaler is not None:
                X_scaled = pd.DataFrame(self.scaler.transform(X), columns=X.columns)
            else:
                X_scaled = X
            
            # Get predictions from each model
            rf_pred = self.rf_model.predict_proba(X_scaled)[:, 1]
            xgb_pred = self.xgb_model.predict_proba(X_scaled)[:, 1]
            lgb_pred = self.lgb_model.predict_proba(X_scaled)[:, 1]
            
            # Weighted average
            ensemble_pred = (0.4 * rf_pred + 0.3 * xgb_pred + 0.3 * lgb_pred)
            return (ensemble_pred > 0.5).astype(int)
            
        except Exception as e:
            logging.error(f"Error in predict: {str(e)}")
            raise

    def predict_proba(self, X):
        """Get probability predictions using the trained ensemble.
        
        Args:
            X (pd.DataFrame): Features to predict on
            
        Returns:
            np.ndarray: Probability predictions
        """
        try:
            # Scale features if scaler exists
            if self.scaler is not None:
                X_scaled = pd.DataFrame(self.scaler.transform(X), columns=X.columns)
            else:
                X_scaled = X
            
            # Get predictions from each model
            rf_pred = self.rf_model.predict_proba(X_scaled)[:, 1]
            xgb_pred = self.xgb_model.predict_proba(X_scaled)[:, 1]
            lgb_pred = self.lgb_model.predict_proba(X_scaled)[:, 1]
            
            # Weighted average
            return 0.4 * rf_pred + 0.3 * xgb_pred + 0.3 * lgb_pred
            
        except Exception as e:
            logging.error(f"Error in predict_proba: {str(e)}")
            raise 