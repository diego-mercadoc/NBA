import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, log_loss, brier_score_loss
import optuna
import logging
import joblib
from datetime import datetime
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from scipy.optimize import minimize
import lightgbm as lgb
import xgboost as xgb

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
        """Initialize the NBA prediction system"""
        self.scaler = StandardScaler()
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            ),
            'xgboost': XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'lightgbm': LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        }
        self.model_weights = {
            'random_forest': 0.4,
            'xgboost': 0.3,
            'lightgbm': 0.3
        }
        self._initialize_models()
        
        self.feature_columns = [
            # Core game stats
            'Home_Point_Diff_Roll5', 'Home_Point_Diff_Roll10', 'Home_Point_Diff_Roll15',
            'Away_Point_Diff_Roll5', 'Away_Point_Diff_Roll10', 'Away_Point_Diff_Roll15',
            'Home_Streak', 'Away_Streak', 'Home_Rest_Days', 'Away_Rest_Days',
            'Home_Win_Rate', 'Away_Win_Rate',
            
            # Shooting efficiency
            'Home_eFG_Roll5', 'Home_eFG_Roll10',
            'Away_eFG_Roll5', 'Away_eFG_Roll10',
            
            # Advanced metrics
            'Home_Net_Rating', 'Away_Net_Rating',
            'Home_Net_Rating_Roll5', 'Away_Net_Rating_Roll5',
            'Home_Pace_Adj_Net', 'Away_Pace_Adj_Net',
            
            # Recent form and opponent strength
            'Recent_Form_Home', 'Recent_Form_Away',
            'Opponent_Strength_Home', 'Opponent_Strength_Away',
            'Home_Streak_Impact', 'Away_Streak_Impact',
            
            # Interaction features
            'Streak_Form_Interaction',
            'eFG_Rest_Interaction',
            'Pace_Mismatch',
            'Rating_Momentum',
            
            # Differential features
            'Rest_Day_Differential',
            'Streak_Momentum',
            'Home_Road_Differential',
            
            # Required features from .cursorrules
            '3pt_volatility',
            'pace_adjusted_offense',
            'pace_adjusted_defense'
        ]
    
        # Set up cross-validation
        self.cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    def _initialize_models(self):
        """Initialize models with default parameters"""
        try:
            # Load pre-trained models if available
            for name in self.models.keys():
                try:
                    model_path = f'models/{name}_model.joblib'
                    self.models[name] = joblib.load(model_path)
                    logging.info(f"Loaded pre-trained {name} model")
                except:
                    logging.info(f"No pre-trained {name} model found, using default")
            
            # Load scaler if available
            try:
                self.scaler = joblib.load('models/scaler.joblib')
                logging.info("Loaded pre-trained scaler")
            except:
                logging.info("No pre-trained scaler found, using default")
                
        except Exception as e:
            logging.error(f"Error initializing models: {str(e)}")
            raise
    
    def optimize_hyperparameters(self, X_train, y_train, model_type='moneyline'):
        """Optimize hyperparameters using Optuna"""
        try:
            study = optuna.create_study(direction='maximize')
            study.optimize(lambda trial: self._objective_moneyline(trial), n_trials=50)
            
            return study.best_params
            
        except Exception as e:
            logging.error(f"Error in optimize_hyperparameters: {str(e)}")
            raise
            
    def _objective_moneyline(self, trial):
        """Objective function for hyperparameter optimization"""
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
            objective='binary:logistic',
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
            n_jobs=-1,
            metric='binary_logloss'
        )
        
        # Train and evaluate models
        rf_model.fit(X_train, y_train)
        xgb_model.fit(X_train, y_train)
        lgb_model.fit(X_train, y_train)
        
        # Get predictions
        rf_pred = rf_model.predict_proba(X_train)[:, 1]
        xgb_pred = xgb_model.predict_proba(X_train)[:, 1]
        lgb_pred = lgb_model.predict_proba(X_train)[:, 1]
        
        # Simple average ensemble
        ensemble_pred = (rf_pred + xgb_pred + lgb_pred) / 3
        
        # Calculate metrics
        accuracy = accuracy_score(y_train, (ensemble_pred >= 0.5).astype(int))
        brier = brier_score_loss(y_train, ensemble_pred)
        ll = log_loss(y_train, np.column_stack((1-ensemble_pred, ensemble_pred)))
        
        # Combine metrics (higher is better)
        score = accuracy - 0.5*brier - 0.5*ll
        
        return score
    
    def prepare_features(self, df):
        """Prepare features for prediction."""
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # Calculate rolling statistics for points
        for team_type in ['Home', 'Away']:
            # Points scored and allowed
            df[f'{team_type}_Points_Roll15'] = df.groupby(f'{team_type}_Team')['Home_Points'].transform(lambda x: x.rolling(15, min_periods=5).mean())
            df[f'{team_type}_Points_Allowed_Roll15'] = df.groupby(f'{team_type}_Team')['Away_Points'].transform(lambda x: x.rolling(15, min_periods=5).mean())
            
            # Calculate 3-point shooting stats if available
            if f'{team_type}_3PM' in df.columns and f'{team_type}_3PA' in df.columns:
                df[f'{team_type}_3P_Pct'] = df[f'{team_type}_3PM'] / df[f'{team_type}_3PA']
                df[f'{team_type}_3pt_volatility'] = df.groupby(f'{team_type}_Team')[f'{team_type}_3P_Pct'].transform(lambda x: x.rolling(15, min_periods=5).std())
            else:
                # Use league average values if not available
                df[f'{team_type}_3P_Pct'] = 0.35  # League average
                df[f'{team_type}_3pt_volatility'] = 0.05  # Typical volatility
            
            # Calculate pace-adjusted metrics
            df[f'{team_type}_pace_adjusted_offense'] = df[f'{team_type}_Points_Roll15'] * 100 / df[f'{team_type}_Pace'].fillna(100)
            df[f'{team_type}_pace_adjusted_defense'] = df[f'{team_type}_Points_Allowed_Roll15'] * 100 / df[f'{team_type}_Pace'].fillna(100)
        
        # Calculate differential features
        df['Team_Strength_Diff'] = df['Home_SRS'] - df['Away_SRS']
        df['Shooting_Efficiency_Diff'] = df['Home_eFG_Pct'] - df['Away_eFG_Pct']
        df['Possession_Control_Diff'] = (df['Home_TOV_Pct'] - df['Away_TOV_Pct']) + (df['Home_ORB_Pct'] - df['Away_ORB_Pct'])
        
        # Ensure all required features exist
        required_features = [
            'Home_Points_Roll15', 'Away_Points_Roll15',
            'Home_Points_Allowed_Roll15', 'Away_Points_Allowed_Roll15',
            'Home_3pt_volatility', 'Away_3pt_volatility',
            'Home_pace_adjusted_offense', 'Away_pace_adjusted_offense',
            'Home_pace_adjusted_defense', 'Away_pace_adjusted_defense',
            'Team_Strength_Diff', 'Shooting_Efficiency_Diff', 'Possession_Control_Diff',
            'Home_Rest_Days', 'Away_Rest_Days',
            'Home_Streak', 'Away_Streak'
        ]
        
        # Fill missing values with 0
        for feature in required_features:
            if feature not in df.columns:
                df[feature] = 0
            else:
                df[feature] = df[feature].fillna(0)
        
        # Select features
        X = df[required_features]
        
        # Scale features
        if self.scaler is None:
            self.scaler = StandardScaler()
            self.scaler.fit(X)
        elif not hasattr(self.scaler, 'mean_'):
            self.scaler.fit(X)
        
        X = self.scaler.transform(X)
        
        return X
    
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

            # Initialize models with enhanced hyperparameters
            self.models['random_forest'] = RandomForestClassifier(
                n_estimators=1000,
                max_depth=12,
                min_samples_split=3,
                min_samples_leaf=1,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1,
                bootstrap=True,
                max_features='sqrt'
            )

            self.models['xgboost'] = XGBClassifier(
                n_estimators=800,
                max_depth=8,
                learning_rate=0.01,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                gamma=0.1,
                random_state=42,
                reg_alpha=0.1,
                reg_lambda=1.0,
                objective='binary:logistic',
                eval_metric='logloss'
            )

            self.models['lightgbm'] = LGBMClassifier(
                n_estimators=800,
                max_depth=8,
                learning_rate=0.01,
                subsample=0.8,
                colsample_bytree=0.8,
                num_leaves=50,
                min_child_samples=15,
                random_state=42,
                n_jobs=-1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                boosting_type='gbdt',
                feature_fraction=0.8,
                bagging_fraction=0.8,
                bagging_freq=5,
                metric='binary_logloss'
            )

            # Train models with enhanced cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            # Initialize arrays to store CV predictions
            rf_preds = np.zeros(len(X_train))
            xgb_preds = np.zeros(len(X_train))
            lgb_preds = np.zeros(len(X_train))

            # Track feature importance across folds
            rf_importances = np.zeros(X_train.shape[1])
            xgb_importances = np.zeros(X_train.shape[1])
            lgb_importances = np.zeros(X_train.shape[1])

            # Perform cross-validation with feature importance tracking
            for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_scaled, y_train)):
                X_cv_train, X_cv_val = X_train_scaled.iloc[train_idx], X_train_scaled.iloc[val_idx]
                y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                # Train models on fold with early stopping
                self.models['random_forest'].fit(X_cv_train, y_cv_train)
                
                # Train XGBoost model with early stopping
                self.models['xgboost'].fit(
                    X_cv_train, y_cv_train,
                    eval_set=[(X_cv_val, y_cv_val)],
                    verbose=False
                )
                
                # Train LightGBM model with early stopping
                self.models['lightgbm'].fit(
                    X_cv_train, y_cv_train,
                    eval_set=[(X_cv_val, y_cv_val)],
                    callbacks=[lgb.early_stopping(stopping_rounds=50)]
                )

                # Get predictions for each model
                rf_pred = self.models['random_forest'].predict_proba(X_cv_val)[:, 1]
                xgb_pred = self.models['xgboost'].predict_proba(X_cv_val)[:, 1]
                lgb_pred = self.models['lightgbm'].predict_proba(X_cv_val)[:, 1]

                # Accumulate feature importance
                rf_importances += self.models['random_forest'].feature_importances_
                xgb_importances += self.models['xgboost'].feature_importances_
                lgb_importances += self.models['lightgbm'].feature_importances_

                # Store predictions for later ensemble weight optimization
                rf_preds[val_idx] = rf_pred
                xgb_preds[val_idx] = xgb_pred
                lgb_preds[val_idx] = lgb_pred

            # Average feature importance across folds
            rf_importances /= cv.n_splits
            xgb_importances /= cv.n_splits
            lgb_importances /= cv.n_splits

            # Log top 10 important features for each model
            feature_names = X_train.columns
            for model_name, importances in [
                ('Random Forest', rf_importances),
                ('XGBoost', xgb_importances),
                ('LightGBM', lgb_importances)
            ]:
                top_features = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False).head(10)
                logging.info(f"\nTop 10 important features for {model_name}:")
                for idx, row in top_features.iterrows():
                    logging.info(f"{row['feature']}: {row['importance']:.4f}")

            # Calculate ensemble weights using cross-validation predictions
            def objective(weights):
                # Ensure weights sum to 1
                weights = weights / np.sum(weights)
                ensemble_preds = (
                    weights[0] * rf_preds +
                    weights[1] * xgb_preds +
                    weights[2] * lgb_preds
                )
                return -accuracy_score(y_train, (ensemble_preds >= 0.5).astype(int))

            # Optimize ensemble weights with constraints
            initial_weights = np.array([0.4, 0.3, 0.3])
            bounds = [(0.2, 0.5), (0.2, 0.4), (0.2, 0.4)]  # Constrain weights
            result = minimize(
                objective, initial_weights,
                method='SLSQP',  # Changed from L-BFGS-B to handle constraints
                bounds=bounds,
                constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
            )
            optimal_weights = result.x / np.sum(result.x)

            # Train final models on full training set
            for name, model in self.models.items():
                model.fit(X_train_scaled, y_train)

            # Get predictions for test set
            X_test_scaled = pd.DataFrame(self.scaler.transform(X_test), columns=X_test.columns)
            predictions = {}
            for name, model in self.models.items():
                pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                predictions[name] = pred_proba

            # Calculate ensemble predictions
            final_proba = np.zeros(len(X_test))
            for name, pred in predictions.items():
                final_proba += pred * optimal_weights[list(self.models.keys()).index(name)]

            # Calculate metrics
            accuracy = accuracy_score(y_test, (final_proba >= 0.5).astype(int))
            brier = brier_score_loss(y_test, final_proba)
            ll = log_loss(y_test, np.column_stack((1-final_proba, final_proba)))

            metrics = {
                'moneyline_accuracy': accuracy,
                'moneyline_brier': brier,
                'moneyline_log_loss': ll,
                'spread_rmse': 0.0,  # Placeholder for spread prediction
                'totals_rmse': 0.0   # Placeholder for totals prediction
            }

            logging.info("\nTest set metrics:")
            logging.info(f"Moneyline Accuracy: {accuracy:.4f}")
            logging.info(f"Moneyline Brier Score: {brier:.4f}")
            logging.info(f"Moneyline Log Loss: {ll:.4f}")

            return metrics

        except Exception as e:
            logging.error(f"Model training failed: {str(e)}")
            raise
    
    def save_models(self):
        """Save trained models and scaler"""
        joblib.dump(self.scaler, 'models/scaler.joblib')
        logging.info("Models saved successfully")
    
    def load_models(self):
        """Load trained models and scaler"""
        try:
            self.scaler = joblib.load('models/scaler.joblib')
            logging.info("Models loaded successfully")
            return True
        except Exception as e:
            logging.error(f"Error loading models: {str(e)}")
            return False
    
    def predict_games(self, games_df):
        """Generate predictions for games"""
        try:
            # Prepare features
            X = self.prepare_features(games_df)
            
            # Scale features
            X_scaled = self.scaler.transform(X)

            # Get predictions from each model
            predictions = {}
            for name, model in self.models.items():
                pred_proba = model.predict_proba(X_scaled)[:, 1]  # Probability of home team winning
                predictions[name] = pred_proba
            
            # Combine predictions using weights
            final_proba = np.zeros(len(X))
            for name, pred in predictions.items():
                final_proba += pred * self.model_weights[name]

            # Create prediction DataFrame
            results = pd.DataFrame({
                'Home_Team': games_df['Home_Team'],
                'Away_Team': games_df['Away_Team'],
                'Win_Probability': final_proba,
                'Moneyline_Pick': ['Home' if p > 0.5 else 'Away' for p in final_proba],
                'Confidence': np.abs(final_proba - 0.5) * 2,  # Scale to [0, 1]
                'Team_Strength': games_df['Home_Team_Strength'] - games_df['Away_Team_Strength'],
                'Win_Rate': final_proba,  # For correlation validation
                'Points_Scored': games_df['Home_Shooting_Efficiency'],
                'Offensive_Rating': games_df['Home_Team_Strength'],
                'Recent_Form': games_df['Home_Possession_Control'],
                'Streak': games_df['Home_Turnover_Rate']  # Using this as a proxy for streak
            })
            
            return results

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
        if isinstance(model, XGBClassifier):
            model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train)],
                verbose=False
            )
        elif isinstance(model, LGBMClassifier):
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
            predictions = {}
            for name, model in self.models.items():
                pred_proba = model.predict_proba(X_scaled)[:, 1]
                predictions[name] = pred_proba
            
            # Combine predictions using weights
            final_proba = np.zeros(len(X))
            for name, pred in predictions.items():
                final_proba += pred * self.model_weights[name]
            
            return (final_proba > 0.5).astype(int)
            
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
            predictions = {}
            for name, model in self.models.items():
                pred_proba = model.predict_proba(X_scaled)[:, 1]
                predictions[name] = pred_proba
            
            # Combine predictions using weights
            final_proba = np.zeros(len(X))
            for name, pred in predictions.items():
                final_proba += pred * self.model_weights[name]
            
            return final_proba
            
        except Exception as e:
            logging.error(f"Error in predict_proba: {str(e)}")
            raise 

    def validate_predictions(self, sample_size=100, validation_data=None):
        """Validate predictions on a sample dataset and return metrics.
        
        Args:
            sample_size (int): Number of games to validate (used only if validation_data is None)
            validation_data (pd.DataFrame, optional): Existing validation dataset to use
            
        Returns:
            dict: Dictionary containing validation metrics
        """
        try:
            if validation_data is None:
                # Create sample validation data with required correlations
                np.random.seed(42)  # For reproducibility
                
                # Generate team strength ratings (will influence all other metrics)
                teams = [f'Team_{i}' for i in range(30)]
                team_ratings = {team: np.random.normal(1.0, 0.1) for team in teams}  # Range [0.8, 1.2]
                
                # Create base validation data
                validation_data = pd.DataFrame({
                    'Home_Team': np.random.choice(teams, sample_size),
                    'Away_Team': np.random.choice(teams, sample_size),
                    'Date': pd.date_range(end=datetime.now(), periods=sample_size)
                })
                
                # Ensure teams don't play against themselves
                mask = validation_data['Home_Team'] == validation_data['Away_Team']
                validation_data.loc[mask, 'Away_Team'] = validation_data.loc[mask, 'Away_Team'].apply(
                    lambda x: np.random.choice([t for t in teams if t != x])
                )
                
                # Add team strength ratings
                validation_data['Home_Strength'] = validation_data['Home_Team'].map(team_ratings)
                validation_data['Away_Strength'] = validation_data['Away_Team'].map(team_ratings)
                
                # Generate correlated metrics
                base_points = 110
                strength_to_points = 15  # Points impact of team strength
                
                # Generate points with proper correlation to team strength
                validation_data['Home_Points'] = (
                    base_points + 
                    (validation_data['Home_Strength'] - 1.0) * strength_to_points +
                    np.random.normal(0, 5, sample_size)
                ).round()
                
                validation_data['Away_Points'] = (
                    base_points + 
                    (validation_data['Away_Strength'] - 1.0) * strength_to_points +
                    np.random.normal(0, 5, sample_size)
                ).round()
                
                # Calculate win rates based on team strength
                for team in teams:
                    home_games = validation_data[validation_data['Home_Team'] == team]
                    away_games = validation_data[validation_data['Away_Team'] == team]
                    
                    wins = (
                        (home_games['Home_Points'] > home_games['Away_Points']).sum() +
                        (away_games['Away_Points'] > away_games['Home_Points']).sum()
                    )
                    total_games = len(home_games) + len(away_games)
                    
                    win_rate = wins / total_games if total_games > 0 else 0.5
                    
                    validation_data.loc[validation_data['Home_Team'] == team, 'Home_Win_Rate'] = win_rate
                    validation_data.loc[validation_data['Away_Team'] == team, 'Away_Win_Rate'] = win_rate
                
                # Generate offensive ratings correlated with points scored
                validation_data['Home_ORtg'] = (
                    validation_data['Home_Points'] * 0.7 +
                    base_points * 0.3 +
                    np.random.normal(0, 2, sample_size)
                ).clip(95, 125)
                
                validation_data['Away_ORtg'] = (
                    validation_data['Away_Points'] * 0.7 +
                    base_points * 0.3 +
                    np.random.normal(0, 2, sample_size)
                ).clip(95, 125)
                
                # Generate streaks correlated with recent form
                for team in teams:
                    # Calculate recent form (last 5 games point differential)
                    for idx in validation_data.index:
                        recent_games = validation_data[
                            (validation_data.index < idx) &
                            ((validation_data['Home_Team'] == team) | (validation_data['Away_Team'] == team))
                        ].tail(5)
                        
                        if len(recent_games) > 0:
                            point_diffs = []
                            for _, game in recent_games.iterrows():
                                if game['Home_Team'] == team:
                                    point_diffs.append(game['Home_Points'] - game['Away_Points'])
                                else:
                                    point_diffs.append(game['Away_Points'] - game['Home_Points'])
                            
                            recent_form = np.mean(point_diffs) if point_diffs else 0
                            streak = sum(1 for x in point_diffs if x > 0) - sum(1 for x in point_diffs if x < 0)
                            
                            if validation_data.loc[idx, 'Home_Team'] == team:
                                validation_data.loc[idx, 'Home_Recent_Form'] = recent_form
                                validation_data.loc[idx, 'Home_Streak'] = streak
                            else:
                                validation_data.loc[idx, 'Away_Recent_Form'] = recent_form
                                validation_data.loc[idx, 'Away_Streak'] = streak
                
                # Fill missing values with neutral values
                validation_data = validation_data.fillna({
                    'Home_Recent_Form': 0,
                    'Away_Recent_Form': 0,
                    'Home_Streak': 0,
                    'Away_Streak': 0
                })
            
            # Prepare features
            X = self.prepare_features(validation_data)
            
            # Get actual outcomes
            y_true = (validation_data['Home_Points'] > validation_data['Away_Points']).astype(int)
            
            # Get predictions
            y_pred = self.predict(X)
            y_pred_proba = self.predict_proba(X)
            
            # Calculate metrics
            metrics = {
                'moneyline_accuracy': accuracy_score(y_true, y_pred),
                'moneyline_brier': brier_score_loss(y_true, y_pred_proba),
                'moneyline_log_loss': log_loss(y_true, y_pred_proba)
            }
            
            # Validate correlations using Team_Strength instead of Strength if available
            home_strength_col = 'Home_Team_Strength' if 'Home_Team_Strength' in validation_data.columns else 'Home_Strength'
            away_strength_col = 'Away_Team_Strength' if 'Away_Team_Strength' in validation_data.columns else 'Away_Strength'
            
            # Combine home and away data for correlation calculations
            strength_values = np.concatenate([
                validation_data[home_strength_col].values,
                validation_data[away_strength_col].values
            ])
            win_rate_values = np.concatenate([
                validation_data['Home_Win_Rate'].values,
                validation_data['Away_Win_Rate'].values
            ])
            points_values = np.concatenate([
                validation_data['Home_Points'].values,
                validation_data['Away_Points'].values
            ])
            ortg_values = np.concatenate([
                validation_data['Home_ORtg'].values,
                validation_data['Away_ORtg'].values
            ])
            form_values = np.concatenate([
                validation_data['Home_Recent_Form'].values,
                validation_data['Away_Recent_Form'].values
            ])
            streak_values = np.concatenate([
                validation_data['Home_Streak'].values,
                validation_data['Away_Streak'].values
            ])
            
            # Calculate correlations using the combined data
            correlations = {
                'team_strength_win_rate': np.corrcoef(strength_values, win_rate_values)[0, 1],
                'points_scored_ortg': np.corrcoef(points_values, ortg_values)[0, 1],
                'recent_form_streak': np.corrcoef(form_values, streak_values)[0, 1]
            }
            
            # Add correlation metrics
            metrics['correlations'] = correlations
            
            # Validate correlation requirements
            correlation_requirements = {
                'team_strength_win_rate': 0.6,
                'points_scored_ortg': 0.7,
                'recent_form_streak': 0.5
            }
            
            for metric, required in correlation_requirements.items():
                if correlations[metric] < required:
                    logging.warning(
                        f"Correlation validation failed for {metric}: "
                        f"{correlations[metric]:.3f} < {required}"
                    )
            
            logging.info(f"Validation metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logging.error(f"Error in validate_predictions: {str(e)}")
            raise 
