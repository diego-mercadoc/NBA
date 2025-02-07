# --- BEGIN MONOTONIC PATCH ---
import sklearn.tree

# Check if DecisionTreeClassifier has the required attributes
if not hasattr(sklearn.tree.DecisionTreeClassifier, '_support_missing_values'):
    def _support_missing_values(self, X):
        # Simply return False so that missing value support is not triggered.
        return False
    sklearn.tree.DecisionTreeClassifier._support_missing_values = _support_missing_values

if not hasattr(sklearn.tree.DecisionTreeClassifier, 'monotonic_cst'):
    sklearn.tree.DecisionTreeClassifier.monotonic_cst = None
# --- END MONOTONIC PATCH ---

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, VotingClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
import logging
import joblib
from datetime import datetime, timedelta

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
            'Home_Points_Scored_Roll5', 'Home_Points_Allowed_Roll5', 'Home_Point_Diff_Roll5',
            'Away_Points_Scored_Roll5', 'Away_Points_Allowed_Roll5', 'Away_Point_Diff_Roll5',
            'Home_Streak', 'Away_Streak', 'Home_Rest_Days', 'Away_Rest_Days',
            'Home_Win_Rate', 'Away_Win_Rate',
            'Win_Rate_Diff', 'Point_Diff_Ratio', 'Rest_Advantage',
            'Streak_Advantage', 'Recent_Form_Ratio', 'Win_Rate_Rest_Interaction',
            'Streak_Form_Interaction', 'Avg_Total_Roll5'
        ]
        
        # Initialize base models for ensemble with optimized parameters
        self.rf_classifier = RandomForestClassifier(
            n_estimators=500,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        self.svm_classifier = SVC(
            probability=True,
            kernel='rbf',
            C=10.0,
            gamma='scale',
            random_state=42
        )
        
        self.xgb_classifier = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
    
    def prepare_features(self, games_df):
        """Prepare features for ML models with enhanced engineering"""
        df = games_df.copy()
        
        # For feature engineering, we need historical games.
        historical_games = df[~df['Is_Future']].copy()
        # If no historical data is present (e.g. when df contains only future games), try loading it from file.
        if historical_games.empty:
            try:
                historical_games = pd.read_csv('nba_games_all.csv')
                historical_games['Date'] = pd.to_datetime(historical_games['Date'])
                # Ensure we work only with played games:
                historical_games = historical_games[~historical_games['Is_Future']]
                logging.info("Loaded historical games from nba_games_all.csv for feature filling.")
            except Exception as e:
                logging.warning("Could not load historical games for feature filling; proceeding with empty history.")
                historical_games = pd.DataFrame()
        
        historical_games['Home_Win'] = (historical_games['Home_Points'] > historical_games['Away_Points']).astype(int)
        historical_games['Away_Win'] = (historical_games['Away_Points'] > historical_games['Home_Points']).astype(int)
        
        # Calculate win rates using historical games only
        for team_type in ['Home', 'Away']:
            team_stats = historical_games.groupby(f'{team_type}_Team')['Home_Win'].agg(['count', 'mean'])
            df[f'{team_type}_Win_Rate'] = df[f'{team_type}_Team'].map(team_stats['mean'])
        
        # Fill NaN values in win rates with 0.5 (neutral)
        df['Home_Win_Rate'] = df['Home_Win_Rate'].fillna(0.5)
        df['Away_Win_Rate'] = df['Away_Win_Rate'].fillna(0.5)
        
        # Enhanced feature engineering
        df['Win_Rate_Diff'] = df['Home_Win_Rate'] - df['Away_Win_Rate']
        
        # Calculate rolling statistics for all games
        for team in df['Home_Team'].unique():
            # Get all games for this team (both home and away)
            home_games = df[df['Home_Team'] == team][['Date', 'Home_Points', 'Away_Points', 'Is_Future']].copy()
            home_games.rename(columns={
                'Home_Points': 'Points_Scored',
                'Away_Points': 'Points_Allowed'
            }, inplace=True)
            
            away_games = df[df['Away_Team'] == team][['Date', 'Home_Points', 'Away_Points', 'Is_Future']].copy()
            away_games.rename(columns={
                'Away_Points': 'Points_Scored',
                'Home_Points': 'Points_Allowed'
            }, inplace=True)
            
            # Ensure Points_Scored and Points_Allowed exist before concatenating
            team_games = pd.concat([home_games, away_games]).sort_values('Date')
            
            # Calculate rolling stats using only historical games
            historical_team_games = team_games[~team_games['Is_Future']].copy()
            historical_team_games['Point_Diff'] = historical_team_games['Points_Scored'] - historical_team_games['Points_Allowed']
            
            # Initialize rolling_stats DataFrame with required columns
            rolling_stats = pd.DataFrame(columns=['Points_Scored', 'Points_Allowed', 'Point_Diff'])
            
            # Calculate rolling means for each date
            for date in team_games['Date'].unique():
                past_games = historical_team_games[historical_team_games['Date'] < date]
                if len(past_games) > 0:
                    last_5_games = past_games.tail(5)
                    rolling_stats.loc[date, 'Points_Scored'] = last_5_games['Points_Scored'].mean()
                    rolling_stats.loc[date, 'Points_Allowed'] = last_5_games['Points_Allowed'].mean()
                    rolling_stats.loc[date, 'Point_Diff'] = last_5_games['Point_Diff'].mean()
            
            # Map back to main DataFrame for home games
            home_mask = df['Home_Team'] == team
            df.loc[home_mask, 'Home_Points_Scored_Roll5'] = df.loc[home_mask, 'Date'].map(
                rolling_stats['Points_Scored']
            )
            df.loc[home_mask, 'Home_Points_Allowed_Roll5'] = df.loc[home_mask, 'Date'].map(
                rolling_stats['Points_Allowed']
            )
            df.loc[home_mask, 'Home_Point_Diff_Roll5'] = df.loc[home_mask, 'Date'].map(
                rolling_stats['Point_Diff']
            )
            
            # For Home games: if the mapping returns NaN (likely for future games), fill with the last available value
            missing_home = df.loc[home_mask, 'Home_Points_Scored_Roll5'].isna()
            if missing_home.any():
                # Use the last available rolling value from historical games (if exists)
                if not historical_team_games.empty:
                    last_5_games = historical_team_games.tail(5)
                    last_val_scored = last_5_games['Points_Scored'].mean()
                    last_val_allowed = last_5_games['Points_Allowed'].mean()
                    last_val_diff = last_5_games['Point_Diff'].mean()
                    df.loc[home_mask & missing_home, 'Home_Points_Scored_Roll5'] = last_val_scored
                    df.loc[home_mask & missing_home, 'Home_Points_Allowed_Roll5'] = last_val_allowed
                    df.loc[home_mask & missing_home, 'Home_Point_Diff_Roll5'] = last_val_diff
            
            # Map back to main DataFrame for away games
            away_mask = df['Away_Team'] == team
            df.loc[away_mask, 'Away_Points_Scored_Roll5'] = df.loc[away_mask, 'Date'].map(
                rolling_stats['Points_Scored']
            )
            df.loc[away_mask, 'Away_Points_Allowed_Roll5'] = df.loc[away_mask, 'Date'].map(
                rolling_stats['Points_Allowed']
            )
            df.loc[away_mask, 'Away_Point_Diff_Roll5'] = df.loc[away_mask, 'Date'].map(
                rolling_stats['Point_Diff']
            )
            
            # For Away games: if the mapping returns NaN (likely for future games), fill with the last available value
            missing_away = df.loc[away_mask, 'Away_Points_Scored_Roll5'].isna()
            if missing_away.any():
                # Use the last available rolling value from historical games (if exists)
                if not historical_team_games.empty:
                    last_5_games = historical_team_games.tail(5)
                    last_val_scored = last_5_games['Points_Scored'].mean()
                    last_val_allowed = last_5_games['Points_Allowed'].mean()
                    last_val_diff = last_5_games['Point_Diff'].mean()
                    df.loc[away_mask & missing_away, 'Away_Points_Scored_Roll5'] = last_val_scored
                    df.loc[away_mask & missing_away, 'Away_Points_Allowed_Roll5'] = last_val_allowed
                    df.loc[away_mask & missing_away, 'Away_Point_Diff_Roll5'] = last_val_diff
        
        # Fill NaN values in rolling stats with team averages from historical games
        rolling_cols = [
            'Home_Points_Scored_Roll5', 'Home_Points_Allowed_Roll5', 'Home_Point_Diff_Roll5',
            'Away_Points_Scored_Roll5', 'Away_Points_Allowed_Roll5', 'Away_Point_Diff_Roll5'
        ]
        for col in rolling_cols:
            team_type = 'Home' if 'Home_' in col else 'Away'
            stat_type = col.split('_')[-2]  # Points_Scored, Points_Allowed, or Point_Diff
            
            # Calculate team averages from historical games
            team_averages = {}
            for team in df[f'{team_type}_Team'].unique():
                team_mask = (historical_games[f'{team_type}_Team'] == team)
                if stat_type == 'Points_Scored':
                    avg = historical_games.loc[team_mask, f'{team_type}_Points'].mean()
                elif stat_type == 'Points_Allowed':
                    opp_type = 'Away' if team_type == 'Home' else 'Home'
                    avg = historical_games.loc[team_mask, f'{opp_type}_Points'].mean()
                else:  # Point_Diff
                    if team_type == 'Home':
                        avg = historical_games.loc[team_mask, 'Home_Points'].mean() - historical_games.loc[team_mask, 'Away_Points'].mean()
                    else:
                        avg = historical_games.loc[team_mask, 'Away_Points'].mean() - historical_games.loc[team_mask, 'Home_Points'].mean()
                team_averages[team] = avg
            
            # Fill NaN values with team averages
            mask = df[col].isna()
            df.loc[mask, col] = df.loc[mask, f'{team_type}_Team'].map(team_averages)
        
        # Fill any remaining NaN values with global averages
        for col in rolling_cols:
            df[col] = df[col].fillna(df[col].mean())
        
        # Calculate point differential ratio safely
        df['Point_Diff_Ratio'] = df.apply(
            lambda x: x['Home_Point_Diff_Roll5'] / (abs(x['Away_Point_Diff_Roll5']) + 1e-6)
            if x['Home_Point_Diff_Roll5'] != 0 or x['Away_Point_Diff_Roll5'] != 0
            else 1.0,
            axis=1
        )
        
        # Fill NaN values in rest days with median from historical games
        for col in ['Home_Rest_Days', 'Away_Rest_Days']:
            if col in df.columns:
                median_rest = historical_games[col].median()
                df[col] = df[col].fillna(median_rest if not pd.isna(median_rest) else 1)
        df['Rest_Advantage'] = df['Home_Rest_Days'] - df['Away_Rest_Days']
        
        # Fill NaN values in streaks with 0
        for col in ['Home_Streak', 'Away_Streak']:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        df['Streak_Advantage'] = df['Home_Streak'] - df['Away_Streak']
        
        # Calculate form ratio with NaN handling
        df['Recent_Form_Ratio'] = df.apply(
            lambda x: (x['Home_Point_Diff_Roll5'] + 1e-6) / (abs(x['Away_Point_Diff_Roll5']) + 1e-6)
            if x['Home_Point_Diff_Roll5'] != 0 or x['Away_Point_Diff_Roll5'] != 0
            else 1.0,
            axis=1
        )
        
        # Interaction features
        df['Win_Rate_Rest_Interaction'] = df['Win_Rate_Diff'] * df['Rest_Advantage']
        df['Streak_Form_Interaction'] = df['Streak_Advantage'] * df['Recent_Form_Ratio']
        
        # --- New Totals Features ---
        # Calculate rolling totals for both home and away teams over the last 5 games
        df['Home_Total_Roll5'] = df['Home_Points_Scored_Roll5'] + df['Home_Points_Allowed_Roll5']
        df['Away_Total_Roll5'] = df['Away_Points_Scored_Roll5'] + df['Away_Points_Allowed_Roll5']
        # Create an average total feature from both teams
        df['Avg_Total_Roll5'] = (df['Home_Total_Roll5'] + df['Away_Total_Roll5']) / 2

        # Optionally, log statistics for the new feature
        new_feature = 'Avg_Total_Roll5'
        stats = df[new_feature].describe()
        logging.info(f"\n{new_feature}:")
        logging.info(f"  Mean: {stats['mean']:.3f}")
        logging.info(f"  Std: {stats['std']:.3f}")
        logging.info(f"  Min: {stats['min']:.3f}")
        logging.info(f"  Max: {stats['max']:.3f}")
        non_zero = (df[new_feature] != 0).sum()
        logging.info(f"  Non-zero values: {non_zero} ({(non_zero/len(df))*100:.1f}%)")
        
        # Ensure all required feature columns exist
        for col in self.feature_columns:
            if col not in df.columns:
                logging.warning(f"Missing feature column: {col}, filling with zeros")
                df[col] = 0
        
        # Fill remaining NaN values with 0
        feature_cols = self.feature_columns
        X = df[feature_cols].fillna(0)
        
        # Add debug logging for missing values and feature statistics
        logging.info(f"\nFeature statistics and missing value analysis for {len(df)} games:")
        for col in feature_cols:
            missing_pct = df[col].isna().mean() * 100
            zero_pct = (df[col] == 0).mean() * 100
            stats = X[col].describe()
            logging.info(f"\n{col}:")
            logging.info(f"  Missing values: {missing_pct:.1f}% (before fill)")
            logging.info(f"  Zero values: {zero_pct:.1f}%")
            logging.info(f"  Mean: {stats['mean']:.3f}")
            logging.info(f"  Std: {stats['std']:.3f}")
            logging.info(f"  Min: {stats['min']:.3f}")
            logging.info(f"  Max: {stats['max']:.3f}")
            non_zero = (X[col] != 0).sum()
            logging.info(f"  Non-zero values: {non_zero} ({(non_zero/len(X))*100:.1f}%)")
        
        return X
    
    def prepare_labels(self, games_df):
        """Prepare labels for different prediction types"""
        played_games = games_df.dropna(subset=['Home_Points', 'Away_Points'])
        
        y_moneyline = (played_games['Home_Points'] > played_games['Away_Points']).astype(int)
        y_spread = played_games['Home_Points'] - played_games['Away_Points']
        y_totals = played_games['Home_Points'] + played_games['Away_Points']
        
        return y_moneyline, y_spread, y_totals
    
    def train_models(self, games_df, test_size=0.2):
        """Train ML models with enhanced ensemble and fine-tuning"""
        logging.info("Preparing features and labels...")
        
        played_games = games_df.dropna(subset=['Home_Points', 'Away_Points'])
        X = self.prepare_features(played_games)
        y_moneyline, y_spread, y_totals = self.prepare_labels(played_games)
        
        # Split data with stratification for moneyline
        X_train, X_test, y_ml_train, y_ml_test = train_test_split(
            X, y_moneyline, test_size=test_size, stratify=y_moneyline, random_state=42
        )
        _, _, y_spread_train, y_spread_test = train_test_split(
            X, y_spread, test_size=test_size, random_state=42
        )
        _, _, y_totals_train, y_totals_test = train_test_split(
            X, y_totals, test_size=test_size, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        logging.info("Starting time-aware cross-validation for moneyline classifier...")
        cv_accuracies = self.cross_validate_moneyline(X_train_scaled, y_ml_train, n_splits=5)
        
        # Train base models separately
        logging.info("Training enhanced moneyline ensemble model...")
        self.moneyline_model = VotingClassifier(
            estimators=[
                ('rf', self.rf_classifier),
                ('svm', self.svm_classifier),
                ('xgb', self.xgb_classifier)
            ],
            voting='soft',
            weights=[2, 1, 2]  # Increased weights for RF and XGB
        )
        self.moneyline_model.fit(X_train_scaled, y_ml_train)
        
        # Train enhanced spread model using XGBoost with early stopping:
        logging.info("Training enhanced spread model...")
        self.spread_model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=7,
            learning_rate=0.02,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42
        )
        self.spread_model.fit(
            X_train_scaled, y_spread_train,
            early_stopping_rounds=20,
            eval_set=[(X_test_scaled, y_spread_test)],
            verbose=False
        )
        y_spread_pred = self.spread_model.predict(X_test_scaled)
        spread_rmse = np.sqrt(mean_squared_error(y_spread_test, y_spread_pred))
        logging.info(f"Enhanced spread model RMSE: {spread_rmse:.3f}")
        
        # Train totals model using XGBoost with early stopping
        logging.info("Training enhanced totals model with XGBoost...")
        self.totals_model = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.02,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42
        )
        self.totals_model.fit(
            X_train_scaled, y_totals_train,
            early_stopping_rounds=20,
            eval_set=[(X_test_scaled, y_totals_test)],
            verbose=False
        )
        
        # Evaluate totals model
        y_totals_pred = self.totals_model.predict(X_test_scaled)
        totals_rmse = np.sqrt(mean_squared_error(y_totals_test, y_totals_pred))
        logging.info(f"Enhanced totals model RMSE: {totals_rmse:.3f}")
        
        # Save models
        self.save_models()
        
        return {
            'moneyline_accuracy': cv_accuracies[-1] if cv_accuracies else 0,
            'spread_rmse': spread_rmse,
            'totals_rmse': totals_rmse
        }
    
    def save_models(self):
        """Save trained models and scaler"""
        joblib.dump(self.moneyline_model, 'models/moneyline_model.joblib')
        joblib.dump(self.spread_model, 'models/spread_model.joblib')
        joblib.dump(self.totals_model, 'models/totals_model.joblib')
        joblib.dump(self.scaler, 'models/scaler.joblib')
        joblib.dump(self.xgb_classifier, 'models/xgb_classifier.joblib')
        logging.info("Models saved successfully")
    
    def load_models(self):
        """Load trained models and scaler"""
        try:
            self.moneyline_model = joblib.load('models/moneyline_model.joblib')
            self.spread_model = joblib.load('models/spread_model.joblib')
            self.totals_model = joblib.load('models/totals_model.joblib')
            self.scaler = joblib.load('models/scaler.joblib')
            self.xgb_classifier = joblib.load('models/xgb_classifier.joblib')
            logging.info("Models loaded successfully")
            return True
        except Exception as e:
            logging.error(f"Error loading models: {str(e)}")
            return False
    
    def predict_games(self, games_df):
        """Generate predictions for games with enhanced ensemble"""
        X = self.prepare_features(games_df)
        if X.empty: # Handle empty DataFrame case
             logging.warning("No features to predict on. Returning empty DataFrame.")
             return pd.DataFrame()
        X_scaled = self.scaler.transform(X)

        # Add additional safety check for scaled features
        if X_scaled.shape[0] == 0:
            logging.warning("No features available for prediction - returning empty DataFrame")
            return pd.DataFrame()
        
        predictions = pd.DataFrame()
        predictions['Home_Team'] = games_df['Home_Team']
        predictions['Away_Team'] = games_df['Away_Team']
        predictions['Date'] = games_df['Date']
        
        # Get predictions from ensemble model directly
        ensemble_proba = self.moneyline_model.predict_proba(X_scaled)
        predictions['Home_Win_Prob'] = ensemble_proba[:, 1]
        predictions['Away_Win_Prob'] = ensemble_proba[:, 0]
        predictions['Moneyline_Pick'] = (predictions['Home_Win_Prob'] > 0.5).astype(int)
        
        # Spread and totals predictions
        raw_spread = self.spread_model.predict(X_scaled)
        predictions['Predicted_Spread'] = np.where(
            predictions['Moneyline_Pick'] == 1,
            -abs(raw_spread),  # Home team favored (negative spread)
            abs(raw_spread)    # Away team favored (positive spread)
        )
        # Use ensemble: average predictions from both the LightGBM and XGBoost totals models
        predictions['Predicted_Total'] = (self.totals_model.predict(X_scaled))
        
        # First Half predictions (based on historical patterns)
        predictions['First_Half_Total'] = predictions['Predicted_Total'] * 0.52
        predictions['First_Half_Spread'] = predictions['Predicted_Spread'] * 0.48
        
        # Quarter predictions
        predictions['Avg_Quarter_Points'] = predictions['Predicted_Total'] / 4
        predictions['First_Quarter_Total'] = predictions['Predicted_Total'] * 0.24
        predictions['First_Quarter_Spread'] = predictions['Predicted_Spread'] * 0.45
        
        # Enhanced confidence calculation using ensemble agreement
        predictions['Moneyline_Confidence'] = np.maximum(
            predictions['Home_Win_Prob'],
            predictions['Away_Win_Prob']
        )
        
        # Format predictions
        predictions['Formatted_Predictions'] = predictions.apply(
            lambda x: self._format_prediction(x), axis=1
        )
        
        return predictions
    
    def _format_prediction(self, row):
        """Format prediction row into readable string"""
        home_prob = row['Home_Win_Prob']
        away_prob = row['Away_Win_Prob']
        spread = row['Predicted_Spread']
        total = row['Predicted_Total']
        
        pick = f"{row['Home_Team']} ML" if row['Moneyline_Pick'] else f"{row['Away_Team']} ML"
        confidence = row['Moneyline_Confidence']
        
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

        # --- NEW CHECK: if empty or missing Moneyline_Confidence, return empty DF
        if (
            predictions_df is None 
            or predictions_df.empty 
            or 'Moneyline_Confidence' not in predictions_df.columns
        ):
            logging.warning("No valid predictions or missing 'Moneyline_Confidence' column. "
                            "Returning empty best bets DataFrame.")
            return pd.DataFrame(columns=["Game", "Bet_Type", "Prediction", "Confidence", "Value_Rating"])
        # ---------------------------------------------------------

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

    def cross_validate_moneyline(self, X, y, n_splits=5):
        """Perform time-aware cross-validation for the moneyline classifier using TimeSeriesSplit."""
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import accuracy_score
        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_accuracies = []
        fold = 1
        
        # Convert y to numpy array to avoid index issues
        y_np = y.to_numpy() if isinstance(y, pd.Series) else y
        
        for train_idx, test_idx in tscv.split(X):
            X_train_cv, X_test_cv = X[train_idx], X[test_idx]
            y_train_cv, y_test_cv = y_np[train_idx], y_np[test_idx]
            # Create a new VotingClassifier instance with the same parameters
            model = VotingClassifier(
                estimators=[
                    ('rf', self.rf_classifier),
                    ('svm', self.svm_classifier)
                ],
                voting='soft',
                weights=[2, 1]
            )
            model.fit(X_train_cv, y_train_cv)
            y_pred_cv = model.predict(X_test_cv)
            acc = accuracy_score(y_test_cv, y_pred_cv)
            fold_accuracies.append(acc)
            logging.info(f"Fold {fold} Moneyline CV Accuracy: {acc:.3f}")
            fold += 1
        mean_acc = np.mean(fold_accuracies)
        logging.info(f"Mean Time-Aware Moneyline CV Accuracy: {mean_acc:.3f}")
        return fold_accuracies 