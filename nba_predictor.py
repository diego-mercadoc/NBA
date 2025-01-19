import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, VotingClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import xgboost as xgb
import lightgbm as lgb
import logging
import joblib
from datetime import datetime, timedelta
from nba_injury_tracker import NBAInjuryTracker

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
        self.injury_tracker = NBAInjuryTracker()
        self.feature_columns = [
            'Home_Points_Scored_Roll5', 'Home_Points_Allowed_Roll5', 'Home_Point_Diff_Roll5',
            'Away_Points_Scored_Roll5', 'Away_Points_Allowed_Roll5', 'Away_Point_Diff_Roll5',
            'Home_Points_Scored_Roll3', 'Home_Points_Allowed_Roll3', 'Home_Point_Diff_Roll3',
            'Away_Points_Scored_Roll3', 'Away_Points_Allowed_Roll3', 'Away_Point_Diff_Roll3',
            'Home_Points_Scored_Roll10', 'Home_Points_Allowed_Roll10', 'Home_Point_Diff_Roll10',
            'Away_Points_Scored_Roll10', 'Away_Points_Allowed_Roll10', 'Away_Point_Diff_Roll10',
            'Home_Streak', 'Away_Streak', 'Home_Rest_Days', 'Away_Rest_Days',
            'Home_Win_Rate', 'Away_Win_Rate',
            'Home_Injury_Impact', 'Away_Injury_Impact'
        ]
        
        # Initialize base models for ensemble with optimized parameters
        self.rf_classifier = RandomForestClassifier(
            n_estimators=500,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        self.lr_classifier = LogisticRegression(
            C=0.8,
            max_iter=2000,
            class_weight='balanced',
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
            random_state=42
        )
        
        self.lgb_classifier = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
    
    def prepare_features(self, games_df):
        """Prepare features for ML models with enhanced engineering and injury data"""
        df = games_df.copy()
        
        # Update injury data
        self.injury_tracker.update_injuries()
        
        # Add injury impact scores
        df['Home_Injury_Impact'] = df['Home_Team'].apply(self.injury_tracker.get_team_injury_impact)
        df['Away_Injury_Impact'] = df['Away_Team'].apply(self.injury_tracker.get_team_injury_impact)
        
        # Calculate win/loss columns
        df['Home_Win'] = (df['Home_Points'] > df['Away_Points']).astype(int)
        df['Away_Win'] = (df['Away_Points'] > df['Home_Points']).astype(int)
        
        # Calculate rolling stats for multiple windows
        for window in [3, 5, 10]:
            for team_type in ['Home', 'Away']:
                # Points scored rolling average
                df[f'{team_type}_Points_Scored_Roll{window}'] = df.groupby(f'{team_type}_Team')['Home_Points' if team_type == 'Home' else 'Away_Points'].transform(lambda x: x.rolling(window, min_periods=1).mean())
                
                # Points allowed rolling average
                df[f'{team_type}_Points_Allowed_Roll{window}'] = df.groupby(f'{team_type}_Team')['Away_Points' if team_type == 'Home' else 'Home_Points'].transform(lambda x: x.rolling(window, min_periods=1).mean())
                
                # Point differential rolling average
                df[f'{team_type}_Point_Diff_Roll{window}'] = df[f'{team_type}_Points_Scored_Roll{window}'] - df[f'{team_type}_Points_Allowed_Roll{window}']
        
        # Calculate weighted recent form
        for team_type in ['Home', 'Away']:
            df[f'{team_type}_Recent_Form_Weighted'] = (
                0.5 * df[f'{team_type}_Point_Diff_Roll3'] +
                0.3 * df[f'{team_type}_Point_Diff_Roll5'] +
                0.2 * df[f'{team_type}_Point_Diff_Roll10']
            )
            
            # Add weighted form to feature columns if not already present
            if f'{team_type}_Recent_Form_Weighted' not in self.feature_columns:
                self.feature_columns.append(f'{team_type}_Recent_Form_Weighted')
        
        # Add opponent style interaction features
        # Pace differential
        df['Pace_Differential'] = df['Home_Points_Scored_Roll5'] + df['Home_Points_Allowed_Roll5'] - (df['Away_Points_Scored_Roll5'] + df['Away_Points_Allowed_Roll5'])
        self.feature_columns.append('Pace_Differential')

        # Offensive vs Defensive matchup
        df['Home_Off_vs_Away_Def'] = df['Home_Points_Scored_Roll5'] / df['Away_Points_Allowed_Roll5'].replace(0, 1)
        df['Away_Off_vs_Home_Def'] = df['Away_Points_Scored_Roll5'] / df['Home_Points_Allowed_Roll5'].replace(0, 1)
        self.feature_columns.extend(['Home_Off_vs_Away_Def', 'Away_Off_vs_Home_Def'])

        # Style similarity score (lower means more similar playing styles)
        df['Style_Similarity'] = abs(df['Pace_Differential']) + abs(df['Home_Off_vs_Away_Def'] - df['Away_Off_vs_Home_Def'])
        self.feature_columns.append('Style_Similarity')

        # Form-matchup interaction
        df['Home_Form_Matchup'] = df['Home_Recent_Form_Weighted'] * df['Home_Off_vs_Away_Def']
        df['Away_Form_Matchup'] = df['Away_Recent_Form_Weighted'] * df['Away_Off_vs_Home_Def']
        self.feature_columns.extend(['Home_Form_Matchup', 'Away_Form_Matchup'])
        
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
        
        # Enhanced feature engineering
        df['Win_Rate_Diff'] = df['Home_Win_Rate'] - df['Away_Win_Rate']
        
        # Handle potential division by zero or NaN
        df['Away_Point_Diff_Roll5'] = df['Away_Point_Diff_Roll5'].fillna(0)
        df['Home_Point_Diff_Roll5'] = df['Home_Point_Diff_Roll5'].fillna(0)
        df['Point_Diff_Ratio'] = df.apply(
            lambda x: x['Home_Point_Diff_Roll5'] / (x['Away_Point_Diff_Roll5'] + 1e-6)
            if x['Away_Point_Diff_Roll5'] != 0
            else x['Home_Point_Diff_Roll5'],
            axis=1
        )
        
        # Fill NaN values in rest days with median
        df['Home_Rest_Days'] = df['Home_Rest_Days'].fillna(df['Home_Rest_Days'].median())
        df['Away_Rest_Days'] = df['Away_Rest_Days'].fillna(df['Away_Rest_Days'].median())
        df['Rest_Advantage'] = df['Home_Rest_Days'] - df['Away_Rest_Days']
        
        # Fill NaN values in streaks with 0
        df['Home_Streak'] = df['Home_Streak'].fillna(0)
        df['Away_Streak'] = df['Away_Streak'].fillna(0)
        df['Streak_Advantage'] = df['Home_Streak'] - df['Away_Streak']
        
        # Calculate form ratio with NaN handling
        df['Recent_Form_Ratio'] = df.apply(
            lambda x: (x['Home_Point_Diff_Roll5'] + 1e-6) / (x['Away_Point_Diff_Roll5'] + 1e-6)
            if x['Away_Point_Diff_Roll5'] != 0
            else x['Home_Point_Diff_Roll5'],
            axis=1
        )
        
        # Interaction features
        df['Win_Rate_Rest_Interaction'] = df['Win_Rate_Diff'] * df['Rest_Advantage']
        df['Streak_Form_Interaction'] = df['Streak_Advantage'] * df['Recent_Form_Ratio']
        
        # Fill remaining NaN values with 0
        feature_cols = self.feature_columns + [
            'Win_Rate_Diff', 'Point_Diff_Ratio', 'Rest_Advantage',
            'Streak_Advantage', 'Recent_Form_Ratio', 'Win_Rate_Rest_Interaction',
            'Streak_Form_Interaction'
        ]
        X = df[feature_cols].fillna(0)
        
        return X
    
    def prepare_labels(self, games_df):
        """Prepare labels for different prediction types"""
        played_games = games_df.dropna(subset=['Home_Points', 'Away_Points'])
        
        y_moneyline = (played_games['Home_Points'] > played_games['Away_Points']).astype(int)
        y_spread = played_games['Home_Points'] - played_games['Away_Points']
        y_totals = played_games['Home_Points'] + played_games['Away_Points']
        
        return y_moneyline, y_spread, y_totals
    
    def train_models(self, games_df, test_size=0.2):
        """Train ML models with enhanced ensemble and calibration"""
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
        
        # Train base models separately with calibration
        logging.info("Training enhanced moneyline models with calibration...")
        
        # Calibrate each base model
        self.rf_calibrated = CalibratedClassifierCV(
            self.rf_classifier, 
            cv=5, 
            method='sigmoid'
        )
        self.rf_calibrated.fit(X_train_scaled, y_ml_train)
        
        self.lr_calibrated = CalibratedClassifierCV(
            self.lr_classifier,
            cv=5,
            method='sigmoid'
        )
        self.lr_calibrated.fit(X_train_scaled, y_ml_train)
        
        self.svm_calibrated = CalibratedClassifierCV(
            self.svm_classifier,
            cv=5,
            method='sigmoid'
        )
        self.svm_calibrated.fit(X_train_scaled, y_ml_train)
        
        # Train sklearn ensemble with calibrated models
        self.moneyline_model = VotingClassifier(
            estimators=[
                ('rf', self.rf_calibrated),
                ('lr', self.lr_calibrated),
                ('svm', self.svm_calibrated)
            ],
            voting='soft',
            weights=[2, 1, 1]
        )
        self.moneyline_model.fit(X_train_scaled, y_ml_train)
        
        # Train and calibrate XGBoost
        self.xgb_calibrated = CalibratedClassifierCV(
            self.xgb_classifier,
            cv=5,
            method='sigmoid'
        )
        self.xgb_calibrated.fit(X_train_scaled, y_ml_train)
        
        # Train and calibrate LightGBM
        self.lgb_calibrated = CalibratedClassifierCV(
            self.lgb_classifier,
            cv=5,
            method='sigmoid'
        )
        self.lgb_calibrated.fit(X_train_scaled, y_ml_train)
        
        # Get predictions from all calibrated models
        sklearn_proba = self.moneyline_model.predict_proba(X_test_scaled)
        xgb_proba = self.xgb_calibrated.predict_proba(X_test_scaled)
        lgb_proba = self.lgb_calibrated.predict_proba(X_test_scaled)
        
        # Weighted average of calibrated probabilities
        ensemble_proba = (2*sklearn_proba + 2*xgb_proba + 2*lgb_proba) / 6
        y_ml_pred = (ensemble_proba[:, 1] > 0.5).astype(int)
        
        # Evaluate calibrated moneyline model
        ml_accuracy = accuracy_score(y_ml_test, y_ml_pred)
        logging.info(f"Enhanced calibrated ensemble moneyline model accuracy: {ml_accuracy:.3f}")
        logging.info("\nMoneyline Classification Report:")
        logging.info(classification_report(y_ml_test, y_ml_pred))
        
        # Check calibration quality
        prob_true, prob_pred = calibration_curve(y_ml_test, ensemble_proba[:, 1], n_bins=10)
        logging.info("\nCalibration Check (ideal values should be similar):")
        for true_prob, pred_prob in zip(prob_true, prob_pred):
            logging.info(f"True probability: {true_prob:.3f}, Predicted probability: {pred_prob:.3f}")
        
        # Train enhanced spread model with XGBoost
        logging.info("Training enhanced spread model...")
        self.spread_model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        self.spread_model.fit(X_train_scaled, y_spread_train)
        
        # Evaluate spread model
        y_spread_pred = self.spread_model.predict(X_test_scaled)
        spread_rmse = np.sqrt(mean_squared_error(y_spread_test, y_spread_pred))
        logging.info(f"Enhanced spread model RMSE: {spread_rmse:.3f}")
        
        # Train enhanced totals model with LightGBM
        logging.info("Training enhanced totals model...")
        self.totals_model = lgb.LGBMRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        self.totals_model.fit(X_train_scaled, y_totals_train)
        
        # Evaluate totals model
        y_totals_pred = self.totals_model.predict(X_test_scaled)
        totals_rmse = np.sqrt(mean_squared_error(y_totals_test, y_totals_pred))
        logging.info(f"Enhanced totals model RMSE: {totals_rmse:.3f}")
        
        # Save models
        self.save_models()
        
        return {
            'moneyline_accuracy': ml_accuracy,
            'spread_rmse': spread_rmse,
            'totals_rmse': totals_rmse
        }
    
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
        """Generate predictions for games with enhanced ensemble, value ratings, and injury consideration"""
        X = self.prepare_features(games_df)
        X_scaled = self.scaler.transform(X)
        
        predictions = pd.DataFrame()
        predictions['Home_Team'] = games_df['Home_Team']
        predictions['Away_Team'] = games_df['Away_Team']
        predictions['Date'] = games_df['Date']
        
        # Get predictions from all calibrated models
        sklearn_proba = self.moneyline_model.predict_proba(X_scaled)
        xgb_proba = self.xgb_calibrated.predict_proba(X_scaled)
        lgb_proba = self.lgb_calibrated.predict_proba(X_scaled)
        
        # Weighted average of calibrated probabilities
        ensemble_proba = (2*sklearn_proba + 2*xgb_proba + 2*lgb_proba) / 6
        
        # Adjust probabilities based on injury impact
        home_injury_impact = X['Home_Injury_Impact']
        away_injury_impact = X['Away_Injury_Impact']
        
        # Reduce win probability based on injury impact
        ensemble_proba[:, 1] *= (1 - home_injury_impact)  # Home team
        ensemble_proba[:, 0] *= (1 - away_injury_impact)  # Away team
        
        # Renormalize probabilities
        row_sums = ensemble_proba.sum(axis=1)
        ensemble_proba = ensemble_proba / row_sums[:, np.newaxis]
        
        predictions['Home_Win_Prob'] = ensemble_proba[:, 1]
        predictions['Away_Win_Prob'] = ensemble_proba[:, 0]
        predictions['Moneyline_Pick'] = (predictions['Home_Win_Prob'] > 0.5).astype(int)
        
        # Add injury information to predictions
        predictions['Home_Injury_Impact'] = home_injury_impact
        predictions['Away_Injury_Impact'] = away_injury_impact
        
        # Get detailed injury reports
        predictions['Home_Injury_Report'] = predictions['Home_Team'].apply(
            lambda x: self.injury_tracker.get_injury_report(x).to_dict('records')
        )
        predictions['Away_Injury_Report'] = predictions['Away_Team'].apply(
            lambda x: self.injury_tracker.get_injury_report(x).to_dict('records')
        )
        
        # Spread and totals predictions
        predictions['Predicted_Spread'] = self.spread_model.predict(X_scaled)
        predictions['Predicted_Total'] = self.totals_model.predict(X_scaled)
        
        # Enhanced value ratings
        predictions['Form_Factor'] = np.tanh(
            (X['Home_Recent_Form_Weighted'] - X['Away_Recent_Form_Weighted']) / 10
        )
        
        predictions['Rest_Advantage'] = np.tanh(
            (X['Home_Rest_Days'] - X['Away_Rest_Days']) / 3
        )
        
        predictions['Streak_Impact'] = np.tanh(
            (X['Home_Streak'] - X['Away_Streak']) / 5
        )
        
        # Style matchup consideration
        predictions['Style_Edge'] = np.tanh(
            (X['Home_Form_Matchup'] - X['Away_Form_Matchup']) / 5
        )
        
        # Calculate base value rating
        predictions['Base_Value_Rating'] = (
            0.4 * predictions['Form_Factor'] +
            0.2 * predictions['Rest_Advantage'] +
            0.2 * predictions['Streak_Impact'] +
            0.2 * predictions['Style_Edge']
        ).abs()
        
        # Probability margin boost
        prob_margin = abs(predictions['Home_Win_Prob'] - 0.5)
        predictions['Probability_Margin_Boost'] = np.where(
            prob_margin > 0.4,
            1.1,
            1.0
        )
        
        # Final value rating
        predictions['Value_Rating'] = predictions['Base_Value_Rating'] * predictions['Probability_Margin_Boost']
        
        # First Half predictions (based on historical patterns)
        predictions['First_Half_Total'] = predictions['Predicted_Total'] * 0.52
        predictions['First_Half_Spread'] = predictions['Predicted_Spread'] * 0.48
        
        # Quarter predictions
        predictions['First_Quarter_Total'] = predictions['Predicted_Total'] * 0.24
        predictions['First_Quarter_Spread'] = predictions['Predicted_Spread'] * 0.45
        
        # Add confidence levels based on calibrated probabilities and value ratings
        predictions['Confidence_Level'] = pd.cut(
            predictions['Home_Win_Prob'].apply(lambda x: max(x, 1-x)),
            bins=[0, 0.75, 0.80, 0.90, 1.0],
            labels=['Low', 'Medium', 'High', 'Very High']
        )
        
        # Flag high-value bets
        predictions['High_Value_Bet'] = (
            (predictions['Value_Rating'] > 0.7) &
            (predictions['Home_Win_Prob'].apply(lambda x: max(x, 1-x)) > 0.8)
        )
        
        return predictions
    
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
        """Enhanced value rating calculation with injury impact consideration"""
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
            
            # Injury consideration
            injury_diff = abs(row.get('Home_Injury_Impact', 0) - row.get('Away_Injury_Impact', 0))
            injury_factor = np.tanh(injury_diff * 3)  # Amplify injury impact
            
            # Higher value if one team is significantly more injured
            injury_advantage = 1 + (0.2 * injury_factor)
            
            # Combine factors with weighted importance
            value_rating *= (
                (1 + 0.3*form_factor + 0.2*rest_factor + 0.2*streak_factor) * 
                recent_weight * 
                injury_advantage
            )
        
        # Additional confidence boost for extreme differentials
        if prob_margin > 0.4:  # 40% probability difference
            value_rating *= 1.1
        
        # Reduce value if both teams heavily injured
        if row.get('Home_Injury_Impact', 0) > 0.2 and row.get('Away_Injury_Impact', 0) > 0.2:
            value_rating *= 0.8  # Reduce confidence when both teams missing key players
        
        # Cap the value rating at 1.0
        return min(value_rating, 1.0)
    
    def _calculate_half_value(self, row):
        """Calculate value rating for first half totals with injury consideration"""
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
        
        # Further reduce confidence for halves if significant injuries
        total_injury_impact = row.get('Home_Injury_Impact', 0) + row.get('Away_Injury_Impact', 0)
        if total_injury_impact > 0.3:  # Significant combined injuries
            base_value *= 0.9  # Reduce confidence more for half predictions
        
        return min(base_value * 0.95, 1.0)  # Slightly lower confidence for halves
    
    def _calculate_quarter_value(self, row):
        """Calculate value rating for first quarter totals with injury consideration"""
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
        
        # Further reduce confidence for quarters if significant injuries
        total_injury_impact = row.get('Home_Injury_Impact', 0) + row.get('Away_Injury_Impact', 0)
        if total_injury_impact > 0.3:  # Significant combined injuries
            base_value *= 0.85  # Reduce confidence even more for quarter predictions
        
        return min(base_value * 0.90, 1.0)  # Lower confidence for quarters 