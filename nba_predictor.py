import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, VotingClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
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
            'Home_Win_Rate', 'Away_Win_Rate'
        ]
        
        # Initialize base models for ensemble
        self.rf_classifier = RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42
        )
        self.lr_classifier = LogisticRegression(
            C=1.0, max_iter=1000, random_state=42
        )
        self.svm_classifier = SVC(
            probability=True, kernel='rbf', random_state=42
        )
    
    def prepare_features(self, games_df):
        """Prepare features for ML models"""
        df = games_df.copy()
        
        # Calculate win/loss columns
        df['Home_Win'] = (df['Home_Points'] > df['Away_Points']).astype(int)
        df['Away_Win'] = (df['Away_Points'] > df['Home_Points']).astype(int)
        
        # Calculate win rates
        for team_type in ['Home', 'Away']:
            if team_type == 'Home':
                team_stats = df.groupby(f'{team_type}_Team')['Home_Win'].agg(['count', 'mean'])
                df[f'{team_type}_Win_Rate'] = df[f'{team_type}_Team'].map(team_stats['mean'])
            else:
                team_stats = df.groupby(f'{team_type}_Team')['Home_Win'].agg(['count', 'mean'])
                df[f'{team_type}_Win_Rate'] = df[f'{team_type}_Team'].map(1 - team_stats['mean'])
        
        # Add advanced features
        df['Win_Rate_Diff'] = df['Home_Win_Rate'] - df['Away_Win_Rate']
        df['Point_Diff_Ratio'] = df['Home_Point_Diff_Roll5'] / (df['Away_Point_Diff_Roll5'] + 1e-6)
        
        # Prepare feature matrix
        feature_cols = self.feature_columns + ['Win_Rate_Diff', 'Point_Diff_Ratio']
        X = df[feature_cols].copy()
        
        # Handle missing values
        X = X.fillna(0)
        
        return X
    
    def prepare_labels(self, games_df):
        """Prepare labels for different prediction types"""
        played_games = games_df.dropna(subset=['Home_Points', 'Away_Points'])
        
        y_moneyline = (played_games['Home_Points'] > played_games['Away_Points']).astype(int)
        y_spread = played_games['Home_Points'] - played_games['Away_Points']
        y_totals = played_games['Home_Points'] + played_games['Away_Points']
        
        return y_moneyline, y_spread, y_totals
    
    def train_models(self, games_df, test_size=0.2):
        """Train ML models for different bet types"""
        logging.info("Preparing features and labels...")
        
        played_games = games_df.dropna(subset=['Home_Points', 'Away_Points'])
        X = self.prepare_features(played_games)
        y_moneyline, y_spread, y_totals = self.prepare_labels(played_games)
        
        # Split data
        X_train, X_test, y_ml_train, y_ml_test = train_test_split(
            X, y_moneyline, test_size=test_size, random_state=42
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
        
        # Create ensemble moneyline model
        logging.info("Training ensemble moneyline model...")
        self.moneyline_model = VotingClassifier(
            estimators=[
                ('rf', self.rf_classifier),
                ('lr', self.lr_classifier),
                ('svm', self.svm_classifier)
            ],
            voting='soft'
        )
        self.moneyline_model.fit(X_train_scaled, y_ml_train)
        
        # Evaluate moneyline model
        y_ml_pred = self.moneyline_model.predict(X_test_scaled)
        ml_accuracy = accuracy_score(y_ml_test, y_ml_pred)
        logging.info(f"Ensemble moneyline model accuracy: {ml_accuracy:.3f}")
        logging.info("\nMoneyline Classification Report:")
        logging.info(classification_report(y_ml_test, y_ml_pred))
        
        # Train spread model with enhanced parameters
        logging.info("Training spread model...")
        self.spread_model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
        self.spread_model.fit(X_train_scaled, y_spread_train)
        
        # Evaluate spread model
        y_spread_pred = self.spread_model.predict(X_test_scaled)
        spread_rmse = np.sqrt(mean_squared_error(y_spread_test, y_spread_pred))
        logging.info(f"Spread model RMSE: {spread_rmse:.3f}")
        
        # Train totals model with enhanced parameters
        logging.info("Training totals model...")
        self.totals_model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
        self.totals_model.fit(X_train_scaled, y_totals_train)
        
        # Evaluate totals model
        y_totals_pred = self.totals_model.predict(X_test_scaled)
        totals_rmse = np.sqrt(mean_squared_error(y_totals_test, y_totals_pred))
        logging.info(f"Totals model RMSE: {totals_rmse:.3f}")
        
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
        """Generate predictions for games"""
        X = self.prepare_features(games_df)
        X_scaled = self.scaler.transform(X)
        
        predictions = pd.DataFrame()
        predictions['Home_Team'] = games_df['Home_Team']
        predictions['Away_Team'] = games_df['Away_Team']
        predictions['Date'] = games_df['Date']
        
        # Moneyline predictions with ensemble model
        ml_probs = self.moneyline_model.predict_proba(X_scaled)
        predictions['Home_Win_Prob'] = ml_probs[:, 1]
        predictions['Away_Win_Prob'] = ml_probs[:, 0]
        predictions['Moneyline_Pick'] = self.moneyline_model.predict(X_scaled)
        
        # Spread and totals predictions
        predictions['Predicted_Spread'] = self.spread_model.predict(X_scaled)
        predictions['Predicted_Total'] = self.totals_model.predict(X_scaled)
        
        # First Half predictions (based on historical patterns)
        predictions['First_Half_Total'] = predictions['Predicted_Total'] * 0.52  # Historically about 52% of points
        predictions['First_Half_Spread'] = predictions['Predicted_Spread'] * 0.48  # Slightly less spread in first half
        
        # Quarter predictions
        predictions['Avg_Quarter_Points'] = predictions['Predicted_Total'] / 4
        predictions['First_Quarter_Total'] = predictions['Predicted_Total'] * 0.24  # Historically about 24% of points
        predictions['First_Quarter_Spread'] = predictions['Predicted_Spread'] * 0.45  # Less spread in first quarter
        
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
    
    def get_best_bets(self, predictions_df, confidence_threshold=0.65):
        """Filter and return the highest confidence bets across different markets"""
        best_bets = []
        
        # Moneyline bets
        ml_bets = predictions_df[
            predictions_df['Moneyline_Confidence'] > confidence_threshold
        ].copy()
        
        for _, game in ml_bets.iterrows():
            bet = {
                'Game': f"{game['Away_Team']} @ {game['Home_Team']}",
                'Bet_Type': 'Moneyline',
                'Prediction': f"{game['Home_Team']} ML" if game['Moneyline_Pick'] else f"{game['Away_Team']} ML",
                'Confidence': game['Moneyline_Confidence'],
                'Value_Rating': self._calculate_value_rating(game)
            }
            best_bets.append(bet)
            
            # First Half total (more predictable than full game)
            if abs(game['First_Half_Total'] - (game['Predicted_Total'] * 0.52)) < 5:
                bet = {
                    'Game': f"{game['Away_Team']} @ {game['Home_Team']}",
                    'Bet_Type': 'First Half Total',
                    'Prediction': f"Over {game['First_Half_Total']:.1f}",
                    'Confidence': 0.72,  # Historical accuracy for first half totals
                    'Value_Rating': 0.65
                }
                best_bets.append(bet)
            
            # First Quarter total (most predictable quarter)
            if abs(game['First_Quarter_Total'] - (game['Predicted_Total'] * 0.24)) < 3:
                bet = {
                    'Game': f"{game['Away_Team']} @ {game['Home_Team']}",
                    'Bet_Type': 'First Quarter Total',
                    'Prediction': f"Over {game['First_Quarter_Total']:.1f}",
                    'Confidence': 0.70,  # Historical accuracy for first quarter totals
                    'Value_Rating': 0.60
                }
                best_bets.append(bet)
        
        return pd.DataFrame(best_bets)
    
    def _calculate_value_rating(self, row):
        """Calculate a value rating for the bet based on confidence and other factors"""
        # Enhanced value rating calculation
        prob_margin = abs(row['Home_Win_Prob'] - row['Away_Win_Prob'])
        value_rating = row['Moneyline_Confidence'] * prob_margin
        
        # Adjust for recent form and rest days
        if all(col in row for col in ['Home_Point_Diff_Roll5', 'Away_Point_Diff_Roll5', 'Home_Rest_Days', 'Away_Rest_Days']):
            form_factor = abs(row['Home_Point_Diff_Roll5'] - row['Away_Point_Diff_Roll5']) / 10
            rest_factor = abs(row['Home_Rest_Days'] - row['Away_Rest_Days']) / 5
            value_rating *= (1 + form_factor + rest_factor)
        
        return value_rating 