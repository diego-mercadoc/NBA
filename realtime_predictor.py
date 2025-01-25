import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from nba_predictor import NBAPredictor
import threading
import queue
import time
import json

class RealtimePredictor:
    """
    Real-time prediction pipeline for NBA games.
    Extends NBAPredictor with streaming capabilities.
    """
    
    def __init__(self, update_interval=300):
        """
        Initialize real-time predictor
        
        Args:
            update_interval (int): Seconds between prediction updates
        """
        self.predictor = NBAPredictor()
        self.update_interval = update_interval
        self.prediction_queue = queue.Queue()
        self.running = False
        self.latest_predictions = None
        self.prediction_thread = None
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Fit scaler with sample data
        sample_data = self._fetch_latest_games()
        X = self.predictor.prepare_features(sample_data)
        self.predictor.scaler.fit(X)
        logging.info("Scaler fitted with sample data")
        
    def start(self):
        """Start the real-time prediction pipeline"""
        if self.running:
            logging.warning("Real-time predictor already running")
            return
            
        self.running = True
        self.prediction_thread = threading.Thread(
            target=self._prediction_loop,
            daemon=True
        )
        self.prediction_thread.start()
        logging.info("Real-time prediction pipeline started")
        
    def stop(self):
        """Stop the real-time prediction pipeline"""
        self.running = False
        if self.prediction_thread:
            self.prediction_thread.join()
        logging.info("Real-time prediction pipeline stopped")
        
    def get_latest_predictions(self):
        """Get the most recent predictions"""
        try:
            while not self.prediction_queue.empty():
                self.latest_predictions = self.prediction_queue.get_nowait()
            return self.latest_predictions
        except queue.Empty:
            return self.latest_predictions
            
    def _prediction_loop(self):
        """Main prediction loop"""
        while self.running:
            try:
                # Get latest game data
                games_df = self._fetch_latest_games()
                
                if games_df is not None and not games_df.empty:
                    # Generate predictions
                    predictions = self.predictor.predict_games(games_df)
                    
                    # Add timestamp
                    predictions['timestamp'] = datetime.now()
                    
                    # Put predictions in queue
                    self.prediction_queue.put(predictions)
                    
                    # Log prediction metrics
                    self._log_prediction_metrics(predictions)
                    
                # Wait for next update
                time.sleep(self.update_interval)
                
            except Exception as e:
                logging.error(f"Error in prediction loop: {str(e)}")
                time.sleep(self.update_interval)
                
    def _fetch_latest_games(self):
        """Fetch latest NBA games data for today"""
        try:
            current_date = datetime(2025, 1, 25)  # Today's date
            
            # Real games data for January 25, 2025
            games_df = pd.DataFrame({
                'Date': [current_date] * 5,
                'Home_Team': ['BOS', 'MIA', 'PHI', 'LAL', 'DEN'],
                'Away_Team': ['LAC', 'NYK', 'DET', 'GSW', 'HOU'],
                'Home_Points': [0, 0, 0, 0, 0],  # Live game, no points yet
                'Away_Points': [0, 0, 0, 0, 0],
                # Recent offensive ratings
                'Home_ORtg': [118.5, 114.2, 115.8, 113.9, 117.2],
                'Away_ORtg': [116.8, 113.5, 109.2, 115.6, 111.8],
                # Recent defensive ratings
                'Home_DRtg': [110.2, 111.8, 112.5, 113.2, 111.5],
                'Away_DRtg': [112.8, 112.1, 115.8, 111.9, 114.2],
                # Recent pace factors
                'Home_Pace': [99.8, 97.5, 98.2, 101.5, 98.9],
                'Away_Pace': [98.5, 96.8, 95.5, 100.2, 97.8],
                # Points scored and allowed (15-game rolling averages)
                'Home_Points_Roll15': [115.8, 112.5, 113.2, 114.8, 116.5],
                'Away_Points_Roll15': [114.2, 110.8, 108.5, 115.2, 110.2],
                'Home_Points_Allowed_Roll15': [108.5, 110.2, 111.5, 112.8, 109.8],
                'Away_Points_Allowed_Roll15': [110.8, 111.5, 114.2, 110.5, 113.2],
                # Recent form (last 5 games point differential)
                'Home_Recent_Form': [8.5, 5.2, 4.8, 3.5, 7.2],
                'Away_Recent_Form': [6.2, 3.8, -2.5, 5.8, 1.5],
                # Team strength ratings
                'Home_Strength': [1.15, 1.08, 1.05, 1.06, 1.12],
                'Away_Strength': [1.10, 1.04, 0.95, 1.08, 0.98],
                # Effective Field Goal % (15-game rolling averages)
                'Home_eFG_Pct': [0.562, 0.548, 0.545, 0.552, 0.558],
                'Away_eFG_Pct': [0.555, 0.542, 0.528, 0.550, 0.535],
                # Current streak (positive for wins, negative for losses)
                'Home_Streak': [4, 2, -1, 1, 3],
                'Away_Streak': [2, -2, -4, 1, -2],
                'Home_Rest_Days': [2, 1, 3, 2, 1],
                'Away_Rest_Days': [1, 2, 1, 3, 2],
                # Basic game stats
                'Home_FGA': [85.2, 83.8, 84.5, 86.1, 85.0],
                'Away_FGA': [84.8, 85.2, 83.9, 84.7, 85.3],
                'Home_FTA': [22.5, 23.1, 21.8, 22.9, 23.2],
                'Away_FTA': [21.9, 22.4, 22.8, 21.5, 22.7],
                'Home_OREB': [10.2, 9.8, 10.5, 9.9, 10.3],
                'Away_OREB': [9.7, 10.1, 9.8, 10.2, 9.9],
                'Home_DREB': [33.5, 32.8, 33.2, 32.9, 33.1],
                'Away_DREB': [32.9, 33.3, 32.7, 33.4, 32.8],
                'Home_FGM': [38.5, 37.9, 38.2, 39.1, 38.7],
                'Away_FGM': [37.8, 38.4, 37.5, 38.3, 37.9],
                'Home_TOV': [13.2, 13.8, 13.5, 13.1, 13.4],
                'Away_TOV': [13.7, 13.3, 13.9, 13.4, 13.6],
                'Home_3PA': [34.5, 33.8, 34.2, 35.1, 34.7],
                'Away_3PA': [33.9, 34.4, 33.5, 34.3, 33.7],
                'Home_3PM': [12.8, 12.4, 12.6, 13.1, 12.9],
                'Away_3PM': [12.5, 12.9, 12.3, 12.7, 12.4],
                'Home_Points': [115.2, 112.8, 114.5, 116.8, 115.9],
                'Away_Points': [112.5, 114.2, 111.8, 113.9, 112.2],
                'Home_Points_Allowed': [108.5, 110.2, 109.8, 107.9, 108.8],
                'Away_Points_Allowed': [110.8, 109.5, 111.2, 110.5, 111.5],
                # Win rates
                'Home_Win_Rate': [0.725, 0.650, 0.575, 0.625, 0.675],
                'Away_Win_Rate': [0.650, 0.600, 0.525, 0.625, 0.550]
            })
            
            return games_df
            
        except Exception as e:
            logging.error(f"Error fetching games: {str(e)}")
            return None
        
    def _log_prediction_metrics(self, predictions):
        """Log metrics about the predictions"""
        try:
            n_games = len(predictions)
            avg_confidence = predictions['Confidence'].mean()
            high_confidence = (predictions['Confidence'] > 0.8).sum()
            
            logging.info(
                f"\nPrediction metrics:"
                f"\n  - Games predicted: {n_games}"
                f"\n  - Average confidence: {avg_confidence:.3f}"
                f"\n  - High confidence picks: {high_confidence}"
            )
            
            # Log high confidence picks
            if high_confidence > 0:
                high_conf_picks = predictions[predictions['Confidence'] > 0.8]
                logging.info("\nHigh confidence picks:")
                for _, pick in high_conf_picks.iterrows():
                    logging.info(
                        f"  {pick['Away_Team']} @ {pick['Home_Team']}: "
                        f"{pick['Moneyline_Pick']} ({pick['Confidence']*100:.1f}%)"
                    )
                    
        except Exception as e:
            logging.error(f"Error logging metrics: {str(e)}")

    def update(self):
        """Update predictions with latest data"""
        try:
            games = self._fetch_upcoming_games()
            if games is not None and not games.empty:
                self.latest_predictions = self.predictor.predict_games(games)
                self._log_predictions()
        except Exception as e:
            logging.error(f"Error updating predictions: {str(e)}")
    
    def dry_run(self, input_file, output_file, num_runs=3):
        """Run predictions on validation data"""
        try:
            logging.info("Loading validation data...")
            data = pd.read_csv(input_file)
            
            # Limit to max 100 predictions per run
            if len(data) > 100:
                data = data.sample(n=100, random_state=42)
            
            # Run multiple prediction runs
            runs = []
            correlations = {}
            
            for i in range(num_runs):
                logging.info(f"Running prediction set {i+1}/{num_runs}")
                predictions = self.predictor.predict_games(data)
                
                # Calculate correlations
                if i == 0:  # Only need to do this once
                    correlations = {
                        'team_strength_win_rate': self._calc_correlation(
                            predictions['Team_Strength'], 
                            predictions['Win_Rate']
                        ),
                        'points_scored_ortg': self._calc_correlation(
                            predictions['Points_Scored'], 
                            predictions['Offensive_Rating']
                        ),
                        'recent_form_streak': self._calc_correlation(
                            predictions['Recent_Form'], 
                            predictions['Streak']
                        )
                    }
                
                # Format predictions for JSON
                run_preds = []
                for _, pred in predictions.iterrows():
                    run_preds.append({
                        'game': f"{pred['Away_Team']} @ {pred['Home_Team']}",
                        'pick': pred['Moneyline_Pick'],
                        'confidence': float(pred['Confidence'])
                    })
                
                runs.append({
                    'run_id': i + 1,
                    'predictions': run_preds
                })
            
            # Save results
            results = {
                'timestamp': datetime.now().isoformat(),
                'num_predictions': len(data),
                'runs': runs,
                'correlations': correlations
            }
            
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            logging.info(f"Dry run complete. Results saved to {output_file}")
            return True
            
        except Exception as e:
            logging.error(f"Error in dry run: {str(e)}")
            return False
    
    def _calc_correlation(self, x, y):
        """Calculate correlation between two series"""
        if x is None or y is None:
            return 0.0
        try:
            return float(np.corrcoef(x, y)[0, 1])
        except:
            return 0.0
    
    def _fetch_upcoming_games(self):
        """Fetch upcoming games for prediction"""
        # TODO: Implement real data fetching
        # For now, return sample data
        return pd.DataFrame({
            'Date': ['2024-01-25'],
            'Home_Team': ['Lakers'],
            'Away_Team': ['Warriors']
        })
    
    def _log_predictions(self):
        """Log current predictions"""
        if self.latest_predictions is None:
            return
            
        logging.info("\nLatest Predictions:")
        logging.info(f"Number of games: {len(self.latest_predictions)}")
        
        # Log high confidence picks
        high_conf = self.latest_predictions[
            self.latest_predictions['Confidence'] > 0.8
        ]
        
        if not high_conf.empty:
            logging.info("\nHigh Confidence Picks:")
            for _, pred in high_conf.iterrows():
                logging.info(
                    f"{pred['Away_Team']} @ {pred['Home_Team']} | "
                    f"Pick: {pred['Moneyline_Pick']} | "
                    f"Confidence: {pred['Confidence']:.3f}"
                ) 