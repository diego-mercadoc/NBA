import pandas as pd
import numpy as np
import logging
from nba_scraper import NBADataScraper
from nba_predictor import NBAPredictor
import os
import shutil
import json
from datetime import datetime
import argparse

def main():
    """Train models and generate predictions for today's games"""
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('--date', type=str, help='Date to predict games for (YYYY-MM-DD)')
        args = parser.parse_args()
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Initialize scraper for computing rolling stats
        scraper = NBADataScraper(start_season=2021, end_season=2025)
        
        # Load historical data
        logging.info("Loading historical data...")
        games_df = pd.read_csv('nba_games_all.csv')
        games_df['Date'] = pd.to_datetime(games_df['Date'])
        
        # Load team stats
        logging.info("Loading team stats...")
        team_stats = pd.read_csv('nba_team_stats_all.csv')
        
        # Rename columns for home team merge
        home_stats = team_stats.copy()
        home_stats.columns = [f'Home_{col}' if col != 'Team' and col != 'Season' else col for col in home_stats.columns]
        
        # Rename columns for away team merge
        away_stats = team_stats.copy()
        away_stats.columns = [f'Away_{col}' if col != 'Team' and col != 'Season' else col for col in away_stats.columns]
        
        # Merge team stats with games data
        games_df = games_df.merge(
            home_stats,
            left_on=['Season', 'Home_Team'],
            right_on=['Season', 'Team'],
            how='left'
        ).merge(
            away_stats,
            left_on=['Season', 'Away_Team'],
            right_on=['Season', 'Team'],
            how='left'
        )
        
        # Drop redundant columns
        games_df = games_df.drop(['Team_x', 'Team_y'], axis=1, errors='ignore')
        
        # Convert Is_Future to boolean
        games_df['Is_Future'] = games_df['Is_Future'].astype(bool)
        
        # Compute rolling stats for all games
        logging.info("Computing rolling statistics for historical data...")
        games_df = scraper.compute_rolling_stats(games_df, window=15)  # Increased window for stability
        
        # Initialize predictor
        predictor = NBAPredictor()
        
        # Check if models exist and are recent
        models_exist = os.path.exists('models') and len(os.listdir('models')) > 0
        if models_exist:
            model_time = os.path.getmtime('models')
            current_time = datetime.now().timestamp()
            models_recent = (current_time - model_time) < 24*60*60  # 24 hours
        else:
            models_recent = False
            
        if not models_exist or not models_recent:
            logging.info("Training new models with Bayesian optimization...")
            if os.path.exists('models'):
                shutil.rmtree('models')
            os.makedirs('models')
            
            # Use only completed games for training
            training_data = games_df[~games_df['Is_Future']].copy()
            
            # Add season-based weighting for training
            current_season = 2025
            training_data['season_weight'] = training_data['Season'].apply(
                lambda x: 1.0 if x == current_season else 0.8 if x == current_season-1 else 0.6
            )
            
            # Initialize metrics dictionary
            metrics = {
                'moneyline_accuracy': 0.0,
                'moneyline_brier': 0.0,
                'moneyline_log_loss': 0.0,
                'spread_rmse': 0.0,
                'totals_rmse': 0.0,
                'metrics_history': []
            }
            
            # Train models and update metrics
            train_metrics = predictor.train_models(training_data)
            metrics.update(train_metrics)
            
            # Save metrics to track model performance over time
            metrics_file = 'model_metrics.json'
            metrics['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            try:
                with open(metrics_file, 'r') as f:
                    historical_metrics = json.load(f)
            except FileNotFoundError:
                historical_metrics = {'metrics_history': []}
            
            historical_metrics['metrics_history'].append(metrics)
            with open(metrics_file, 'w') as f:
                json.dump(historical_metrics, f, indent=4)
            
            logging.info("\nModel Performance Metrics:")
            logging.info(f"Moneyline Accuracy: {metrics.get('moneyline_accuracy', 0.0):.3f}")
            logging.info(f"Moneyline Brier Score: {metrics.get('moneyline_brier', 0.0):.3f}")
            logging.info(f"Moneyline Log Loss: {metrics.get('moneyline_log_loss', 0.0):.3f}")
            logging.info(f"Spread RMSE: {metrics.get('spread_rmse', 0.0):.3f}")
            logging.info(f"Totals RMSE: {metrics.get('totals_rmse', 0.0):.3f}")
        else:
            logging.info("Using existing models (less than 24 hours old)")
        
        # Get games for specified date or today
        if args.date:
            target_date = pd.to_datetime(args.date)
            today_games = games_df[games_df['Date'].dt.date == target_date.date()].copy()
        else:
            today_games = scraper.get_current_games()
        
        if today_games is not None and not today_games.empty:
            logging.info(f"\nGenerating predictions for {len(today_games)} games...")
            
            # Add recent form metrics
            today_games = scraper.compute_rolling_stats(today_games, window=15)
            predictions = predictor.predict_games(today_games)
            
            # Get best bets with stricter criteria
            best_bets = predictor.get_best_bets(
                predictions, 
                confidence_threshold=0.65,  # Lowered threshold for testing
                min_value_rating=0.60
            )
            
            logging.info("\nPredictions for today's games:")
            for pred in predictions['Formatted_Predictions']:
                logging.info(f"\n{pred}")
            
            if not best_bets.empty:
                logging.info("\nHigh Confidence Bets:")
                for _, bet in best_bets.iterrows():
                    logging.info(
                        f"\n{bet['Game']}: {bet['Bet_Type']} - {bet['Prediction']}"
                        f"\nConfidence: {bet['Confidence']*100:.1f}%"
                        f"\nValue Rating: {bet['Value_Rating']:.3f}"
                    )
            else:
                logging.info("\nNo bets meeting confidence criteria")
            
            # Save predictions with additional metrics
            predictions['Value_Rating'] = predictions.apply(
                lambda x: predictor._calculate_value_rating(x), axis=1
            )
            predictions.to_csv('predictions.csv', index=False)
            logging.info("\nPredictions saved to predictions.csv")
        else:
            logging.info("No games found for the specified date")
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 