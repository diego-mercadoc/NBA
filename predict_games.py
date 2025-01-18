import pandas as pd
import numpy as np
import logging
from nba_scraper import NBADataScraper
from nba_predictor import NBAPredictor
import os
import shutil

def main():
    """Train models and generate predictions for today's games"""
    try:
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Create models directory if it doesn't exist
        if not os.path.exists('models'):
            os.makedirs('models')
        else:
            # Remove old models to force retraining with new features
            shutil.rmtree('models')
            os.makedirs('models')
        
        # Load historical data
        logging.info("Loading historical data...")
        games_df = pd.read_csv('nba_games_all.csv')
        games_df['Date'] = pd.to_datetime(games_df['Date'])
        
        # Convert Is_Future to boolean
        games_df['Is_Future'] = games_df['Is_Future'].astype(bool)
        
        # Initialize predictor
        predictor = NBAPredictor()
        
        logging.info("Training new models with enhanced features...")
        # Use only completed games for training
        training_data = games_df[~games_df['Is_Future']].copy()
        metrics = predictor.train_models(training_data)
        
        logging.info("\nModel Performance Metrics:")
        logging.info(f"Moneyline Accuracy: {metrics['moneyline_accuracy']:.3f}")
        logging.info(f"Spread RMSE: {metrics['spread_rmse']:.3f}")
        logging.info(f"Totals RMSE: {metrics['totals_rmse']:.3f}")
        
        # Get today's games
        scraper = NBADataScraper(start_season=2021, end_season=2025)
        today_games = scraper.get_current_games()
        
        if today_games is not None and not today_games.empty:
            logging.info(f"\nGenerating predictions for {len(today_games)} games...")
            predictions = predictor.predict_games(today_games)
            
            # Get best bets
            best_bets = predictor.get_best_bets(predictions, confidence_threshold=0.65)
            
            logging.info("\nPredictions for today's games:")
            for pred in predictions['Formatted_Predictions']:
                logging.info(f"\n{pred}")
            
            if not best_bets.empty:
                logging.info("\nRecommended Bets (65%+ confidence):")
                for _, bet in best_bets.iterrows():
                    logging.info(f"\n{bet['Formatted_Predictions']}")
                    logging.info(f"Value Rating: {bet['Value_Rating']:.3f}")
            else:
                logging.info("\nNo high-confidence bets found for today's games")
            
            # Save predictions
            predictions.to_csv('predictions.csv', index=False)
            logging.info("\nPredictions saved to predictions.csv")
        else:
            logging.info("No games scheduled for today")
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 