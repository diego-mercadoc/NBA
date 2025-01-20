import pandas as pd
import numpy as np
import logging
from nba_scraper import NBADataScraper
from nba_predictor import NBAPredictor
import os
import shutil
import argparse
from datetime import datetime

def main():
    """Train models and generate predictions for specified date"""
    try:
        # Configure argument parser
        parser = argparse.ArgumentParser(description='Generate NBA game predictions')
        parser.add_argument('--date', type=str, help='Target date in YYYY-MM-DD format (default: today)')
        args = parser.parse_args()
        
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
        
        # Get games for target date
        scraper = NBADataScraper(start_season=2021, end_season=2025)
        target_games = scraper.get_current_games(args.date)
        
        if target_games is not None and not target_games.empty:
            logging.info(f"\nGenerating predictions for {len(target_games)} games...")
            predictions = predictor.predict_games(target_games)
            
            # Get best bets
            best_bets = predictor.get_best_bets(predictions, confidence_threshold=0.65)
            
            logging.info("\nPredictions for games:")
            for pred in predictions['Formatted_Predictions']:
                logging.info(f"\n{pred}")
            
            if not best_bets.empty:
                logging.info("\nRecommended Non-Overlapping Parlays (65%+ confidence):")
                
                # Group bets by game
                game_groups = best_bets.groupby('Game')
                
                # Find non-overlapping combinations
                parlay_options = []
                for game1, bets1 in game_groups:
                    for game2, bets2 in game_groups:
                        if game1 < game2:  # Avoid duplicates
                            for _, bet1 in bets1.iterrows():
                                for _, bet2 in bets2.iterrows():
                                    # Check if bet types are allowed together
                                    if [bet1['Bet_Type'], bet2['Bet_Type']] in [
                                        ["Moneyline", "First Half Total"],
                                        ["Moneyline", "First Quarter Total"],
                                        ["First Half Total", "First Quarter Total"]
                                    ]:
                                        parlay = {
                                            'bets': [
                                                f"{bet1['Game']}: {bet1['Bet_Type']} - {bet1['Prediction']} ({bet1['Confidence']*100:.1f}%)",
                                                f"{bet2['Game']}: {bet2['Bet_Type']} - {bet2['Prediction']} ({bet2['Confidence']*100:.1f}%)"
                                            ],
                                            'combined_value': (bet1['Value_Rating'] + bet2['Value_Rating']) / 2
                                        }
                                        parlay_options.append(parlay)
                
                # Sort parlays by combined value
                parlay_options.sort(key=lambda x: x['combined_value'], reverse=True)
                
                # Display top 5 parlay options
                for i, parlay in enumerate(parlay_options[:5], 1):
                    logging.info(f"\nParlay Option {i} (Value: {parlay['combined_value']:.3f}):")
                    for bet in parlay['bets']:
                        logging.info(f"  {bet}")
            else:
                logging.info("\nNo high-confidence bets found for the specified date")
            
            # Save predictions
            predictions.to_csv('predictions.csv', index=False)
            logging.info("\nPredictions saved to predictions.csv")
        else:
            logging.info("No games found for the specified date")
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 