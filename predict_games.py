# predict_games.py
import pandas as pd
import numpy as np
import logging
from nba_scraper import NBADataScraper
from nba_predictor import NBAPredictor
import os
import shutil
from datetime import datetime
import pytz

def main():
    """Train models and generate predictions for today's games"""
    try:
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

        # Ensure models directory exists
        if not os.path.exists('models'):
            os.makedirs('models')
        # else: # <-- REMOVE or COMMENT OUT these lines to avoid deleting models
        #     # Remove old models to force retraining with new features
        #     shutil.rmtree('models')
        #     os.makedirs('models')

        # Initialize scraper for computing rolling stats
        scraper = NBADataScraper(start_season=2021, end_season=2025)

        # Load historical data
        logging.info("Loading historical data...")
        games_df = pd.read_csv('nba_games_all.csv')
        games_df['Date'] = pd.to_datetime(games_df['Date']) # Read as naive

        # Clean games data - IMPORTANT: DO THIS IMMEDIATELY AFTER LOADING
        logging.info("Cleaning games data...")
        cleaned_df = scraper.clean_games_data(games_df)
        if cleaned_df is None:
            logging.error("Error cleaning games data, exiting...")
            return
        games_df = cleaned_df


        # Filter out data before October 18, 2022
        cutoff_date = pd.Timestamp('2022-10-18') # Naive cutoff_date
        initial_rows = len(games_df)
        games_df = games_df[games_df['Date'] >= cutoff_date].copy()
        removed_rows = initial_rows - len(games_df)
        logging.info(f"Removed {removed_rows} games before {cutoff_date.date()}")

        # Convert Is_Future to boolean
        games_df['Is_Future'] = games_df['Is_Future'].astype(bool)

        # Initialize predictor with enhanced settings
        predictor = NBAPredictor()

        # Load trained models
        if predictor.load_models():
            logging.info("Loaded existing models.")
        else:
            logging.info("No existing models found. Training new models...")
            # Compute rolling stats for all games (needed for training)
            logging.info("Computing rolling statistics for historical data for training...")
            games_df = scraper.compute_rolling_stats(games_df, window=5)

            logging.info("Training new models with loaded data...")
            # Use only completed games for training
            training_data = games_df[~games_df['Is_Future']].copy()

            # Add season-based weighting for training
            current_season = 2025
            training_data['season_weight'] = training_data['Season'].apply(
                lambda x: 1.0 if x == current_season else 0.8 if x == current_season-1 else 0.6
            )

            metrics = predictor.train_models(training_data)

            logging.info("\nModel Performance Metrics (after training):")
            logging.info(f"Moneyline Accuracy: {metrics['moneyline_accuracy']:.3f}")
            logging.info(f"Spread RMSE: {metrics['spread_rmse']:.3f}")
            logging.info(f"Totals RMSE: {metrics['totals_rmse']:.3f}")

        # Get today's games with enhanced context
        today_games = scraper.get_current_games()

        if today_games is not None and not today_games.empty:
            if today_games is None or today_games.empty: # ADD THIS CHECK
                logging.info("No games to predict. Exiting prediction routine.")
                return  # Exit function if no games to predict

            logging.info(f"\nGenerating predictions for {len(today_games)} games...")

            # Clean today's games data - IMPORTANT: CLEAN CURRENT GAMES TOO
            logging.info("Cleaning today's games data...")
            if today_games is not None:
                today_games = scraper.clean_games_data(today_games, preserve_future_games=True)
            if today_games is None: # Added check, although get_current_games should not return None in normal cases
                logging.warning("Could not clean today's games data, proceeding without cleaning...")
            else:
                # Compute rolling stats for today's games
                logging.info("Computing rolling statistics for today's games...")
                today_games = scraper.compute_rolling_stats(today_games, window=5)

                # Add recent form metrics
                logging.info("Computing rolling statistics for recent form (window=3)...")
                today_games = scraper.compute_rolling_stats(today_games, window=3)  # Shorter window for recent form

            logging.info("Making predictions...")
            predictions = predictor.predict_games(today_games)

            if predictions.empty:
                logging.info("No predictions returned (empty DataFrame). Skipping best_bets calculation.")
                return  # or just skip get_best_bets if you'd rather continue gracefully

            # Get best bets with stricter criteria
            best_bets = predictor.get_best_bets(
                predictions,
                confidence_threshold=0.90,  # Increased from 0.65
                min_value_rating=0.70      # Added minimum value rating
            )

            # Filter predictions for January 27, 2025
            prediction_date = pd.Timestamp('January 27, 2025') # Naive prediction_date
            filtered_predictions = predictions[pd.to_datetime(predictions['Date']).dt.date == prediction_date.date()]

            # Add Game column to predictions if it doesn't exist
            if 'Game' not in predictions.columns:
                predictions['Game'] = predictions.apply(lambda x: f"{x['Away_Team']} @ {x['Home_Team']}", axis=1)

            # Only filter best bets if we have any
            if not best_bets.empty:
                filtered_best_bets = best_bets[best_bets['Game'].isin(predictions['Game'])]
            else:
                filtered_best_bets = pd.DataFrame(columns=["Game", "Bet_Type", "Prediction", "Confidence", "Value_Rating"])

            logging.info("\nPredictions for today's games:")
            for pred in predictions['Formatted_Predictions']:
                logging.info(f"\n{pred}")

            if not best_bets.empty:
                logging.info("\nHigh Confidence Bets (90%+ confidence) for January 27, 2025:")
                for _, bet in best_bets.iterrows():
                    logging.info(
                        f"\n{bet['Game']}: {bet['Bet_Type']} - {bet['Prediction']}"
                        f"\nConfidence: {bet['Confidence']*100:.1f}%"
                        f"\nValue Rating: {bet['Value_Rating']:.3f}"
                    )

                # Find non-overlapping parlays with stricter criteria
                parlay_options = []
                game_groups = best_bets.groupby('Game')

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
                                        # Calculate enhanced parlay value
                                        combined_confidence = (bet1['Confidence'] + bet2['Confidence']) / 2
                                        combined_value = (bet1['Value_Rating'] + bet2['Value_Rating']) / 2

                                        if combined_confidence >= 0.90 and combined_value >= 0.70:
                                            parlay = {
                                                'bets': [
                                                    f"{bet1['Game']}: {bet1['Bet_Type']} - {bet1['Prediction']} ({bet1['Confidence']*100:.1f}%)",
                                                    f"{bet2['Game']}: {bet2['Bet_Type']} - {bet2['Prediction']} ({bet2['Confidence']*100:.1f}%)"
                                                ],
                                                'combined_confidence': combined_confidence,
                                                'combined_value': combined_value
                                            }
                                            parlay_options.append(parlay)

                # Sort parlays by combined metrics
                parlay_options.sort(key=lambda x: (x['combined_confidence'], x['combined_value']), reverse=True)

                # Display only the highest confidence parlays
                if parlay_options:
                    logging.info("\nHigh Confidence Parlay Options (90%+ combined confidence):")
                    for i, parlay in enumerate(parlay_options[:min(3, len(parlay_options))], 1): # Limit to max 3 parlay options
                        logging.info(
                            f"\nParlay Option {i}"
                            f"\nCombined Confidence: {parlay['combined_confidence']*100:.1f}%"
                            f"\nCombined Value Rating: {parlay['combined_value']:.3f}"
                        )
                        for bet in parlay['bets']:
                            logging.info(f"  {bet}")
                else:
                    logging.info("\nNo high-confidence parlay combinations found")
            else:
                logging.info("\nNo bets meeting enhanced confidence criteria")

            # Save predictions with additional metrics
            predictions['Value_Rating'] = predictions.apply(
                lambda x: predictor._calculate_value_rating(x), axis=1
            )
            predictions.to_csv('predictions.csv', index=False)
            logging.info("\nEnhanced predictions saved to predictions.csv")
        else:
            logging.info("No games scheduled for today")
        logging.info("\nTeam stats are up to date")
        logging.info(f"Successfully processed stats for {len(team_stats)} team-seasons")
        logging.info("\nNBA data scraping process completed successfully")
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()