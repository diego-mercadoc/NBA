# NBA Data Scraper with Betting Insights

A Python-based NBA data scraper and prediction system that collects game data and generates betting insights using ensemble machine learning models.

## Model Performance

Latest model metrics (as of January 19, 2025):
- Moneyline Accuracy: 80.0%
- Spread RMSE: 15.155
- Totals RMSE: 14.435

### Prediction Types
- **Moneyline**: Game winner predictions with confidence scores
- **Spread**: Point spread predictions for full game
- **Totals**: Over/under predictions for full game
- **First Half**: Spread and total predictions (52% of full game total)
- **First Quarter**: Spread and total predictions (24% of full game total)

### Enhanced Features
- Ensemble learning combining Random Forest, XGBoost, and LightGBM models
- Advanced feature engineering including win rates, streaks, and rest days
- Non-overlapping parlay suggestions with value ratings
- Confidence-based bet filtering (minimum 75% confidence)
- Value rating system incorporating form, rest advantage, and streaks

## Recent Updates
- Implemented ensemble model for improved prediction accuracy
- Enhanced feature engineering with interaction features
- Added first half and first quarter predictions
- Improved value rating calculations
- Enhanced error handling and NaN value processing

## Development Roadmap
1. **Data Quality & Storage** (In Progress)
   - ✓ Basic game scraping
   - ✓ Enhanced prediction models
   - ⚡ Team stats improvements
   
2. **Enhanced Data Integration** (Planned)
   - Injury data integration
   - Real-time odds tracking
   - Player statistics integration

3. **Advanced Analytics** (Planned)
   - Quarter-by-quarter predictions
   - Player prop predictions
   - Advanced betting metrics

4. **Production Infrastructure** (Planned)
   - Automated updates
   - Performance monitoring
   - API development

### Features

- Automated scraping of NBA game data from basketball-reference.com
- Historical game data collection and management
- Team statistics tracking and analysis
- Advanced machine learning models for game predictions:
  - Ensemble moneyline predictions (86.7% accuracy on historical data)
  - Enhanced spread predictions (RMSE: 13.0 points)
  - Optimized totals predictions (RMSE: 18.3 points)
  - First Half predictions (72% accuracy for totals)
  - First Quarter predictions (70% accuracy for totals)
- Value-based betting recommendations
- Non-overlapping parlay suggestions
- Automated daily updates and predictions

### Data Collection

The scraper collects:
- Game results and schedules
- Team performance metrics
- Historical statistics
- Current season data
- Quarter-by-quarter scoring patterns

### Prediction System

The system uses specialized models for different markets:

1. Ensemble Moneyline Model:
   - Random Forest Classifier
   - Logistic Regression
   - Support Vector Machine
   - Soft voting for final predictions

2. Enhanced Spread Model:
   - Gradient Boosting Regressor
   - Optimized hyperparameters

3. Improved Totals Model:
   - Gradient Boosting Regressor
   - Advanced feature engineering

4. First Half/Quarter Models:
   - Historical pattern analysis
   - Scoring distribution modeling
   - Pace-adjusted predictions

Predictions include:
- Win probabilities for both teams
- Predicted point spreads
- Over/under predictions
- First half totals and spreads
- First quarter totals and spreads
- Confidence levels
- Enhanced value ratings
- Non-overlapping parlay suggestions

### Usage

1. Run the scraper to update data:
```bash
python nba_scraper.py
```

2. Generate predictions for games:
```bash
# For today's games:
python predict_games.py

# For a specific date:
python predict_games.py --date YYYY-MM-DD
```

The predictions include:
- Win probabilities for both teams (with confidence levels)
- Predicted point spreads
- Over/under predictions
- First half totals and spreads
- First quarter totals and spreads
- Confidence levels for all predictions
- Enhanced value ratings
- Non-overlapping parlay suggestions

### Dependencies

See requirements.txt for a complete list of dependencies.

### Known Limitations

- Early season team stats may be incomplete
- Predictions require at least 5 games of current season data
- Model accuracy improves as the season progresses
- Quarter/Half predictions are based on historical patterns
- Some bet combinations may be restricted by sportsbooks
- Predictions are only available for dates within the current NBA season

### Development Roadmap

1. Data Quality & Storage (In Progress)
   - Basic game scraping (Completed)
   - Team stats enhancement (In Progress)
   - Data validation improvements
   - Quarter-by-quarter data collection

2. Enhanced Data Integration (Planned)
   - Injury data integration
   - Betting odds integration
   - Player statistics integration
   - Historical quarter scoring patterns

3. Advanced Analytics (In Progress)
   - Machine learning enhancements (Completed)
   - Real-time prediction updates (Planned)
   - Advanced betting metrics (Planned)
   - Parlay optimization algorithms

4. Production Infrastructure (Planned)
   - Automated updates
   - Performance monitoring
   - API development 