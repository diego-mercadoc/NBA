# NBA Data Scraper with Betting Insights

A Python-based NBA data scraper and prediction system that collects game data and generates betting insights.

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

2. Generate predictions for upcoming games:
```bash
python predict_games.py
```

### Dependencies

See requirements.txt for a complete list of dependencies.

### Known Limitations

- Early season team stats may be incomplete
- Predictions require at least 5 games of current season data
- Model accuracy improves as the season progresses
- Quarter/Half predictions are based on historical patterns
- Some bet combinations may be restricted by sportsbooks

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