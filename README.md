# NBA Data Scraper with Betting Insights

A Python-based NBA data scraper and prediction system that collects game data and generates betting insights using ensemble machine learning models.

## Model Performance

Latest model metrics (as of January 18, 2025):
- Moneyline Accuracy: 80.0%
- Spread RMSE: 15.155
- Totals RMSE: 14.435

### Enhanced Confidence Measures
- Increased confidence threshold to 90% (from 65%)
- Minimum value rating requirement of 70%
- Normalized form factors using tanh transformation
- Exponential weighting for recent performance
- Stricter validation for half/quarter predictions
- Additional confidence boost for strong differentials

### Prediction Types
- **Moneyline**: Game winner predictions with enhanced confidence scores
- **Spread**: Point spread predictions for full game
- **Totals**: Over/under predictions for full game
- **First Half**: Spread and total predictions (52% of full game total)
  - Enhanced validation (±3 points from expected)
  - Confidence capped at 95%
- **First Quarter**: Spread and total predictions (24% of full game total)
  - Tighter validation (±1.5 points from expected)
  - Confidence capped at 95%

### Enhanced Features
- Ensemble learning combining Random Forest, XGBoost, and LightGBM models
- Injury tracking and impact analysis:
  - Multiple data sources with fallback mechanism:
    * Primary: NBA.com
    * Secondary: ESPN
    * Tertiary: Rotoworld
    * Quaternary: Basketball Reference
  - Automatic 12-hour data updates
  - Player impact tiers:
    * Superstar: 15% impact
    * Star: 10% impact
    * Starter: 5% impact
    * Rotation: 2.5% impact
    * Bench: 1% impact
  - Injury status weighting:
    * Out: 100% of impact
    * Doubtful: 100% of impact
    * Questionable: 50% of impact
    * Probable: 25% of impact
  - Mock data support for testing and development
  - Comprehensive error handling and logging
- Season-based weighting for training data
  - Current season: 1.0x weight
  - Previous season: 0.8x weight
  - Earlier seasons: 0.6x weight
- Advanced feature engineering including:
  - Win rates with exponential weighting
  - Normalized form factors
  - Rest advantage with diminishing returns
  - Streak impact with momentum consideration
  - Recent performance weighting (last 3 games)
- Non-overlapping parlay suggestions with enhanced criteria:
  - Combined confidence ≥ 90%
  - Combined value rating ≥ 70%
- Value rating system incorporating:
  - Form factor (tanh normalized)
  - Rest advantage (diminishing returns)
  - Streak impact (momentum adjusted)
  - Recent performance boost
  - Probability margin bonus
  - Injury impact adjustment

## Recent Updates
- Enhanced injury tracking system with multiple data sources and fallback mechanism
- Implemented tiered player impact scoring system
- Added mock data support for testing and development
- Improved error handling and logging for injury data collection
- Enhanced prediction confidence with injury impact consideration
- Improved value rating calculations with injury factors
- Implemented ensemble model for improved prediction accuracy
- Enhanced feature engineering with interaction features
- Added first half and first quarter predictions

## Development Roadmap
1. **Data Quality & Storage** (In Progress)
   - ✓ Basic game scraping
   - ✓ Enhanced prediction models
   - ⚡ Team stats improvements
   
2. **Enhanced Data Integration** (In Progress)
   - ✓ Injury data integration
   - ⚡ Real-time odds tracking
   - ⚡ Player statistics integration

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

### Training Data & Model Specifications

#### Dataset Composition
- Total games: 3,931
- Seasons covered:
  - 2021: 1,163 games (complete)
  - 2022: 1,323 games (complete)
  - 2023: 1,320 games (complete)
  - 2024: 54 games (partial)
  - 2025: 71 games (current)

#### Feature Engineering
1. Core Features:
   - Rolling 5-game scoring metrics
   - Team performance indicators
   - Win rates and streaks
   - Rest days between games

2. Enhanced Features:
   - Win Rate Differential
   - Point Differential Ratio
   - Rest Day Advantage
   - Streak Advantage
   - Recent Form Ratio
   - Win Rate-Rest Interaction
   - Streak-Form Interaction

#### Model Architecture
1. Moneyline Ensemble:
   - Random Forest (500 estimators, max depth 12)
   - Logistic Regression (C=0.8)
   - SVM (RBF kernel, C=10.0)
   - XGBoost (300 estimators, max depth 8)
   - LightGBM (300 estimators, max depth 8)

2. Data Processing:
   - 80/20 train-test split
   - StandardScaler normalization
   - Comprehensive NaN handling
   - Stratified sampling for balanced classes

3. Performance Metrics:
   - Moneyline: 80.0% accuracy
     * Class 0: 0.86 precision, 0.75 recall
     * Class 1: 0.75 precision, 0.86 recall
   - Spread: 15.155 RMSE
   - Totals: 14.435 RMSE

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