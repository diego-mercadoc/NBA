# NBA Data Scraper with Betting Insights

A Python-based NBA data scraper and prediction system that collects game data and generates betting insights using ensemble machine learning models.

## Known Issues
- Model training stability improvements in progress
- Temporary timeout increases during validation
- Enhanced feature validation being implemented
- Memory optimization updates ongoing

## Model Performance

Latest model metrics (as of January 18, 2025):
- Moneyline Accuracy: 80.0%
- Moneyline Brier Score: 0.175
- Moneyline Log Loss: 0.502
- Spread RMSE: 15.155
- Totals RMSE: 14.435

### Enhanced Model Architecture
- Bayesian optimization (Optuna) for hyperparameter tuning
- Probability calibration using isotonic regression
- Optimized ensemble weights through cross-validation
- Improved metric tracking (accuracy, Brier score, log loss)
- Automated model performance history tracking

### Enhanced Confidence Measures
- Increased confidence threshold to 90% (from 65%)
- Minimum value rating requirement of 70%
- Normalized form factors using tanh transformation
- Exponential weighting for recent performance
- Stricter validation for half/quarter predictions
- Additional confidence boost for strong differentials
- Probability calibration for better confidence estimates

### Prediction Types
- **Moneyline**: Game winner predictions with enhanced confidence scores
  - Calibrated probabilities for better decision making
  - Ensemble weighting optimized through Bayesian search
- **Spread**: Point spread predictions for full game
  - Optimized XGBoost model with cross-validation
  - Enhanced feature importance analysis
- **Totals**: Over/under predictions for full game
  - Optimized LightGBM model with Bayesian tuning
  - Improved scoring pattern recognition
- **First Half**: Spread and total predictions (52% of full game total)
  - Enhanced validation (±3 points from expected)
  - Confidence capped at 95%
- **First Quarter**: Spread and total predictions (24% of full game total)
  - Tighter validation (±1.5 points from expected)
  - Confidence capped at 95%

### Enhanced Features
- **3-Point Volatility**: Measures team shooting consistency vs opponent defense
  - Rolling 10-game standard deviation of 3P%
  - Identifies teams with stable vs streaky shooting
  - Used to adjust confidence in predictions
- **Pace-Adjusted Metrics**: Combines offensive/defensive ratings with tempo
  - Normalizes team performance by possessions
  - Better captures true offensive/defensive efficiency
  - Improves accuracy of totals predictions
- Ensemble learning combining Random Forest, XGBoost, and LightGBM models
- Bayesian optimization for all model hyperparameters
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

## Recent Updates
- Implemented Bayesian optimization for hyperparameter tuning
- Added probability calibration for better confidence estimates
- Enhanced metric tracking and model history
- Improved ensemble weighting through optimization
- Enhanced error handling and NaN value processing
- Implemented 3-point shooting variance analysis
- Added pace-adjusted performance metrics
- Enhanced feature engineering pipeline

## Development Roadmap
1. **Data Quality & Storage** (In Progress)
   - ✓ Basic game scraping
   - ✓ Enhanced prediction models
   - ✓ Bayesian optimization
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
  - Ensemble moneyline predictions (80.0% accuracy)
  - Enhanced spread predictions (RMSE: 15.155 points)
  - Optimized totals predictions (RMSE: 14.435 points)
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
- 3-point shooting statistics
- Pace and efficiency metrics

### Prediction System

The system uses specialized models for different markets:

1. Ensemble Moneyline Model:
   - Random Forest Classifier (Bayesian optimized)
   - XGBoost Classifier (Bayesian optimized)
   - LightGBM Classifier (Bayesian optimized)
   - Optimized ensemble weights
   - Probability calibration
   - Enhanced feature engineering:
     * Win rates with exponential weighting
     * 3-point shooting variance analysis
     * Pace-adjusted performance metrics
     * Rest advantage with diminishing returns
     * Streak impact with momentum consideration
     * Recent performance weighting (last 3 games)

2. Enhanced Spread Model:
   - XGBoost Regressor
   - Bayesian hyperparameter optimization
   - Cross-validation for robustness
   - Incorporates pace-adjusted metrics
   - Accounts for shooting variance

3. Improved Totals Model:
   - LightGBM Regressor
   - Bayesian hyperparameter optimization
   - Advanced feature engineering
   - Pace-normalized scoring projections
   - 3-point variance consideration

4. First Half/Quarter Models:
   - Historical pattern analysis
   - Scoring distribution modeling
   - Pace-adjusted predictions
   - Shooting variance impact analysis

Predictions include:
- Win probabilities for both teams (calibrated)
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
   - 3-point shooting variance
   - Pace-adjusted ratings

2. Enhanced Features:
   - Win Rate Differential
   - Point Differential Ratio
   - Rest Day Advantage
   - Streak Advantage
   - Recent Form Ratio
   - Win Rate-Rest Interaction
   - Streak-Form Interaction
   - 3-Point Volatility
   - Pace-Adjusted Offense/Defense

#### Model Architecture
1. Moneyline Ensemble:
   - Random Forest (Bayesian optimized)
   - XGBoost (Bayesian optimized)
   - LightGBM (Bayesian optimized)
   - Optimized ensemble weights
   - Probability calibration

2. Data Processing:
   - 80/20 train-test split
   - StandardScaler normalization
   - Comprehensive NaN handling
   - Stratified sampling for balanced classes

3. Performance Metrics:
   - Moneyline:
     * Accuracy: 80.0%
     * Brier Score: 0.175
     * Log Loss: 0.502
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