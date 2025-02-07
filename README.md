# NBA Data Scraper with Betting Insights

A Python-based NBA data scraper and prediction system that collects game data and generates betting insights using ensemble machine learning models.

## Model Performance

Latest model metrics (as of January 26, 2025):
- Training data cutoff: Previous day in Mexico City timezone (dynamic)
- Historical data cutoff: October 18, 2022 (focusing on recent seasons for stability)
- Moneyline Accuracy: 74.1%
- Spread RMSE: 18.083
- Totals RMSE: 21.734

### Current Limitations
- Model performance currently below target thresholds
- Moneyline predictions require higher confidence threshold (90%+)
- Enhanced validation for first half/quarter predictions
- Strict value rating requirements (minimum 0.70)

### Enhanced Data Management
- Automatic timezone handling using Mexico City (UTC-6) as reference
- Dynamic training data cutoff at previous day's end
- Ensures consistent model training across different timezones
- Prevents data leakage from future games
- Maintains data freshness for predictions

### Automated Model Retraining
- Models automatically retrain after new data is scraped
- Training data cutoff at previous day's end (Mexico City timezone)
- Performance validation against previous versions:
  * Maximum allowed accuracy drop: 2%
  * Maximum allowed RMSE increase: 1.0 points
- Model versioning with backup system:
  * Keeps up to 3 previous versions
  * Tracks performance metrics history
  * Automatic rollback on performance regression
- Minimum requirements for retraining:
  * At least 1 new game
  * Force retrain after 7 days
  * Must meet minimum accuracy thresholds

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
- Implemented data cutoff at October 18, 2022 for improved model stability
- Enhanced feature engineering with interaction features
- Added first half and first quarter predictions
- Improved value rating calculations
- Enhanced error handling and NaN value processing
- Fixed prediction formatting to correctly display confidence values (e.g. "38.6%" instead of "3861.9%")
- Improved rolling statistics handling for future games by using last available historical values
- **XGBoost Early Stopping and Tuning Improvements:**  
  We have resolved previous errors by removing the unsupported `early_stopping_rounds` from the XGBoost constructor and instead passing it during the model fit. A manual parameter search using ParameterSampler and a fixed 80/20 train/validation split now ensures that XGBoost receives a valid, sequential eval_set for early stopping while preserving DataFrame feature names. Separate tuning is now performed for moneyline and totals models, with results stored under distinct keys ("xgboost_moneyline" and "xgboost_totals"). This update resolves both the "unhashable type: 'numpy.ndarray'" error and the "Must have at least 1 validation dataset" error.
- Simplified totals prediction system:
  * Single XGBoost model with optimized hyperparameters
  * Early stopping with validation set
  * Enhanced feature engineering for better predictions
  * Removed ensemble averaging to reduce complexity

## Development Roadmap
1. **Data Quality & Storage** (In Progress)
   - ✓ Basic game scraping
   - ✓ Enhanced prediction models
   - ✓ Data stability improvements
   - ⚡ Team stats improvements
   
2. **Enhanced Data Integration** (Planned)
   - Injury data integration
   - Real-time odds tracking
   - Player statistics integration

3. **Advanced Analytics** (In Progress)
   - ✓ Model stability enhancements
   - ✓ Training data optimization
   - Quarter-by-quarter predictions
   - Player prop predictions
   - Advanced betting metrics

4. **Production Infrastructure** (Planned)
   - Automated updates
   - Performance monitoring
   - API development

### Features

- Automated scraping of NBA game data from basketball-reference.com
- Historical game data collection and management (from October 18, 2022)
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
- Historical statistics (from October 18, 2022 onwards)
- Current season data (up to previous day in Mexico City time)
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
- Training data from October 18, 2022 onwards
- Seasons covered:
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
   - The ensemble now uses a simplified architecture with proven models:
     * RandomForest: 500 estimators, max depth 12 (weight: 2)
     * SVM: RBF kernel, C=10.0 (weight: 1)
     * XGBoost: 300 estimators, max depth 8, learning_rate=0.03 (weight: 2)
   - LightGBM removed due to overfitting concerns
   - Weights optimized for model stability and performance
   - Each model's contribution is proportional to its historical performance

2. Regression Models (Spread and Totals):
   - Both spread and totals predictions now use XGBoost exclusively
   - Early stopping configuration:
     * Patience: 50 rounds
     * Minimum delta: 0.001
     * Best model saving enabled
     * Custom CV iterator with index reset
     * Sequential indices for eval_set
     * Preserved feature names for analysis
   - Spread Model: XGBoost with optimized hyperparameters
   - Totals Model: Single XGBoost model (removed LightGBM ensemble)
   - Version compatibility checks ensure proper functionality
   - Distinct model storage and tracking
   - Enhanced validation metrics

3. Data Processing:
   - 80/20 train-test split
   - StandardScaler normalization
   - Comprehensive NaN handling
   - Stratified sampling for balanced classes
   - Time-based validation to prevent data leakage

4. Performance Metrics (as of January 26, 2025):
   - Moneyline: 74.1% accuracy (target: 85%)
   - Spread: 18.083 RMSE (target: 12.0)
   - Totals: 21.734 RMSE (target: 16.0)
   - Note: Performance metrics are below target thresholds, but enhanced validation and feature engineering are in place

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

- Training data limited to games after October 18, 2022
- Early season team stats may be incomplete
- Predictions require at least 5 games of current season data
- Model accuracy improves as the season progresses
- Quarter/Half predictions are based on historical patterns
- Some bet combinations may be restricted by sportsbooks
- Model performance may fluctuate during validation periods
- High confidence threshold (90%+) reduces number of picks
- Value ratings strictly enforced (minimum 0.70)
- A monkey-patch is applied in nba_predictor.py to handle scikit-learn compatibility (adds _support_missing_values to DecisionTreeClassifier)
- Rolling statistics for future games are now filled using last available historical values to avoid zero-value features

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
   - Data stability improvements (Completed)
   - Real-time prediction updates (Planned)
   - Advanced betting metrics (Planned)
   - Parlay optimization algorithms

4. Production Infrastructure (Planned)
   - Automated updates
   - Performance monitoring
   - API development

### Model Retraining Status

Latest retraining (January 26, 2025):
- Data coverage: October 18, 2022 to January 26, 2025
- Total games: 5,089
- Season breakdown:
  * 2023: 1,320 games (complete)
  * 2024: 54 games (partial)
  * 2025: 1,229 games (current)

Current Performance:
- Moneyline accuracy: 74.1% (target: 85%)
- Spread RMSE: 18.083 (target: 12.0)
- Totals RMSE: 21.734 (target: 16.0)

Validation Thresholds:
- Confidence requirement: 90%+
- Value rating minimum: 0.70
- Enhanced validation for partial game predictions
- Non-overlapping bet combinations enforced
- Automatic rollback if performance drops >2% 

### Model Optimization Strategy

Automated performance optimization with enhanced validation:

1. Data Validation:
   - Required column verification
   - Data type checking
   - Invalid value detection
   - Season format validation
   - Comprehensive error logging

2. Feature Validation:
   - NaN/infinite value detection
   - Win rate range validation [0,1]
   - Form ratio bounds checking [-1,1]
   - Rolling window verification
   - Feature consistency checks

3. Performance Validation:
   - Minimum accuracy requirements
   - Cross-validation scoring
   - Overfitting detection (max 10% train-test gap)
   - Performance regression checks
   - Model stability verification

4. Hyperparameter Tuning:
   - RandomizedSearchCV with 50 iterations
   - Optimized parameter ranges for each model
   - Early stopping with proper index handling
   - Multiple validation periods
   - Automatic model backup

5. Model-Specific Parameters:
   - Random Forest:
     * Estimators: 450-550
     * Max depth: 10-14
     * Min samples split: 2-4
     * Min samples leaf: 1-3
   
   - XGBoost (Moneyline):
     * Estimators: 200-400
     * Max depth: 6-9
     * Learning rate: 0.01-0.1
     * Early stopping: 50 rounds
     * Eval metric: logloss
     * Custom CV with reset indices
   
   - XGBoost (Totals):
     * Estimators: 200-400
     * Max depth: 6-9
     * Learning rate: 0.01-0.1
     * Early stopping: 50 rounds
     * Eval metric: rmse
     * Custom CV with reset indices

6. Safety Measures:
   - Best model versioning
   - Performance thresholds
   - Automatic rollback
   - Cross-validation: 5 folds
   - Multiple test periods

7. Feature Engineering:
   - Rolling statistics (3, 5, 10 games)
   - Win rate calculations
   - Form factor normalization
   - Rest impact with diminishing returns
   - Streak importance with momentum
   - Season-based weighting
   - Interaction features

8. Validation Thresholds:
   - Confidence requirement: 90%+
   - Value rating minimum: 0.70
   - Enhanced validation for partial game predictions
   - Non-overlapping bet combinations enforced
   - Automatic rollback if performance drops >2%

9. XGBoost Early Stopping Improvements:
   - Custom CV iterator with index reset
   - Sequential indices for eval_set
   - Separate moneyline and totals models
   - Distinct model storage and tracking
   - Improved validation metrics
   - Resolved "Must have at least 1 validation dataset" error
   - Preserved feature names for analysis
   - Enhanced error handling and logging

## Data Processing

### Historical Data Handling
- Full historical data is preserved for computing rolling statistics
- Rolling statistics are calculated using complete team history to ensure accurate feature generation
- Data filtering (e.g., cutoff at 2022-10-18) is applied only after computing rolling statistics
- Training data is filtered to recent seasons while maintaining historical context for features

### Feature Generation
- Rolling statistics use 5-game windows for main features
- 3-game windows for recent form analysis
- Both home and away games are considered for team statistics
- League averages are used as fallback for teams with insufficient historical data