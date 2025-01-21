import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, VotingClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, log_loss, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgb
import optuna
import logging
import joblib
from datetime import datetime, timedelta
from sklearn.exceptions import UndefinedMetricWarning, ConvergenceWarning

class NBAPredictor:
    """
    ML-based predictor for NBA games using historical data.
    
    Features:
    - Predicts game winners (moneyline)
    - Predicts point spreads
    - Predicts game totals (over/under)
    - Generates confidence scores for predictions
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.moneyline_model = None
        self.spread_model = None
        self.totals_model = None
        self.feature_columns = [
            # Core game stats
            'Home_Points_Scored_Roll5', 'Home_Points_Allowed_Roll5', 'Home_Point_Diff_Roll5',
            'Away_Points_Scored_Roll5', 'Away_Points_Allowed_Roll5', 'Away_Point_Diff_Roll5',
            'Home_Streak', 'Away_Streak', 'Home_Rest_Days', 'Away_Rest_Days',
            'Home_Win_Rate', 'Away_Win_Rate',
            # Shooting efficiency
            'Home_eFG_Pct', 'Away_eFG_Pct',
            # Advanced metrics
            '3pt_volatility', 'pace_adjusted_offense', 'pace_adjusted_defense',
            'Win_Rate_Diff', 'Point_Diff_Ratio'
        ]
        
        # Initialize base models (parameters will be optimized)
        self.rf_classifier = None
        self.lr_classifier = None
        self.svm_classifier = None
        self.xgb_classifier = None
        self.lgb_classifier = None
        
        # Set up cross-validation
        self.cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    def optimize_hyperparameters(self, X_train, y_train, X_val, y_val, model_type='moneyline'):
        """
        Optimize hyperparameters using Optuna for different model types.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            model_type: Type of model to optimize ('moneyline', 'spread', 'totals')
            
        Returns:
            dict: Best hyperparameters
        """
        def objective_moneyline(trial):
            # Reset index to ensure proper indexing in cross-validation
            X_train_reset = pd.DataFrame(X_train).reset_index(drop=True)
            y_train_reset = pd.Series(y_train).reset_index(drop=True)
            
            # Create cross-validation folds
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            scores = []
            
            for train_idx, val_idx in kf.split(X_train_reset):
                X_fold_train, X_fold_val = X_train_reset.iloc[train_idx], X_train_reset.iloc[val_idx]
                y_fold_train, y_fold_val = y_train_reset.iloc[train_idx], y_train_reset.iloc[val_idx]
                
                # RandomForest parameters - refined search space
                rf_params = {
                    'n_estimators': trial.suggest_int('rf_n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('rf_max_depth', 3, 15),
                    'min_samples_split': trial.suggest_int('rf_min_samples_split', 2, 10),
                    'min_samples_leaf': trial.suggest_int('rf_min_samples_leaf', 1, 5),
                    'class_weight': trial.suggest_categorical('rf_class_weight', ['balanced', None])
                }
                
                # XGBoost parameters - refined search space
                xgb_params = {
                    'n_estimators': trial.suggest_int('xgb_n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('xgb_max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.1, log=True),
                    'subsample': trial.suggest_float('xgb_subsample', 0.7, 1.0),
                    'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.7, 1.0),
                    'min_child_weight': trial.suggest_int('xgb_min_child_weight', 1, 5),
                    'gamma': trial.suggest_float('xgb_gamma', 0, 0.5)
                }
                
                # LightGBM parameters - refined search space
                lgb_params = {
                    'n_estimators': trial.suggest_int('lgb_n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('lgb_max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('lgb_learning_rate', 0.01, 0.1, log=True),
                    'subsample': trial.suggest_float('lgb_subsample', 0.7, 1.0),
                    'colsample_bytree': trial.suggest_float('lgb_colsample_bytree', 0.7, 1.0),
                    'num_leaves': trial.suggest_int('lgb_num_leaves', 20, 50),
                    'min_child_samples': trial.suggest_int('lgb_min_child_samples', 10, 50)
                }
                
                # Initialize models with early stopping
                rf_model = RandomForestClassifier(**rf_params, random_state=42)
                xgb_model = xgb.XGBClassifier(**xgb_params, random_state=42, early_stopping_rounds=10)
                lgb_model = lgb.LGBMClassifier(**lgb_params, random_state=42, early_stopping_rounds=10)
                
                # Cross-validation scores for each model
                cv_scores = []
                for model in [rf_model, xgb_model, lgb_model]:
                    cv_scores.append(self._cross_val_score(X_fold_train, y_fold_train, model))
                
                # Optimize ensemble weights
                w1 = trial.suggest_float('w1', 0.1, 2.0)
                w2 = trial.suggest_float('w2', 0.1, 2.0)
                w3 = trial.suggest_float('w3', 0.1, 2.0)
                
                # Weighted average of model scores
                ensemble_score = (w1*cv_scores[0] + w2*cv_scores[1] + w3*cv_scores[2]) / (w1 + w2 + w3)
                
                # Report intermediate values for pruning
                trial.report(ensemble_score, step=1)
                if trial.should_prune():
                    raise optuna.TrialPruned()
                
                scores.append(ensemble_score)
            
            # Optimize ensemble weights
            w1 = trial.suggest_float('w1', 0.1, 2.0)
            w2 = trial.suggest_float('w2', 0.1, 2.0)
            w3 = trial.suggest_float('w3', 0.1, 2.0)
            
            # Weighted average of model scores
            ensemble_score = (w1*scores[0] + w2*scores[1] + w3*scores[2]) / (w1 + w2 + w3)
            
            # Report intermediate values for pruning
            trial.report(ensemble_score, step=1)
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            return ensemble_score
        
        def objective_regression(trial):
            if model_type == 'spread':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                    'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),
                    'gamma': trial.suggest_float('gamma', 0, 0.5),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0)
                }
                
                model = xgb.XGBRegressor(**params, random_state=42, early_stopping_rounds=10)
            else:  # totals
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                    'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 50),
                    'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0)
                }
                
                model = lgb.LGBMRegressor(**params, random_state=42, early_stopping_rounds=10)
            
            # Cross-validation with early stopping
            cv_scores = []
            for train_idx, val_idx in self.cv.split(X_train, y_train):
                X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
                y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
                
                if isinstance(model, xgb.XGBRegressor):
                    model.fit(
                        X_fold_train, y_fold_train,
                        eval_set=[(X_fold_val, y_fold_val)],
                        verbose=False
                    )
                else:  # LightGBM
                    model.fit(
                        X_fold_train, y_fold_train,
                        eval_set=[(X_fold_val, y_fold_val)],
                        callbacks=[lgb.early_stopping(10)]
                    )
                
                y_pred = model.predict(X_fold_val)
                rmse = np.sqrt(mean_squared_error(y_fold_val, y_pred))
                cv_scores.append(-rmse)  # Negative because we want to maximize
            
            mean_score = np.mean(cv_scores)
            
            # Report intermediate values for pruning
            trial.report(mean_score, step=1)
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            return mean_score
        
        # Create study with pruning
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=5,
            interval_steps=1
        )
        
        if model_type == 'moneyline':
            study = optuna.create_study(
                direction='maximize',
                pruner=pruner,
                study_name='moneyline_optimization'
            )
            objective = objective_moneyline
        else:
            study = optuna.create_study(
                direction='maximize',
                pruner=pruner,
                study_name=f'{model_type}_optimization'
            )
            objective = objective_regression
        
        # Optimize with callbacks for logging
        def logging_callback(study, trial):
            if trial.number % 10 == 0:
                logging.info(f'Trial {trial.number}: Current best score = {study.best_value:.4f}')
        
        n_trials = 50 if model_type == 'moneyline' else 30
        study.optimize(
            objective,
            n_trials=n_trials,
            callbacks=[logging_callback],
            show_progress_bar=True
        )
        
        # Log optimization results
        logging.info(f"\nBest trial for {model_type}:")
        logging.info(f"  Value: {study.best_value:.4f}")
        logging.info("  Params:")
        for key, value in study.best_params.items():
            logging.info(f"    {key}: {value}")
        
        return study.best_params
    
    def prepare_features(self, games_df):
        """Prepare features for ML models with enhanced engineering"""
        df = games_df.copy()
        
        # Load team stats
        try:
            team_stats = pd.read_csv('nba_team_stats_all.csv')
            
            # Check available columns
            available_cols = ['Team', 'Season', 'ORtg', 'DRtg', 'Pace']
            if 'eFG_Pct' in team_stats.columns:
                available_cols.append('eFG_Pct')
            if 'eFG_Pct_Allowed' in team_stats.columns:
                available_cols.append('eFG_Pct_Allowed')
            if '3P_Pct' in team_stats.columns:
                available_cols.append('3P_Pct')
            if '3pt_variance' in team_stats.columns:
                available_cols.append('3pt_variance')
            
            # Merge home team stats
            df = df.merge(
                team_stats[available_cols],
                left_on=['Home_Team', 'Season'],
                right_on=['Team', 'Season'],
                how='left',
                suffixes=('', '_home')
            )
            
            # Rename columns
            if 'eFG_Pct' in available_cols:
                df = df.rename(columns={'eFG_Pct': 'Home_eFG_Pct'})
            if 'eFG_Pct_Allowed' in available_cols:
                df = df.rename(columns={'eFG_Pct_Allowed': 'Home_eFG_Pct_Allowed'})
            if '3P_Pct' in available_cols:
                df = df.rename(columns={'3P_Pct': 'Home_3P_Pct'})
            if '3pt_variance' in available_cols:
                df = df.rename(columns={'3pt_variance': 'Home_3pt_variance'})
            if 'ORtg' in available_cols:
                df = df.rename(columns={'ORtg': 'Home_ORtg'})
            if 'DRtg' in available_cols:
                df = df.rename(columns={'DRtg': 'Home_DRtg'})
            if 'Pace' in available_cols:
                df = df.rename(columns={'Pace': 'Home_Pace'})
            
            # Merge away team stats
            df = df.merge(
                team_stats[available_cols],
                left_on=['Away_Team', 'Season'],
                right_on=['Team', 'Season'],
                how='left',
                suffixes=('', '_away')
            )
            
            # Rename columns
            if 'eFG_Pct' in available_cols:
                df = df.rename(columns={'eFG_Pct': 'Away_eFG_Pct'})
            if 'eFG_Pct_Allowed' in available_cols:
                df = df.rename(columns={'eFG_Pct_Allowed': 'Away_eFG_Pct_Allowed'})
            if '3P_Pct' in available_cols:
                df = df.rename(columns={'3P_Pct': 'Away_3P_Pct'})
            if '3pt_variance' in available_cols:
                df = df.rename(columns={'3pt_variance': 'Away_3pt_variance'})
            if 'ORtg' in available_cols:
                df = df.rename(columns={'ORtg': 'Away_ORtg'})
            if 'DRtg' in available_cols:
                df = df.rename(columns={'DRtg': 'Away_DRtg'})
            if 'Pace' in available_cols:
                df = df.rename(columns={'Pace': 'Away_Pace'})
            
            # Drop redundant columns
            df = df.drop(['Team_home', 'Team_away'], axis=1, errors='ignore')
            
            # Set default values for missing columns
            if 'Home_eFG_Pct' not in df.columns:
                df['Home_eFG_Pct'] = 0.5  # League average approximation
            if 'Away_eFG_Pct' not in df.columns:
                df['Away_eFG_Pct'] = 0.5
            if 'Home_eFG_Pct_Allowed' not in df.columns:
                df['Home_eFG_Pct_Allowed'] = 0.5
            if 'Away_eFG_Pct_Allowed' not in df.columns:
                df['Away_eFG_Pct_Allowed'] = 0.5
            if 'Home_3P_Pct' not in df.columns:
                df['Home_3P_Pct'] = 0.35  # League average approximation
            if 'Away_3P_Pct' not in df.columns:
                df['Away_3P_Pct'] = 0.35
            if 'Home_3pt_variance' not in df.columns:
                df['Home_3pt_variance'] = 0.05  # Reasonable default
            if 'Away_3pt_variance' not in df.columns:
                df['Away_3pt_variance'] = 0.05
            if 'Home_ORtg' not in df.columns:
                df['Home_ORtg'] = 110.0  # League average approximation
            if 'Away_ORtg' not in df.columns:
                df['Away_ORtg'] = 110.0
            if 'Home_DRtg' not in df.columns:
                df['Home_DRtg'] = 110.0
            if 'Away_DRtg' not in df.columns:
                df['Away_DRtg'] = 110.0
            if 'Home_Pace' not in df.columns:
                df['Home_Pace'] = 100.0  # League average approximation
            if 'Away_Pace' not in df.columns:
                df['Away_Pace'] = 100.0
            
            # Fill NaN values with league averages
            for col in df.columns:
                if col.endswith('_Pct') or col.endswith('_variance'):
                    df[col] = df[col].fillna(df[col].mean())
                elif col.endswith('Rtg') or col.endswith('Pace'):
                    df[col] = df[col].fillna(df[col].mean())
        except Exception as e:
            logging.warning(f"Error loading team stats: {str(e)}. Using default values.")
            df['Home_eFG_Pct'] = 0.5
            df['Away_eFG_Pct'] = 0.5
            df['Home_eFG_Pct_Allowed'] = 0.5
            df['Away_eFG_Pct_Allowed'] = 0.5
            df['Home_3P_Pct'] = 0.35
            df['Away_3P_Pct'] = 0.35
            df['Home_3pt_variance'] = 0.05
            df['Away_3pt_variance'] = 0.05
            df['Home_ORtg'] = 110.0
            df['Away_ORtg'] = 110.0
            df['Home_DRtg'] = 110.0
            df['Away_DRtg'] = 110.0
            df['Home_Pace'] = 100.0
            df['Away_Pace'] = 100.0
        
        # Calculate win/loss columns
        df['Home_Win'] = (df['Home_Points'] > df['Away_Points']).astype(int)
        df['Away_Win'] = (df['Away_Points'] > df['Home_Points']).astype(int)
        
        # Calculate win rates with exponential weighting
        for team_type in ['Home', 'Away']:
            if team_type == 'Home':
                team_stats = df.groupby(f'{team_type}_Team')['Home_Win'].agg(['count', 'mean'])
                df[f'{team_type}_Win_Rate'] = df[f'{team_type}_Team'].map(team_stats['mean'])
            else:
                team_stats = df.groupby(f'{team_type}_Team')['Home_Win'].agg(['count', 'mean'])
                df[f'{team_type}_Win_Rate'] = df[f'{team_type}_Team'].map(1 - team_stats['mean'])
        
        # Fill NaN values in win rates with 0.5 (neutral)
        df['Home_Win_Rate'] = df['Home_Win_Rate'].fillna(0.5)
        df['Away_Win_Rate'] = df['Away_Win_Rate'].fillna(0.5)
        
        # Enhanced feature engineering
        df['Win_Rate_Diff'] = df['Home_Win_Rate'] - df['Away_Win_Rate']
        
        # Calculate eFG% mismatch (offensive and defensive)
        df['efg_mismatch_offense'] = df['Home_eFG_Pct'] - df['Away_eFG_Pct']
        df['efg_mismatch_defense'] = df['Away_eFG_Pct_Allowed'] - df['Home_eFG_Pct_Allowed']
        df['efg_mismatch'] = df['efg_mismatch_offense'] + df['efg_mismatch_defense']
        
        # New 3-point volatility feature
        df['3pt_volatility'] = df['Home_3pt_variance'] - df['Away_3pt_variance']
        
        # New pace-adjusted metrics
        df['pace_adjusted_offense'] = df['Home_ORtg'] * df['Home_Pace'] / 100
        df['pace_adjusted_defense'] = df['Away_DRtg'] * df['Away_Pace'] / 100
        
        # Handle potential division by zero or NaN
        df['Away_Point_Diff_Roll5'] = df['Away_Point_Diff_Roll5'].fillna(0)
        df['Home_Point_Diff_Roll5'] = df['Home_Point_Diff_Roll5'].fillna(0)
        df['Point_Diff_Ratio'] = df.apply(
            lambda x: x['Home_Point_Diff_Roll5'] / (x['Away_Point_Diff_Roll5'] + 1e-6)
            if x['Away_Point_Diff_Roll5'] != 0
            else x['Home_Point_Diff_Roll5'],
            axis=1
        )
        
        # Fill NaN values in rest days with median
        df['Home_Rest_Days'] = df['Home_Rest_Days'].fillna(df['Home_Rest_Days'].median())
        df['Away_Rest_Days'] = df['Away_Rest_Days'].fillna(df['Away_Rest_Days'].median())
        df['Rest_Advantage'] = df['Home_Rest_Days'] - df['Away_Rest_Days']
        
        # Fill NaN values in streaks with 0
        df['Home_Streak'] = df['Home_Streak'].fillna(0)
        df['Away_Streak'] = df['Away_Streak'].fillna(0)
        df['Streak_Advantage'] = df['Home_Streak'] - df['Away_Streak']
        
        # Calculate form ratio with NaN handling
        df['Recent_Form_Ratio'] = df.apply(
            lambda x: (x['Home_Point_Diff_Roll5'] + 1e-6) / (x['Away_Point_Diff_Roll5'] + 1e-6)
            if x['Away_Point_Diff_Roll5'] != 0
            else x['Home_Point_Diff_Roll5'],
            axis=1
        )
        
        # Interaction features
        df['Win_Rate_Rest_Interaction'] = df['Win_Rate_Diff'] * df['Rest_Advantage']
        df['Streak_Form_Interaction'] = df['Streak_Advantage'] * df['Recent_Form_Ratio']
        df['eFG_Rest_Interaction'] = df['efg_mismatch'] * df['Rest_Advantage']
        
        # Fill remaining NaN values with 0
        feature_cols = self.feature_columns + [
            'Rest_Advantage', 'Streak_Advantage', 'Recent_Form_Ratio',
            'Win_Rate_Rest_Interaction', 'Streak_Form_Interaction',
            'efg_mismatch', 'efg_mismatch_offense', 'efg_mismatch_defense',
            'eFG_Rest_Interaction'
        ]
        X = df[feature_cols].fillna(0)
        
        return X
    
    def prepare_labels(self, games_df):
        """Prepare labels for different prediction types"""
        played_games = games_df.dropna(subset=['Home_Points', 'Away_Points'])
        
        y_moneyline = (played_games['Home_Points'] > played_games['Away_Points']).astype(int)
        y_spread = played_games['Home_Points'] - played_games['Away_Points']
        y_totals = played_games['Home_Points'] + played_games['Away_Points']
        
        return y_moneyline, y_spread, y_totals
    
    def train_models(self, games_df, test_size=0.2):
        """Train ML models with enhanced ensemble and Bayesian optimization"""
        logging.info("Preparing features and labels...")
        
        played_games = games_df.dropna(subset=['Home_Points', 'Away_Points'])
        X = self.prepare_features(played_games)
        y_moneyline, y_spread, y_totals = self.prepare_labels(played_games)
        
        # Split data with stratification for moneyline
        X_train, X_test, y_ml_train, y_ml_test = train_test_split(
            X, y_moneyline, test_size=test_size, stratify=y_moneyline, random_state=42
        )
        _, _, y_spread_train, y_spread_test = train_test_split(
            X, y_spread, test_size=test_size, random_state=42
        )
        _, _, y_totals_train, y_totals_test = train_test_split(
            X, y_totals, test_size=test_size, random_state=42
        )
        
        # Scale features with updated NaN handling
        self.scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
        # Fill NaN values before scaling
        X_train = pd.DataFrame(X_train).fillna(0)
        X_test = pd.DataFrame(X_test).fillna(0)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Optimize hyperparameters for each model type
        logging.info("Optimizing hyperparameters for moneyline model...")
        ml_params = self.optimize_hyperparameters(
            X_train_scaled, y_ml_train,
            X_test_scaled, y_ml_test,
            model_type='moneyline'
        )
        
        # Initialize models with optimized parameters
        self.rf_classifier = RandomForestClassifier(
            n_estimators=ml_params['rf_n_estimators'],
            max_depth=ml_params['rf_max_depth'],
            min_samples_split=ml_params['rf_min_samples_split'],
            min_samples_leaf=ml_params['rf_min_samples_leaf'],
            random_state=42
        )
        
        self.xgb_classifier = xgb.XGBClassifier(
            n_estimators=ml_params['xgb_n_estimators'],
            max_depth=ml_params['xgb_max_depth'],
            learning_rate=ml_params['xgb_learning_rate'],
            subsample=ml_params['xgb_subsample'],
            colsample_bytree=ml_params['xgb_colsample_bytree'],
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        self.lgb_classifier = lgb.LGBMClassifier(
            n_estimators=ml_params['lgb_n_estimators'],
            max_depth=ml_params['lgb_max_depth'],
            learning_rate=ml_params['lgb_learning_rate'],
            subsample=ml_params['lgb_subsample'],
            colsample_bytree=ml_params['lgb_colsample_bytree'],
            random_state=42
        )
        
        # Train moneyline models with calibration
        logging.info("Training calibrated moneyline models...")
        try:
            # Train base models first
            self.rf_classifier.fit(X_train_scaled, y_ml_train)
            self.xgb_classifier.fit(X_train_scaled, y_ml_train)
            self.lgb_classifier.fit(X_train_scaled, y_ml_train)

            # Then calibrate them
            self.rf_classifier = CalibratedClassifierCV(
                self.rf_classifier, cv='prefit', method='sigmoid'
            ).fit(X_train_scaled, y_ml_train)

            self.xgb_classifier = CalibratedClassifierCV(
                self.xgb_classifier, cv='prefit', method='sigmoid'
            ).fit(X_train_scaled, y_ml_train)

            self.lgb_classifier = CalibratedClassifierCV(
                self.lgb_classifier, cv='prefit', method='sigmoid'
            ).fit(X_train_scaled, y_ml_train)

        except (UndefinedMetricWarning, ConvergenceWarning) as e:
            logging.warning(f"Warning during model calibration: {str(e)}")
            logging.info("Proceeding with uncalibrated models...")
        
        # Get predictions from all models
        rf_proba = self.rf_classifier.predict_proba(X_test_scaled)
        xgb_proba = self.xgb_classifier.predict_proba(X_test_scaled)
        lgb_proba = self.lgb_classifier.predict_proba(X_test_scaled)
        
        # Use optimized weights for ensemble
        ensemble_proba = (
            ml_params['w1']*rf_proba +
            ml_params['w2']*xgb_proba +
            ml_params['w3']*lgb_proba
        ) / (ml_params['w1'] + ml_params['w2'] + ml_params['w3'])
        
        y_ml_pred = (ensemble_proba[:, 1] > 0.5).astype(int)
        
        # Evaluate moneyline model
        ml_accuracy = accuracy_score(y_ml_test, y_ml_pred)
        brier = brier_score_loss(y_ml_test, ensemble_proba[:, 1])
        ml_log_loss = log_loss(y_ml_test, ensemble_proba)
        
        logging.info(f"Enhanced ensemble moneyline model metrics:")
        logging.info(f"Accuracy: {ml_accuracy:.3f}")
        logging.info(f"Brier Score: {brier:.3f}")
        logging.info(f"Log Loss: {ml_log_loss:.3f}")
        logging.info("\nClassification Report:")
        logging.info(classification_report(y_ml_test, y_ml_pred))
        
        # Optimize and train spread model
        logging.info("Optimizing hyperparameters for spread model...")
        spread_params = self.optimize_hyperparameters(
            X_train_scaled, y_spread_train,
            X_test_scaled, y_spread_test,
            model_type='spread'
        )
        
        self.spread_model = xgb.XGBRegressor(**spread_params, random_state=42)
        self.spread_model.fit(X_train_scaled, y_spread_train)
        
        # Evaluate spread model
        y_spread_pred = self.spread_model.predict(X_test_scaled)
        spread_rmse = np.sqrt(mean_squared_error(y_spread_test, y_spread_pred))
        logging.info(f"Enhanced spread model RMSE: {spread_rmse:.3f}")
        
        # Optimize and train totals model
        logging.info("Optimizing hyperparameters for totals model...")
        totals_params = self.optimize_hyperparameters(
            X_train_scaled, y_totals_train,
            X_test_scaled, y_totals_test,
            model_type='totals'
        )
        
        self.totals_model = lgb.LGBMRegressor(**totals_params, random_state=42)
        self.totals_model.fit(X_train_scaled, y_totals_train)
        
        # Evaluate totals model
        y_totals_pred = self.totals_model.predict(X_test_scaled)
        totals_rmse = np.sqrt(mean_squared_error(y_totals_test, y_totals_pred))
        logging.info(f"Enhanced totals model RMSE: {totals_rmse:.3f}")
        
        # Save models
        self.save_models()
        
        return {
            'moneyline_accuracy': ml_accuracy,
            'moneyline_brier': brier,
            'moneyline_log_loss': ml_log_loss,
            'spread_rmse': spread_rmse,
            'totals_rmse': totals_rmse
        }
    
    def save_models(self):
        """Save trained models and scaler"""
        joblib.dump(self.moneyline_model, 'models/moneyline_model.joblib')
        joblib.dump(self.spread_model, 'models/spread_model.joblib')
        joblib.dump(self.totals_model, 'models/totals_model.joblib')
        joblib.dump(self.scaler, 'models/scaler.joblib')
        logging.info("Models saved successfully")
    
    def load_models(self):
        """Load trained models and scaler"""
        try:
            self.moneyline_model = joblib.load('models/moneyline_model.joblib')
            self.spread_model = joblib.load('models/spread_model.joblib')
            self.totals_model = joblib.load('models/totals_model.joblib')
            self.scaler = joblib.load('models/scaler.joblib')
            logging.info("Models loaded successfully")
            return True
        except Exception as e:
            logging.error(f"Error loading models: {str(e)}")
            return False
    
    def predict_games(self, games_df):
        """Generate predictions for games with enhanced ensemble"""
        X = self.prepare_features(games_df)
        X_scaled = self.scaler.transform(X)
        
        predictions = pd.DataFrame()
        predictions['Home_Team'] = games_df['Home_Team']
        predictions['Away_Team'] = games_df['Away_Team']
        predictions['Date'] = games_df['Date']
        
        # Get predictions from all models
        sklearn_proba = self.moneyline_model.predict_proba(X_scaled)
        xgb_proba = self.xgb_classifier.predict_proba(X_scaled)
        lgb_proba = self.lgb_classifier.predict_proba(X_scaled)
        
        # Weighted average of probabilities
        ensemble_proba = (2*sklearn_proba + 2*xgb_proba + 2*lgb_proba) / 6
        predictions['Home_Win_Prob'] = ensemble_proba[:, 1]
        predictions['Away_Win_Prob'] = ensemble_proba[:, 0]
        predictions['Moneyline_Pick'] = (predictions['Home_Win_Prob'] > 0.5).astype(int)
        
        # Spread and totals predictions
        predictions['Predicted_Spread'] = self.spread_model.predict(X_scaled)
        predictions['Predicted_Total'] = self.totals_model.predict(X_scaled)
        
        # First Half predictions (based on historical patterns)
        predictions['First_Half_Total'] = predictions['Predicted_Total'] * 0.52
        predictions['First_Half_Spread'] = predictions['Predicted_Spread'] * 0.48
        
        # Quarter predictions
        predictions['Avg_Quarter_Points'] = predictions['Predicted_Total'] / 4
        predictions['First_Quarter_Total'] = predictions['Predicted_Total'] * 0.24
        predictions['First_Quarter_Spread'] = predictions['Predicted_Spread'] * 0.45
        
        # Enhanced confidence calculation using ensemble agreement
        predictions['Moneyline_Confidence'] = np.maximum(
            predictions['Home_Win_Prob'],
            predictions['Away_Win_Prob']
        )
        
        # Format predictions
        predictions['Formatted_Predictions'] = predictions.apply(
            lambda x: self._format_prediction(x), axis=1
        )
        
        return predictions
    
    def _format_prediction(self, row):
        """Format prediction row into readable string"""
        home_prob = row['Home_Win_Prob'] * 100
        away_prob = row['Away_Win_Prob'] * 100
        spread = row['Predicted_Spread']
        total = row['Predicted_Total']
        
        pick = f"{row['Home_Team']} ML" if row['Moneyline_Pick'] else f"{row['Away_Team']} ML"
        confidence = row['Moneyline_Confidence'] * 100
        
        # Format quarter and half predictions
        first_half_total = row['First_Half_Total']
        first_half_spread = row['First_Half_Spread']
        first_q_total = row['First_Quarter_Total']
        first_q_spread = row['First_Quarter_Spread']
        
        return (
            f"Game: {row['Away_Team']} @ {row['Home_Team']}\n"
            f"Moneyline: {pick} ({confidence:.1f}% confidence)\n"
            f"Win Probabilities: {row['Home_Team']}: {home_prob:.1f}%, "
            f"{row['Away_Team']}: {away_prob:.1f}%\n"
            f"Full Game:\n"
            f"  - Spread: {row['Home_Team']} {spread:+.1f}\n"
            f"  - Total: {total:.1f}\n"
            f"First Half:\n"
            f"  - Spread: {row['Home_Team']} {first_half_spread:+.1f}\n"
            f"  - Total: {first_half_total:.1f}\n"
            f"First Quarter:\n"
            f"  - Spread: {row['Home_Team']} {first_q_spread:+.1f}\n"
            f"  - Total: {first_q_total:.1f}"
        )
    
    def get_best_bets(self, predictions_df, confidence_threshold=0.90, min_value_rating=0.70):
        """Filter and return only the highest confidence bets with enhanced value ratings"""
        best_bets = []
        
        # Moneyline bets with stricter confidence requirements
        ml_bets = predictions_df[
            (predictions_df['Moneyline_Confidence'] > confidence_threshold)
        ].copy()
        
        for _, game in ml_bets.iterrows():
            # Enhanced value rating calculation
            value_rating = self._calculate_value_rating(game)
            if value_rating > min_value_rating:
                bet = {
                    'Game': f"{game['Away_Team']} @ {game['Home_Team']}",
                    'Bet_Type': 'Moneyline',
                    'Prediction': f"{game['Home_Team']} ML" if game['Moneyline_Pick'] else f"{game['Away_Team']} ML",
                    'Confidence': game['Moneyline_Confidence'],
                    'Value_Rating': value_rating
                }
                best_bets.append(bet)
            
            # First Half total with enhanced validation
            if abs(game['First_Half_Total'] - (game['Predicted_Total'] * 0.52)) < 3:  # Tighter threshold
                half_confidence = min(0.95, game['Moneyline_Confidence'] * 1.1)  # Cap at 95%
                half_value = self._calculate_half_value(game)
                if half_confidence > confidence_threshold and half_value > min_value_rating:
                    bet = {
                        'Game': f"{game['Away_Team']} @ {game['Home_Team']}",
                        'Bet_Type': 'First Half Total',
                        'Prediction': f"Over {game['First_Half_Total']:.1f}",
                        'Confidence': half_confidence,
                        'Value_Rating': half_value
                    }
                    best_bets.append(bet)
            
            # First Quarter total with enhanced validation
            if abs(game['First_Quarter_Total'] - (game['Predicted_Total'] * 0.24)) < 1.5:  # Tighter threshold
                quarter_confidence = min(0.95, game['Moneyline_Confidence'] * 1.05)  # Cap at 95%
                quarter_value = self._calculate_quarter_value(game)
                if quarter_confidence > confidence_threshold and quarter_value > min_value_rating:
                    bet = {
                        'Game': f"{game['Away_Team']} @ {game['Home_Team']}",
                        'Bet_Type': 'First Quarter Total',
                        'Prediction': f"Over {game['First_Quarter_Total']:.1f}",
                        'Confidence': quarter_confidence,
                        'Value_Rating': quarter_value
                    }
                    best_bets.append(bet)
        
        return pd.DataFrame(best_bets)
    
    def _calculate_value_rating(self, row):
        """Enhanced value rating calculation with more factors and stricter thresholds"""
        # Base value from probability margin
        prob_margin = abs(row['Home_Win_Prob'] - row['Away_Win_Prob'])
        value_rating = row['Moneyline_Confidence'] * prob_margin
        
        # Enhanced adjustments with stricter thresholds
        if all(col in row for col in ['Home_Point_Diff_Roll5', 'Away_Point_Diff_Roll5', 'Home_Rest_Days', 'Away_Rest_Days']):
            # Form factor with exponential weighting
            form_diff = abs(row['Home_Point_Diff_Roll5'] - row['Away_Point_Diff_Roll5'])
            form_factor = np.tanh(form_diff / 10)  # Normalized between -1 and 1
            
            # Rest advantage with diminishing returns
            rest_diff = abs(row['Home_Rest_Days'] - row['Away_Rest_Days'])
            rest_factor = np.tanh(rest_diff / 3)  # Normalized between -1 and 1
            
            # Streak impact with momentum consideration
            streak_diff = abs(row.get('Home_Streak', 0) - row.get('Away_Streak', 0))
            streak_factor = np.tanh(streak_diff / 5)  # Normalized between -1 and 1
            
            # Recent performance weight (last 3 games)
            recent_weight = 1.2 if form_factor > 0.5 else 1.0
            
            # Combine factors with weighted importance
            value_rating *= (1 + 0.3*form_factor + 0.2*rest_factor + 0.2*streak_factor) * recent_weight
        
        # Additional confidence boost for extreme differentials
        if prob_margin > 0.4:  # 40% probability difference
            value_rating *= 1.1
        
        # Cap the value rating at 1.0
        return min(value_rating, 1.0)
    
    def _calculate_half_value(self, row):
        """Calculate value rating for first half totals"""
        base_value = self._calculate_value_rating(row)
        
        # Adjust based on first half scoring patterns
        if 'First_Half_Total' in row and 'Predicted_Total' in row:
            half_ratio = row['First_Half_Total'] / row['Predicted_Total']
            if 0.50 <= half_ratio <= 0.54:  # Ideal range
                base_value *= 1.1
            elif 0.48 <= half_ratio <= 0.56:  # Acceptable range
                base_value *= 1.0
            else:
                base_value *= 0.8
        
        return min(base_value * 0.95, 1.0)  # Slightly lower confidence for halves
    
    def _calculate_quarter_value(self, row):
        """Calculate value rating for first quarter totals"""
        base_value = self._calculate_value_rating(row)
        
        # Adjust based on first quarter scoring patterns
        if 'First_Quarter_Total' in row and 'Predicted_Total' in row:
            quarter_ratio = row['First_Quarter_Total'] / row['Predicted_Total']
            if 0.23 <= quarter_ratio <= 0.25:  # Ideal range
                base_value *= 1.1
            elif 0.22 <= quarter_ratio <= 0.26:  # Acceptable range
                base_value *= 1.0
            else:
                base_value *= 0.8
        
        return min(base_value * 0.90, 1.0)  # Lower confidence for quarters
    
    def _cross_val_score(self, X_train, y_train, model):
        """Calculate cross-validation score for a model."""
        if isinstance(model, xgb.XGBClassifier):
            model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train)],
                verbose=False
            )
        elif isinstance(model, lgb.LGBMClassifier):
            model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train)],
                callbacks=[lgb.early_stopping(10)]
            )
        else:
            model.fit(X_train, y_train)
        
        proba = model.predict_proba(X_train)
        accuracy = accuracy_score(y_train, proba[:, 1] > 0.5)
        log_loss_val = log_loss(y_train, proba)
        brier = brier_score_loss(y_train, proba[:, 1])
        
        # Combined metric
        return 0.4 * accuracy - 0.3 * log_loss_val - 0.3 * brier 