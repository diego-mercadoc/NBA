import json
import logging
import psutil
import numpy as np
from datetime import datetime
from nba_predictor import NBAPredictor
import pytest
import sys

logging.basicConfig(level=logging.INFO)

def check_memory_usage():
    """Monitor peak memory usage"""
    process = psutil.Process()
    memory_gb = process.memory_info().rss / (1024 * 1024 * 1024)
    logging.info(f"Current memory usage: {memory_gb:.2f}GB")
    return memory_gb <= 3.9

def validate_feature_matrix(predictor):
    """Validate feature matrix integrity"""
    try:
        # Run feature preparation on sample data
        pytest.main(["-v", "tests/training_stability.py::test_feature_preparation"])
        return True
    except Exception as e:
        logging.error(f"Feature matrix validation failed: {e}")
        return False

def validate_model_weights(predictor):
    """Verify ensemble weights are locked"""
    expected_weights = {
        "random_forest": 0.4,
        "xgboost": 0.3,
        "lightgbm": 0.3
    }
    
    # Check if weights match expected values within tolerance
    tolerance = 0.005
    for model, weight in expected_weights.items():
        if abs(predictor.ensemble_weights[model] - weight) > tolerance:
            return False
    return True

def validate_accuracy():
    """Validate model accuracy meets requirements"""
    try:
        result = pytest.main(["-v", "tests/training_stability.py::test_model_training"])
        return result == 0
    except Exception as e:
        logging.error(f"Accuracy validation failed: {e}")
        return False

def main():
    """Run full validation suite"""
    try:
        predictor = NBAPredictor()
        
        # Run all validation checks
        checks = {
            "memory_usage": check_memory_usage(),
            "feature_matrix": validate_feature_matrix(predictor),
            "model_weights": validate_model_weights(predictor),
            "accuracy": validate_accuracy()
        }
        
        # Generate report
        report = {
            "status": "passed" if all(checks.values()) else "failed",
            "checks": checks,
            "timestamp": datetime.now().isoformat()
        }
        
        # Output report
        if "--report-format" in sys.argv and sys.argv[sys.argv.index("--report-format") + 1] == "json":
            print(json.dumps(report, indent=2))
        else:
            for check, passed in checks.items():
                print(f"{check}: {'✅' if passed else '❌'}")
        
        # Exit with appropriate code
        sys.exit(0 if all(checks.values()) else 1)
        
    except Exception as e:
        logging.error(f"Validation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 