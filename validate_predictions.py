import argparse
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
import psutil
import sys
from nba_predictor import NBAPredictor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def check_memory_usage():
    """Monitor memory usage"""
    process = psutil.Process()
    memory_gb = process.memory_info().rss / (1024 * 1024 * 1024)
    if memory_gb > 3.8:
        logging.warning(f"Memory usage warning: {memory_gb:.2f}GB")
    if memory_gb > 3.95:
        raise MemoryError(f"Memory limit exceeded: {memory_gb:.2f}GB")
    return memory_gb

def load_predictions(file_path):
    """Load predictions from JSON file"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading predictions: {str(e)}")
        return None

def validate_prediction_stability(predictions, threshold=0.95):
    """Validate prediction stability across multiple runs"""
    if not predictions or 'runs' not in predictions:
        return False
    
    runs = predictions['runs']
    if len(runs) < 2:
        logging.warning("Not enough runs to validate stability")
        return False
    
    agreements = []
    for i in range(len(runs)-1):
        for j in range(i+1, len(runs)):
            agreement = np.mean([
                r1['pick'] == r2['pick'] 
                for r1, r2 in zip(runs[i]['predictions'], runs[j]['predictions'])
            ])
            agreements.append(agreement)
    
    avg_agreement = np.mean(agreements)
    logging.info(f"Average prediction agreement: {avg_agreement:.3f}")
    return avg_agreement >= threshold

def validate_correlations(predictions):
    """Validate required correlations in predictions"""
    required_correlations = {
        'team_strength_win_rate': 0.6,
        'points_scored_ortg': 0.7,
        'recent_form_streak': 0.5
    }
    
    if not predictions or 'correlations' not in predictions:
        return False
    
    correlations = predictions['correlations']
    all_valid = True
    
    for metric, required in required_correlations.items():
        if metric not in correlations:
            logging.error(f"Missing correlation metric: {metric}")
            all_valid = False
            continue
            
        actual = correlations[metric]
        if actual < required:
            logging.warning(
                f"Correlation below threshold for {metric}: "
                f"{actual:.3f} < {required}"
            )
            all_valid = False
    
    return all_valid

def validate_format_strings(predictions):
    """Check for format string errors in predictions"""
    try:
        # Check all string fields for proper formatting
        for run in predictions.get('runs', []):
            for pred in run.get('predictions', []):
                # Verify numeric formatting
                assert isinstance(pred.get('confidence', 0), (int, float))
                assert 0 <= pred.get('confidence', 0) <= 1
                
                # Verify string formatting
                assert isinstance(pred.get('pick', ''), str)
                assert '@' in pred.get('game', '')
                
        return True
    except Exception as e:
        logging.error(f"Format validation failed: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Validate prediction results')
    parser.add_argument('--source', required=True, help='Path to prediction results JSON')
    parser.add_argument('--memory-limit', type=float, default=3.8, help='Memory limit in GB')
    args = parser.parse_args()
    
    try:
        # Monitor memory
        memory_gb = check_memory_usage()
        logging.info(f"Initial memory usage: {memory_gb:.2f}GB")
        
        # Load and validate predictions
        predictions = load_predictions(args.source)
        if not predictions:
            sys.exit(1)
        
        # Run validations
        validations = {
            'memory': bool(memory_gb <= args.memory_limit),
            'stability': bool(validate_prediction_stability(predictions)),
            'correlations': bool(validate_correlations(predictions)),
            'format': bool(validate_format_strings(predictions))
        }
        
        # Generate report
        report = {
            'timestamp': datetime.now().isoformat(),
            'validations': validations,
            'memory_gb': float(memory_gb),
            'status': 'passed' if all(validations.values()) else 'failed'
        }
        
        # Save report
        report_file = f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logging.info(f"\nValidation Report:")
        for check, passed in validations.items():
            logging.info(f"  {check}: {'✓' if passed else '✗'}")
        
        # Exit with appropriate code
        sys.exit(0 if report['status'] == 'passed' else 1)
        
    except Exception as e:
        logging.error(f"Validation failed: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 