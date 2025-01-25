import argparse
import logging
import sys
import signal
import time
from realtime_predictor import RealtimePredictor

def setup_logging():
    """Configure logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('realtime_predictions.log')
        ]
    )

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logging.info("Shutdown signal received")
    if predictor:
        predictor.stop()
    sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description='Run NBA predictions in real-time')
    parser.add_argument('--interval', type=int, default=300,
                      help='Update interval in seconds (default: 300)')
    parser.add_argument('--log-file', default='realtime_predictions.log',
                      help='Path to log file (default: realtime_predictions.log)')
    parser.add_argument('--dry-run', action='store_true',
                      help='Run in dry-run mode using validation data')
    parser.add_argument('--input', help='Input validation dataset for dry-run mode')
    parser.add_argument('--output', help='Output file for dry-run predictions')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        predictor = RealtimePredictor()
        
        if args.dry_run:
            if not args.input or not args.output:
                logging.error("Both --input and --output are required for dry-run mode")
                return 1
                
            logging.info(f"Running in dry-run mode with {args.input}")
            predictor.dry_run(args.input, args.output)
            return 0
            
        logging.info(f"Starting real-time predictions (interval: {args.interval}s)")
        while True:
            predictor.update()
            time.sleep(args.interval)
            
    except KeyboardInterrupt:
        logging.info("Shutting down gracefully...")
    except Exception as e:
        logging.error(f"Error in prediction loop: {str(e)}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main()) 