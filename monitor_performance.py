import json
import logging
import time
import psutil
from datetime import datetime, timedelta
import numpy as np
from nba_predictor import NBAPredictor
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class PerformanceMonitor:
    def __init__(self, duration_hours=24, alert_thresholds=None):
        self.duration = timedelta(hours=duration_hours)
        self.start_time = datetime.now()
        self.predictor = NBAPredictor()
        self.thresholds = alert_thresholds or {
            "accuracy_drop": 0.77,
            "memory_spike": 3.8  # Updated to match StabilityGuardrails
        }
        self.metrics_history = []
    
    def check_memory(self):
        """Monitor memory usage"""
        process = psutil.Process()
        memory_gb = process.memory_info().rss / (1024 * 1024 * 1024)
        if memory_gb >= self.thresholds["memory_spike"]:
            self.alert(f"Memory spike detected: {memory_gb:.2f}GB (threshold: {self.thresholds['memory_spike']}GB)")
        return memory_gb
    
    def check_accuracy(self):
        """Monitor prediction accuracy"""
        try:
            metrics = self.predictor.validate_predictions()
            if metrics["moneyline_accuracy"] < self.thresholds["accuracy_drop"]:
                self.alert(
                    f"Accuracy drop detected: {metrics['moneyline_accuracy']:.3f} "
                    f"(threshold: {self.thresholds['accuracy_drop']:.3f})"
                )
            return metrics["moneyline_accuracy"]
        except Exception as e:
            logging.error(f"Error checking accuracy: {str(e)}")
            return None
    
    def alert(self, message):
        """Send alert for critical issues"""
        logging.warning(f"ALERT: {message}")
        # In production, this would send to monitoring system
    
    def record_metrics(self):
        """Record current performance metrics"""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "memory_gb": self.check_memory(),
            "accuracy": self.check_accuracy()
        }
        self.metrics_history.append(metrics)
        return metrics
    
    def generate_report(self):
        """Generate monitoring report"""
        if not self.metrics_history:
            return {"status": "error", "message": "No metrics recorded"}
        
        memory_values = [m["memory_gb"] for m in self.metrics_history]
        accuracy_values = [m["accuracy"] for m in self.metrics_history if m["accuracy"] is not None]
        
        report = {
            "status": "healthy",
            "duration_hours": self.duration.total_seconds() / 3600,
            "metrics": {
                "memory": {
                    "mean": np.mean(memory_values) if memory_values else None,
                    "max": np.max(memory_values) if memory_values else None,
                    "latest": memory_values[-1] if memory_values else None
                },
                "accuracy": {
                    "mean": np.mean(accuracy_values) if accuracy_values else None,
                    "min": np.min(accuracy_values) if accuracy_values else None,
                    "latest": accuracy_values[-1] if accuracy_values else None
                }
            },
            "thresholds": self.thresholds,
            "alerts_triggered": len([m for m in self.metrics_history 
                                  if m["memory_gb"] >= self.thresholds["memory_spike"] or 
                                  (m["accuracy"] is not None and m["accuracy"] < self.thresholds["accuracy_drop"])])
        }
        
        # Determine overall status
        if (report["metrics"]["memory"]["max"] >= self.thresholds["memory_spike"] or
            (report["metrics"]["accuracy"]["min"] is not None and 
             report["metrics"]["accuracy"]["min"] < self.thresholds["accuracy_drop"])):
            report["status"] = "warning"
        
        return report
    
    def run(self):
        """Run monitoring for specified duration"""
        logging.info(f"Starting performance monitoring for {self.duration.total_seconds()/3600:.1f} hours")
        
        while datetime.now() - self.start_time < self.duration:
            try:
                metrics = self.record_metrics()
                logging.info(
                    f"Current metrics: Memory={metrics['memory_gb']:.2f}GB, "
                    f"Accuracy={metrics['accuracy']:.3f if metrics['accuracy'] is not None else 'N/A'}"
                )
                time.sleep(300)  # Check every 5 minutes
            except KeyboardInterrupt:
                logging.info("Monitoring interrupted by user")
                break
            except Exception as e:
                logging.error(f"Error during monitoring: {e}")
                time.sleep(60)  # Wait a minute before retrying
        
        return self.generate_report()

def main():
    # Parse command line arguments
    duration = 24  # default 24 hours
    if "--duration" in sys.argv:
        try:
            duration = float(sys.argv[sys.argv.index("--duration") + 1])
        except:
            logging.error("Invalid duration specified")
            sys.exit(1)
    
    # Initialize and run monitor
    monitor = PerformanceMonitor(duration_hours=duration)
    report = monitor.run()
    
    # Output report
    print(json.dumps(report, indent=2))
    
    # Exit with appropriate code
    sys.exit(0 if report["status"] == "healthy" else 1)

if __name__ == "__main__":
    main() 