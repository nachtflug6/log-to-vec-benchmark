"""
Toy Log Generator for PLC-like System

This script generates synthetic time-stamped log files that simulate
a Programmable Logic Controller (PLC) or similar industrial system.
The logs contain various event types with realistic timing patterns.
"""

import random
import pandas as pd
from datetime import datetime, timedelta
import argparse
from pathlib import Path


class PLCLogGenerator:
    """Generate synthetic PLC-like log data."""
    
    # Event types that a typical PLC system might produce
    EVENT_TYPES = {
        "SYSTEM_START": 0.02,
        "SYSTEM_STOP": 0.01,
        "SENSOR_READ": 0.30,
        "ACTUATOR_CMD": 0.25,
        "ALARM_TEMP": 0.03,
        "ALARM_PRESSURE": 0.03,
        "ALARM_CLEAR": 0.02,
        "COMMUNICATION_OK": 0.15,
        "COMMUNICATION_ERROR": 0.05,
        "WATCHDOG_TICK": 0.10,
        "CONFIG_CHANGE": 0.01,
        "MAINTENANCE_MODE": 0.01,
        "PRODUCTION_CYCLE_START": 0.01,
        "PRODUCTION_CYCLE_END": 0.01,
    }
    
    # Severity levels
    SEVERITY_LEVELS = ["INFO", "WARNING", "ERROR", "CRITICAL"]
    
    # Components that generate logs
    COMPONENTS = [
        "MainController",
        "TemperatureSensor",
        "PressureSensor",
        "Actuator1",
        "Actuator2",
        "NetworkModule",
        "SafetyMonitor",
    ]
    
    def __init__(self, seed=42):
        """Initialize the log generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        random.seed(seed)
        self.event_weights = list(self.EVENT_TYPES.values())
        self.event_names = list(self.EVENT_TYPES.keys())
        
    def _get_severity(self, event_type):
        """Determine severity level based on event type."""
        if "ALARM" in event_type or "ERROR" in event_type:
            return random.choice(["WARNING", "ERROR", "CRITICAL"])
        elif event_type in ["SYSTEM_STOP", "COMMUNICATION_ERROR"]:
            return random.choice(["WARNING", "ERROR"])
        else:
            return "INFO"
    
    def _generate_event_data(self, event_type):
        """Generate additional data payload for an event."""
        data = {}
        
        if "SENSOR" in event_type:
            if "TEMP" in event_type.upper() or event_type == "SENSOR_READ":
                data["temperature"] = round(random.uniform(20.0, 85.0), 2)
            if "PRESSURE" in event_type.upper() or event_type == "SENSOR_READ":
                data["pressure"] = round(random.uniform(1.0, 10.0), 2)
                
        if "ACTUATOR" in event_type:
            data["position"] = round(random.uniform(0, 100), 1)
            data["state"] = random.choice(["OPEN", "CLOSED", "MOVING"])
            
        if "ALARM" in event_type:
            data["threshold_exceeded"] = True
            data["value"] = round(random.uniform(80, 100), 2)
            
        return str(data) if data else ""
    
    def generate_logs(self, num_events=10000, start_time=None, 
                     time_variance=1.0, anomaly_rate=0.05):
        """Generate a sequence of log events.
        
        Args:
            num_events: Number of log events to generate
            start_time: Starting timestamp (defaults to now)
            time_variance: Standard deviation for time between events (seconds)
            anomaly_rate: Probability of injecting an anomalous pattern
            
        Returns:
            pandas.DataFrame with columns: timestamp, event_type, component, 
                                          severity, message, data
        """
        if start_time is None:
            start_time = datetime.now()
        
        logs = []
        current_time = start_time
        system_state = "RUNNING"  # Track system state for realistic sequences
        
        for i in range(num_events):
            # Introduce anomalies occasionally
            if random.random() < anomaly_rate:
                # Anomaly: burst of errors
                event_type = random.choice([
                    "COMMUNICATION_ERROR", 
                    "ALARM_TEMP", 
                    "ALARM_PRESSURE"
                ])
            else:
                # Normal operation
                event_type = random.choices(
                    self.event_names, 
                    weights=self.event_weights
                )[0]
            
            # Handle state transitions
            if event_type == "SYSTEM_START":
                system_state = "RUNNING"
            elif event_type == "SYSTEM_STOP":
                system_state = "STOPPED"
            elif event_type == "MAINTENANCE_MODE":
                system_state = "MAINTENANCE"
            
            # Generate event details
            component = random.choice(self.COMPONENTS)
            severity = self._get_severity(event_type)
            data = self._generate_event_data(event_type)
            message = f"{event_type} in {system_state} state"
            
            logs.append({
                "timestamp": current_time,
                "event_type": event_type,
                "component": component,
                "severity": severity,
                "message": message,
                "data": data,
            })
            
            # Time advancement with variance
            # Normal time between events: ~0.5 to 2 seconds
            base_interval = random.uniform(0.5, 2.0)
            time_delta = abs(random.gauss(base_interval, time_variance))
            current_time += timedelta(seconds=time_delta)
        
        df = pd.DataFrame(logs)
        return df
    
    def generate_multiple_scenarios(self, output_dir, scenarios=None):
        """Generate multiple log files with different scenarios.
        
        Args:
            output_dir: Directory to save log files
            scenarios: List of scenario configurations
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if scenarios is None:
            scenarios = [
                {"name": "normal", "num_events": 5000, "anomaly_rate": 0.01},
                {"name": "high_anomaly", "num_events": 5000, "anomaly_rate": 0.15},
                {"name": "long_sequence", "num_events": 20000, "anomaly_rate": 0.05},
            ]
        
        for scenario in scenarios:
            name = scenario.pop("name")
            df = self.generate_logs(**scenario)
            output_file = output_dir / f"toy_logs_{name}.csv"
            df.to_csv(output_file, index=False)
            print(f"Generated {len(df)} log events -> {output_file}")
            print(f"  Time span: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print(f"  Unique event types: {df['event_type'].nunique()}")
            print(f"  Event type distribution:\n{df['event_type'].value_counts().head()}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic PLC-like log data"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory for generated logs"
    )
    parser.add_argument(
        "--num-events",
        type=int,
        default=10000,
        help="Number of log events to generate"
    )
    parser.add_argument(
        "--anomaly-rate",
        type=float,
        default=0.05,
        help="Rate of anomalous events (0.0 to 1.0)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--scenarios",
        action="store_true",
        help="Generate multiple predefined scenarios"
    )
    
    args = parser.parse_args()
    
    generator = PLCLogGenerator(seed=args.seed)
    
    if args.scenarios:
        generator.generate_multiple_scenarios(args.output_dir)
    else:
        df = generator.generate_logs(
            num_events=args.num_events,
            anomaly_rate=args.anomaly_rate
        )
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "toy_logs.csv"
        df.to_csv(output_file, index=False)
        print(f"Generated {len(df)} log events -> {output_file}")
        print(f"\nSample logs:")
        print(df.head(10))
        print(f"\nEvent type distribution:")
        print(df["event_type"].value_counts())


if __name__ == "__main__":
    main()
