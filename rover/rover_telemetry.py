import os
import json
import time

class RoverTelemetryLogger:
    def __init__(self, filepath="rover_telemetry.json"):
        self.filepath = filepath
        self.log_buffer = []
        self.last_write_time = 0.0

    def add_record(self, telemetry_dict):
        record = {
            "timestamp": round(time.time(), 3),
            **telemetry_dict
        }
        self.log_buffer.append(record)

    def write_to_disk(self):
        """Write buffer contents to the file. Limit to every 1 second to avoid heavy disk I/O."""
        current_time = time.time()
        if current_time - self.last_write_time < 1.0 or not self.log_buffer:
            return

        try:
            # Read existing records
            existing_data = []
            if os.path.exists(self.filepath):
                with open(self.filepath, 'r') as f:
                    try:
                        existing_data = json.load(f)
                    except json.JSONDecodeError:
                        # File is corrupt or empty
                        existing_data = []

            # Append new records
            combined_data = existing_data + self.log_buffer
            
            # Keep file size reasonable (max 1000 lines)
            if len(combined_data) > 1000:
                combined_data = combined_data[-1000:]

            with open(self.filepath, 'w') as f:
                json.dump(combined_data, f, indent=2)

            self.log_buffer.clear()
            self.last_write_time = current_time
        except Exception as e:
            print(f"[WARN] Failed to write rover telemetry: {e}")
