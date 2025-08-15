import logging
import os
import time
import torch
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from bayes import BayesianNN, elbo_loss

# Model parameters
WINDOW_SIZE = 10
NUM_FEATURES = 1
HIDDEN_DIM = 50
OUTPUT_DIM = 1
MODEL_PATH = 'model.pth'

# Load the model
model = BayesianNN(WINDOW_SIZE, NUM_FEATURES, HIDDEN_DIM, OUTPUT_DIM)
if os.path.exists(MODEL_PATH):
    logging.info(f"Loading model from {MODEL_PATH}")
    model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# InfluxDB configuration
INFLUXDB_URL = os.getenv("INFLUXDB_URL", "http://localhost:8086")
INFLUXDB_TOKEN = os.getenv("INFLUXDB_TOKEN", "my-super-secret-auth-token")
INFLUXDB_ORG = os.getenv("INFLUXDB_ORG", "obs")
INFLUXDB_BUCKET = os.getenv("INFLUXDB_BUCKET", "metrics")

client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
query_api = client.query_api()
write_api = client.write_api(write_options=SYNCHRONOUS)

# Global buffer for sliding window
window_buffer = []
TRAIN_MEAN = 0.0  # Replace with training mean
TRAIN_STD = 1.0   # Replace with training std

def initialize_buffer(window_size=10):
    global window_buffer
    flux_query = f'''
        from(bucket: "{INFLUXDB_BUCKET}")
            |> range(start: -10m)
            |> filter(fn: (r) => r["_measurement"] == "system.network.io")
            |> filter(fn: (r) => r["_field"] == "gauge")
            |> filter(fn: (r) => r["device"] == "eth0")
            |> filter(fn: (r) => r["direction"] == "transmit")
            |> limit(n: {window_size})
    '''
    try:
        tables = query_api.query(query=flux_query)
        values = [record.get_value() for table in tables for record in table.records]
        window_buffer.extend(values[-window_size:])
        logging.info(f"Initialized buffer with {len(window_buffer)} values")
    except Exception as e:
        logging.error(f"Failed to initialize buffer: {e}")

def detect(record, threshold_std=2.0):
    global window_buffer
    actual_value = record.get_value()

    # Add current value to buffer
    window_buffer.append(actual_value)
    if len(window_buffer) > WINDOW_SIZE:
        window_buffer.pop(0)

    # Check if buffer has enough data
    if len(window_buffer) < WINDOW_SIZE:
        logging.info("Not enough data in buffer for a full window, skipping detection")
        return False

    # Create and normalize input tensor
    input_tensor = torch.tensor(window_buffer, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)  # Shape: (1, window_size, 1)
    input_tensor = (input_tensor - TRAIN_MEAN) / (TRAIN_STD + 1e-6)

    # Online weight update
    target_tensor = torch.tensor([[actual_value]], dtype=torch.float32)  # Shape: (1, 1)
    model.train()  # Switch to training mode
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Use a small learning rate
    optimizer.zero_grad()
    loss = elbo_loss(model, input_tensor, target_tensor, num_samples=100, num_data_points=1)
    loss.backward()
    optimizer.step()
    model.eval()  # Switch back to evaluation mode

    # Get model prediction
    with torch.no_grad():
        mean_pred, epistemic_std, aleatoric_std = model.predict(input_tensor, num_samples=100)

    # Squeeze predictions
    mean_pred = mean_pred.squeeze()
    epistemic_std = epistemic_std.squeeze()
    aleatoric_std = aleatoric_std.squeeze()

    if epistemic_std < 1e-6:
        logging.warning("Epistemic standard deviation too small, skipping anomaly detection")
        return False

    print(f"Value: {actual_value:.2f}, Expected: {mean_pred:.2f}, "
          f"Epistemic Std: {epistemic_std:.2f}, Aleatoric Std: {aleatoric_std:.2f}")

    is_anomaly = abs(actual_value - mean_pred) > threshold_std * epistemic_std
    if is_anomaly:
        print("Anomaly detected!")
        logging.info(f"Anomaly detected! Value: {actual_value:.2f}, Expected: {mean_pred:.2f}, "
                     f"Epistemic Std: {epistemic_std:.2f}, Aleatoric Std: {aleatoric_std:.2f}")
    return is_anomaly.item(), epistemic_std.item(), aleatoric_std.item()

def process_data():
    flux_query = f'''
        from(bucket: "{INFLUXDB_BUCKET}")
            |> range(start: -1m)
            |> filter(fn: (r) => r["_measurement"] == "system.network.io")
            |> filter(fn: (r) => r["_field"] == "gauge")
            |> filter(fn: (r) => r["device"] == "eth0")
            |> filter(fn: (r) => r["direction"] == "transmit")
    '''
    try:
        tables = query_api.query(query=flux_query)
        if not tables:
            logging.warning("No data returned from InfluxDB query")
            return
        for table in tables:
            for record in table.records:
                is_anomaly, epistemic_uncertainty, aleatoric_uncertainty = detect(record)
                if is_anomaly:
                    anomaly_point = Point(record.get_measurement()) \
                        .tag("_tenant_id", record.values.get('_tenant_id')) \
                        .field("_anomalies", float(record.get_value())) \
                        .time(record.get_time())
                    write_api.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=anomaly_point)
                    logging.info(f"Anomaly detected and written for tenant {record.values.get('_tenant_id')} at {record.get_time()}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Anomaly detection service started")
    initialize_buffer(WINDOW_SIZE)
    while True:
        process_data()
        time.sleep(60)