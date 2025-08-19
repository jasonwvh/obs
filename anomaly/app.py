import os
import time
import logging

import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from bayes import BayesianAutoencoder

# Configuration
WINDOW_SIZE = 20
MODEL_PATH = 'model.pth'
INFLUXDB_URL = os.getenv("INFLUXDB_URL", "http://localhost:8086")
INFLUXDB_TOKEN = os.getenv("INFLUXDB_TOKEN", "my-super-secret-auth-token")
INFLUXDB_ORG = os.getenv("INFLUXDB_ORG", "obs")
INFLUXDB_BUCKET = os.getenv("INFLUXDB_BUCKET", "metrics")
ONLINE_LR = 1e-5  # Low learning rate for online updates
UPDATE_FREQ = 10  # Update every 10th normal sample
PERCENTILE = 95  # Percentile for anomaly threshold
ALPHA = 0.1

# Global variables
window_buffer = deque(maxlen=WINDOW_SIZE)
normalization_stats = {'min': 0.0, 'max': 1.0}
model = None
client = None
query_api = None
write_api = None
normal_sample_count = 0  # Track normal samples for periodic updates
data_point_count = 0  # Track total data points for norm updates
anomaly_threshold = 0.1  # Initial default, updated after initialization

def load_model():
    global model
    model = BayesianAutoencoder(input_dim=WINDOW_SIZE, hidden_dims=[64, 32], latent_dim=8)
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        norm_stats = checkpoint.get('normalization_stats', normalization_stats)
        if isinstance(norm_stats, tuple):
            norm_stats = {'min': norm_stats[0], 'max': norm_stats[1]}
        normalization_stats.update(norm_stats)
        for param in model.parameters():
            param.requires_grad_(True)
        logging.info("Model loaded")
        return True
    logging.error("Model file not found")
    return False


def initialize_influxdb():
    global client, query_api, write_api
    try:
        client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
        query_api = client.query_api()
        write_api = client.write_api(write_options=SYNCHRONOUS)
        logging.info("InfluxDB initialized")
        return True
    except Exception as e:
        logging.error(f"InfluxDB initialization failed: {e}")
        return False


def initialize_buffer_and_threshold():
    global window_buffer, normalization_stats, anomaly_threshold
    flux_query = f'''
        from(bucket: "{INFLUXDB_BUCKET}")
            |> range(start: -1h)
            |> filter(fn: (r) => r["_measurement"] == "system.network.io")
            |> filter(fn: (r) => r["_field"] == "gauge")
            |> filter(fn: (r) => r["device"] == "eth0")
            |> filter(fn: (r) => r["direction"] == "transmit")
            |> sort(columns: ["_time"])
            |> tail(n: {WINDOW_SIZE * 10})
    '''
    try:
        tables = query_api.query(query=flux_query)
        values = [float(record.get_value()) for table in tables for record in table.records
                  if record.get_value() is not None]

        if len(values) < WINDOW_SIZE:
            logging.warning(f"Insufficient data: got {len(values)}, using default threshold")
            window_buffer.extend(values[-WINDOW_SIZE:] if values else [0.0] * WINDOW_SIZE)
            return

        # Update buffer
        window_buffer.extend(values[-WINDOW_SIZE:] if values else [0.0] * WINDOW_SIZE)

        # Update normalization stats
        values_array = np.array(values)
        normalization_stats['min'] = float(values_array.min())
        normalization_stats['max'] = float(max(values_array.max(), values_array.min() + 1e-8))

        # Compute anomaly scores for threshold
        model.eval()
        scores = []
        windows = [values[i:i + WINDOW_SIZE] for i in range(max(0, len(values) - WINDOW_SIZE + 1))]
        for window in windows[:100]:
            if len(window) == WINDOW_SIZE:
                norm_window = normalize_data(window)
                input_tensor = torch.tensor(norm_window, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    mean_pred, epistemic_std = model.predict(input_tensor, num_samples=10)
                    recon_error = F.mse_loss(input_tensor, mean_pred, reduction='mean').item()
                    score = recon_error + epistemic_std.mean().item()
                    scores.append(score)

        if scores:
            anomaly_threshold = np.percentile(scores, PERCENTILE)
            logging.info(f"Anomaly threshold set to {anomaly_threshold:.4f} ({PERCENTILE}th percentile)")
        else:
            logging.warning("No scores computed, using default threshold 0.1")

        logging.info("Buffer and threshold initialized")
    except Exception as e:
        logging.error(f"Initialization failed: {e}")
        window_buffer.extend([0.0] * WINDOW_SIZE)

def normalize_data(values):
    min_val, max_val = normalization_stats['min'], normalization_stats['max']
    values = np.array(values)
    return np.clip((values - min_val) / (max_val - min_val), 0, 1)

def online_update(input_tensor):
    global model
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=ONLINE_LR)
    try:
        input_tensor = input_tensor.requires_grad_(True)
        optimizer.zero_grad()
        x_recon, mu, logvar = model(input_tensor, sample=True)
        loss_dict = model.compute_loss(input_tensor, x_recon, mu, logvar)
        loss = loss_dict['total_loss']

        if not torch.isnan(loss) and loss > 0:
            loss.requires_grad = True
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            logging.debug(f"Online update - Loss: {loss.item():.4f}")
        else:
            logging.warning(f"Invalid loss during online update: {loss.item()}")
    except Exception as e:
        logging.warning(f"Online update failed: {e}")
    finally:
        model.eval()


def detect_anomaly(new_value, num_samples=50):
    global window_buffer, normal_sample_count, normalization_stats, anomaly_threshold
    window_buffer.append(float(new_value))

    # Update running min/max with smoothing
    new_value = float(new_value)
    normalization_stats['min'] = (1 - ALPHA) * normalization_stats['min'] + ALPHA * min(new_value, normalization_stats['min'])
    normalization_stats['max'] = (1 - ALPHA) * normalization_stats['max'] + ALPHA * max(new_value, normalization_stats['max'])
    if normalization_stats['max'] - normalization_stats['min'] < 1e-8:
        normalization_stats['max'] = normalization_stats['min'] + 1e-8

    input_tensor = torch.tensor(normalize_data(list(window_buffer)), dtype=torch.float32).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        mean_pred, epistemic_std = model.predict(input_tensor, num_samples)
        recon_error = F.mse_loss(input_tensor, mean_pred, reduction='mean').item()
        anomaly_score = recon_error + epistemic_std.mean().item()
        is_anomaly = anomaly_score > anomaly_threshold

        if not is_anomaly:
            normal_sample_count += 1
            if normal_sample_count % UPDATE_FREQ == 0:
                online_update(input_tensor)

        return is_anomaly, anomaly_score


def process_data():
    global data_point_count
    flux_query = f'''
        from(bucket: "{INFLUXDB_BUCKET}")
            |> range(start: -2m)
            |> filter(fn: (r) => r["_measurement"] == "system.network.io")
            |> filter(fn: (r) => r["_field"] == "gauge")
            |> filter(fn: (r) => r["device"] == "eth0")
            |> filter(fn: (r) => r["direction"] == "transmit")
    '''
    try:
        tables = query_api.query(query=flux_query)
        for table in tables:
            for record in table.records:
                value = record.get_value()
                if value is None:
                    continue
                is_anomaly, score = detect_anomaly(value)
                if is_anomaly:
                    point = Point("anomalies") \
                        .tag("measurement", record.get_measurement()) \
                        .tag("device", "eth0") \
                        .tag("direction", "transmit") \
                        .field("value", float(value)) \
                        .field("anomaly_score", score) \
                        .time(record.get_time(), WritePrecision.NS)
                    write_api.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=point)
                    logging.info(f"Anomaly detected: Value={value:.4f}, Score={score:.4f}")
    except Exception as e:
        logging.error(f"Error processing data: {e}")

def main():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    if not (load_model() and initialize_influxdb()):
        return
    initialize_buffer_and_threshold()
    logging.info(f"Starting anomaly detection with threshold: {anomaly_threshold:.4f}")
    while True:
        process_data()
        time.sleep(1)

if __name__ == "__main__":
    main()