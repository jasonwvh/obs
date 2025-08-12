import logging
import os
import random
import time
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

INFLUXDB_URL = os.getenv("INFLUXDB_URL", "http://localhost:8086")
INFLUXDB_TOKEN = os.getenv("INFLUXDB_TOKEN", "my-super-secret-auth-token")
INFLUXDB_ORG = os.getenv("INFLUXDB_ORG", "obs")
INFLUXDB_BUCKET = os.getenv("INFLUXDB_BUCKET", "metrics")

client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
query_api = client.query_api()
write_api = client.write_api(write_options=SYNCHRONOUS)

def detect():
    return random.random() < 0.01

def process_data():
    flux_query = f'''
        from(bucket: "{INFLUXDB_BUCKET}")
          |> range(start: -1m)
          |> filter(fn: (r) => r["_field"] == "gauge")
    '''
    try:
        tables = query_api.query(query=flux_query)
        for table in tables:
            for record in table.records:
                if detect():
                    tenant_id = record.values.get('_tenant_id')
                    if not tenant_id:
                        continue

                    anomaly_point = Point(record.get_measurement()) \
                        .tag("_tenant_id", tenant_id) \
                        .field("_anomalies", random.random()) \
                        .time(record.get_time())

                    write_api.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=anomaly_point)
                    logging.info(f"Anomaly detected and written for tenant {tenant_id} at {record.get_time()}")

    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Anomaly detection service started")
    while True:
        process_data()
        time.sleep(60)
