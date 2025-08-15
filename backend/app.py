from datetime import timezone, datetime, timedelta
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
import os
import influxdb_client

INFLUXDB_URL = os.getenv("INFLUXDB_URL", "http://localhost:8086")
INFLUXDB_TOKEN = os.getenv("INFLUXDB_TOKEN", "my-super-secret-auth-token")
INFLUXDB_ORG = os.getenv("INFLUXDB_ORG", "obs")
INFLUXDB_BUCKET = os.getenv("INFLUXDB_BUCKET", "metrics")

client = influxdb_client.InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
query_api = client.query_api()

app = FastAPI(title="Observability Platform", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MetricDataPoint(BaseModel):
    timestamp: datetime
    value: Any

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/api/v1/metrics/{tenant_id}")
async def get_metrics(tenant_id: str):
    flux_query = f'''
        from(bucket: "{INFLUXDB_BUCKET}")
          |> range(start: -180s)
          |> filter(fn: (r) => r["_field"] == "gauge")
          |> filter(fn: (r) => r["_tenant_id"] == "{tenant_id}")
          |> yield(name: "mean")
    '''
    try:
        result = query_api.query(query=flux_query)
        metrics = {}
        for table in result:
            # Get measurement and tags
            metric_name = table.records[0].get_measurement()
            tags = table.records[0].values
            tag_key = tuple(
                sorted((k, v) for k, v in tags.items() if k not in ["_time", "_value", "_field", "_measurement"]))

            # Initialize measurement group if not exists
            if metric_name not in metrics:
                metrics[metric_name] = []

            # Create entry for this tag combination
            tag_entry = {
                "tags": dict(tag_key),
                "data_points": []
            }

            # Append data points
            for record in table.records:
                data_point = {
                    "timestamp": record.get_time().isoformat(),
                    "value": record.get_value()
                }
                tag_entry["data_points"].append(data_point)

            metrics[metric_name].append(tag_entry)

        return metrics
    except IndexError:
        return {}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/anomalies/{tenant_id}", response_model=Dict[str, List[MetricDataPoint]])
async def get_anomalies(tenant_id: str):
    flux_query = f'''
        from(bucket: "{INFLUXDB_BUCKET}")
          |> range(start: -180s)
          |> filter(fn: (r) => r["_field"] == "_anomalies")
          |> filter(fn: (r) => r["_tenant_id"] == "{tenant_id}")
          |> yield(name: "mean")
    '''
    try:
        result = query_api.query(query=flux_query)
        metrics = {}
        for table in result:
            for record in table.records:
                metric_name = record.get_measurement()
                data_point = {
                    "timestamp": record.get_time(),
                    "value": record.get_value()
                }
                metrics.setdefault(metric_name, []).append(data_point)
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)