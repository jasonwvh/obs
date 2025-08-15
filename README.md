# Observability
This project has 5 components:
1. Database
2. Anomaly
3. Backend
4. Frontend
5. OpenTelemetry

## OpenTelemetry
Client install OpenTelemetry collector (agent) and configure it to send data to our collector (gateway) through the endpoint `http://localhost:4319/ingest/hostmetrics-tenant`.

## Database
InfluxDB is used as the database to store metrics and anomalies.

## Backend
The backend provides endpoints `/api/v1/metrics/{tenant_id}` and `/api/v1/anomalies/{tenant_id}` to fetch the metrics and anomalies.

## Anomaly
The anomaly engine runs periodically to analyze the data in the database and detect anomalies.

## Frontend
The frontend is a Vue dashboard to visualize the metrics and anomalies.