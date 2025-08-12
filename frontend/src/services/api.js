import axios from 'axios';

const API_URL = '/api/v1';
const TENANT_ID = 'hostmetrics-tenant';

export const getMetrics = () => {
  return axios.get(`${API_URL}/metrics/${TENANT_ID}`);
};

export const getAnomalies = () => {
    return axios.get(`${API_URL}/anomalies/${TENANT_ID}`);
}