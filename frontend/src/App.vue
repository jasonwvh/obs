<template>
  <div id="app">
    <h1>Metrics Dashboard</h1>
    <div v-if="loading">Loading metrics...</div>
    <div v-if="error">{{ error }}</div>
    <div v-for="metric in metrics" :key="metric.name">
      <h2>{{ metric.name }}</h2>
      <div class="chart-container">
        <line-chart v-if="metric.data" :chart-data="metric.data"></line-chart>
      </div>
    </div>
  </div>
</template>

<script>
import { getMetrics, getAnomalies } from './services/api';
import LineChart from './components/LineChart.vue';

export default {
  name: 'App',
  components: {
    LineChart,
  },
  data() {
    return {
      loading: false,
      error: null,
      metrics: [],
    };
  },
  async created() {
    this.loading = true;
    try {
      const metricsResponse = await getMetrics();
      const metricsData = metricsResponse.data;

      const anomaliesResponse = await getAnomalies();
      const anomaliesData = anomaliesResponse.data;

      this.metrics = Object.entries(metricsData).map(([name, dataPoints]) => {
		  const metricAnomalies = anomaliesData[name] || [];
		  const anomalyTimestamps = new Set(metricAnomalies.map(a => a.timestamp));

		  return {
		    name,
		    data: {
		      labels: dataPoints.map(dp => new Date(dp.timestamp).toLocaleTimeString()),
		      datasets: [
		        {
		          label: name,
		          borderColor: '#f87979',
		          backgroundColor: 'rgba(248, 121, 121, 0.1)',
		          data: dataPoints.map(dp => dp.value),
		          pointBackgroundColor: dataPoints.map(dp =>
		            anomalyTimestamps.has(dp.timestamp) ? '#ff4444' : '#f87979'
		          ),
		          pointBorderColor: dataPoints.map(dp =>
		            anomalyTimestamps.has(dp.timestamp) ? '#ff4444' : '#f87979'
		          ),
		          pointRadius: dataPoints.map(dp =>
		            anomalyTimestamps.has(dp.timestamp) ? 5 : 1
		          ),
		          fill: false,
		          tension: 0.1
		        },
		      ],
		    },
		  };
		});
    } catch (err) {
      this.error = 'Failed to load metrics.';
      console.error(err);
    } finally {
      this.loading = false;
    }
  },
};
</script>

<style scoped>
.chart-container {
  position: relative;
  height: 400px;
  width: 100%;
}
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 60px;
}
</style>