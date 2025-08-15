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
  methods: {
    // Generate concise, dynamic labels
    generateLabel(measurement, tags) {
      const priorityTags = ['device', 'direction', 'core', 'cpu', 'state']; // Prioritize these tags
      const tagParts = priorityTags
        .filter(tag => tag in tags)
        .map(tag => tags[tag])
        .filter(value => value); // Exclude empty values
      const suffix = tagParts.length > 0 ? tagParts.join('/') : 'default';
      return `${suffix}`;
    },
  },
  async created() {
    this.loading = true;
    try {
      const metricsResponse = await getMetrics('some_tenant_id');
      const metricsData = metricsResponse.data;
      const anomaliesResponse = await getAnomalies('some_tenant_id');
      const anomaliesData = anomaliesResponse.data;

      // Define colors for time series
      const colors = [
        '#f87979', // Red
        '#36a2eb', // Blue
        '#ffce56', // Yellow
        '#4bc0c0', // Cyan
        '#9966ff', // Purple
      ];

      // Process each measurement
      this.metrics = Object.entries(metricsData).map(([name, timeSeries]) => {
        // Group anomalies by measurement
        var metricAnomalies = [];
        if (anomaliesData.length > 0) {
             metricAnomalies = anomaliesData.filter(a => a.measurement === name);
		}

        // Create datasets for time series and anomalies
        const datasets = [
          ...timeSeries.map((series, index) => {
            return {
              label: this.generateLabel(name, series.tags),
              borderColor: colors[index % colors.length],
              backgroundColor: colors[index % colors.length].replace('#', 'rgba(') + ', 0.1)',
              data: series.data_points.map(dp => dp.value),
              fill: false,
              tension: 0.1,
              pointRadius: 1,
            };
          })
        ];

        // Collect unique timestamps for labels
        const allTimestamps = new Set();
        timeSeries.forEach(series => {
          series.data_points.forEach(dp => {
            allTimestamps.add(new Date(dp.timestamp).toLocaleTimeString());
          });
        });
        metricAnomalies.forEach(anomaly => {
          allTimestamps.add(new Date(anomaly.timestamp).toLocaleTimeString());
        });

        return {
          name,
          data: {
            labels: [...allTimestamps].sort(),
            datasets,
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