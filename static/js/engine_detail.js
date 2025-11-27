// ============================================================
// ENGINE DETAIL DASHBOARD - COMPREHENSIVE ANALYTICS
// ============================================================

let analysisData = null;
let charts = {};

// ============================================================
// FETCH ENGINE ANALYSIS DATA
// ============================================================
async function loadEngineAnalysis() {
  try {
    const response = await fetch(`/api/engine/${ENGINE_ID}/analysis`);
    if (!response.ok) throw new Error('Failed to load engine analysis');
    
    analysisData = await response.json();
    console.log('Engine Analysis Loaded:', analysisData);
    
    renderAllVisualizations();
  } catch (error) {
    console.error('Error loading engine analysis:', error);
    alert('Failed to load engine analysis. Please try again.');
  }
}

// Auto-refresh analysis data every 5 seconds for real-time updates
setInterval(() => {
  loadEngineAnalysis();
}, 5000);

// ============================================================
// RENDER ALL VISUALIZATIONS
// ============================================================
function renderAllVisualizations() {
  if (!analysisData) return;
  
  // Destroy ALL existing charts before re-rendering to prevent "Canvas already in use" errors
  if (charts.healthGauge) charts.healthGauge.destroy();
  if (charts.timeline) charts.timeline.destroy();
  if (charts.scoreDist) charts.scoreDist.destroy();
  if (charts.sensorHeatmap) charts.sensorHeatmap.destroy();
  
  renderHealthGauge();
  renderKeyMetrics();
  renderTopSensors();
  renderTimelineChart();
  renderScoreDistribution();
  renderSensorHeatmap();
  renderAnomalyEvents();
  
  // Show message if data is still being collected
  if (!analysisData.cycles || analysisData.cycles.length < 30) {
    // Only show notification once
    if (!document.getElementById('data-collection-notice')) {
      const msg = document.createElement('div');
      msg.id = 'data-collection-notice';
      msg.style.cssText = 'position:fixed;top:80px;right:20px;background:#f0883e;color:#0d1117;padding:1rem 1.5rem;border-radius:8px;font-weight:600;z-index:1000;box-shadow:0 4px 12px rgba(0,0,0,0.3);';
      msg.innerHTML = '<i class="fa fa-info-circle"></i> Data collection in progress (need 30+ cycles for full analysis)';
      document.body.appendChild(msg);
      setTimeout(() => msg.remove(), 5000);
    }
  }
}

// ============================================================
// HEALTH GAUGE (DOUGHNUT CHART)
// ============================================================
function renderHealthGauge() {
  const ctx = document.getElementById('healthGauge').getContext('2d');
  const health = analysisData.health_score || 0;
  
  const gaugeColor = health >= 80 ? '#3fb950' : health >= 50 ? '#f0883e' : '#f85149';
  const statusText = health >= 80 ? 'HEALTHY' : health >= 50 ? 'WARNING' : 'CRITICAL';
  const statusClass = health >= 80 ? 'healthy' : health >= 50 ? 'warning' : 'critical';
  
  document.getElementById('healthValue').textContent = health.toFixed(0);
  const statusElem = document.getElementById('healthStatus');
  statusElem.textContent = statusText;
  statusElem.className = `health-status ${statusClass}`;
  
  charts.healthGauge = new Chart(ctx, {
    type: 'doughnut',
    data: {
      datasets: [{
        data: [health, 100 - health],
        backgroundColor: [gaugeColor, 'rgba(255,255,255,0.05)'],
        borderWidth: 0
      }]
    },
    options: {
      cutout: '75%',
      responsive: true,
      maintainAspectRatio: true,
      plugins: { legend: { display: false }, tooltip: { enabled: false } }
    }
  });
}

// ============================================================
// KEY METRICS
// ============================================================
function renderKeyMetrics() {
  document.getElementById('totalCycles').textContent = analysisData.total_cycles || 0;
  document.getElementById('totalAnomalies').textContent = analysisData.total_anomalies || 0;
  document.getElementById('anomalyRate').textContent = (analysisData.anomaly_rate || 0).toFixed(2) + '%';
  document.getElementById('avgVae').textContent = (analysisData.avg_vae_score || 0).toFixed(4);
}

// ============================================================
// TOP CONTRIBUTING SENSORS
// ============================================================
function renderTopSensors() {
  const topSensors = analysisData.top_sensors || [];
  const listElem = document.getElementById('topSensorsList');
  
  if (topSensors.length === 0) {
    listElem.innerHTML = '<div style="color:#8b949e;font-size:0.85rem;padding:0.5rem;">Collecting sensor data... (need 30+ cycles)</div>';
    return;
  }
  
  listElem.innerHTML = topSensors.slice(0, 10).map(s => `
    <div class="sensor-item">
      <span class="sensor-name">${s.sensor}</span>
      <span class="sensor-contribution" title="Contribution Score: ${s.contribution_score?.toFixed(2) || 'N/A'}">${s.count}×</span>
    </div>
  `).join('');
}

// ============================================================
// ANOMALY TIMELINE CHART
// ============================================================
function renderTimelineChart() {
  const ctx = document.getElementById('timelineChart').getContext('2d');
  const cycles = analysisData.cycles || [];
  
  if (cycles.length === 0) {
    ctx.fillStyle = '#8b949e';
    ctx.font = '14px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Waiting for cycle data...', ctx.canvas.width / 2, ctx.canvas.height / 2);
    return;
  }
  
  // Show available data even if less than 30 cycles
  const labels = cycles.map(c => `C${c.cycle || 0}`);
  const vaeScores = cycles.map(c => c.vae_score || 0);
  const kalmanScores = cycles.map(c => c.kalman_score || 0);
  const arimaScores = cycles.map(c => c.arima_score || 0);
  const confidences = cycles.map(c => c.confidence || 0);
  
  charts.timeline = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'VAE Score',
          data: vaeScores,
          borderColor: '#58a6ff',
          backgroundColor: 'rgba(88,166,255,0.1)',
          borderWidth: 2,
          pointRadius: 3,
          tension: 0.3
        },
        {
          label: 'Kalman Score',
          data: kalmanScores,
          borderColor: '#3fb950',
          backgroundColor: 'rgba(63,185,80,0.1)',
          borderWidth: 2,
          pointRadius: 3,
          tension: 0.3
        },
        {
          label: 'ARIMA Score',
          data: arimaScores,
          borderColor: '#f0883e',
          backgroundColor: 'rgba(240,136,62,0.1)',
          borderWidth: 2,
          pointRadius: 3,
          tension: 0.3
        },
        {
          label: 'Confidence',
          data: confidences,
          borderColor: '#f85149',
          backgroundColor: 'rgba(248,81,73,0.1)',
          borderWidth: 2,
          pointRadius: 3,
          tension: 0.3,
          yAxisID: 'y1'
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      scales: {
        x: { 
          ticks: { color: '#8b949e', maxTicksLimit: 15 },
          grid: { color: '#30363d' }
        },
        y: {
          type: 'linear',
          position: 'left',
          title: { display: true, text: 'Model Scores', color: '#c9d1d9' },
          ticks: { color: '#8b949e' },
          grid: { color: '#30363d' }
        },
        y1: {
          type: 'linear',
          position: 'right',
          title: { display: true, text: 'Confidence', color: '#c9d1d9' },
          ticks: { color: '#8b949e' },
          grid: { display: false }
        }
      },
      plugins: {
        legend: { labels: { color: '#c9d1d9', usePointStyle: true } },
        tooltip: { 
          backgroundColor: '#161b22',
          titleColor: '#c9d1d9',
          bodyColor: '#c9d1d9',
          borderColor: '#30363d',
          borderWidth: 1
        }
      }
    }
  });
}

// ============================================================
// SCORE DISTRIBUTION (BAR CHART)
// ============================================================
function renderScoreDistribution() {
  const ctx = document.getElementById('scoreDistChart').getContext('2d');
  const cycles = analysisData.cycles || [];
  
  if (cycles.length === 0) {
    ctx.fillStyle = '#8b949e';
    ctx.font = '14px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Waiting for cycle data...', ctx.canvas.width / 2, ctx.canvas.height / 2);
    return;
  }
  
  // Show distribution even with limited data
  const vaeScores = cycles.map(c => c.vae_score || 0);
  const avgVae = vaeScores.reduce((a, b) => a + b, 0) / vaeScores.length;
  const maxVae = Math.max(...vaeScores);
  const minVae = Math.min(...vaeScores);
  
  charts.scoreDist = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['Min VAE', 'Avg VAE', 'Max VAE'],
      datasets: [{
        label: 'VAE Score Distribution',
        data: [minVae, avgVae, maxVae],
        backgroundColor: ['#3fb950', '#58a6ff', '#f85149'],
        borderWidth: 0
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { ticks: { color: '#8b949e' }, grid: { display: false } },
        y: { ticks: { color: '#8b949e' }, grid: { color: '#30363d' } }
      },
      plugins: {
        legend: { display: false },
        tooltip: { 
          backgroundColor: '#161b22',
          titleColor: '#c9d1d9',
          bodyColor: '#c9d1d9',
          borderColor: '#30363d',
          borderWidth: 1
        }
      }
    }
  });
}

// ============================================================
// SENSOR HEATMAP (HORIZONTAL BAR CHART)
// ============================================================
function renderSensorHeatmap() {
  const ctx = document.getElementById('sensorHeatmap').getContext('2d');
  const topSensors = analysisData.top_sensors || [];
  
  if (topSensors.length === 0) {
    ctx.fillStyle = '#8b949e';
    ctx.font = '14px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Collecting sensor data... (need 30+ cycles)', ctx.canvas.width / 2, ctx.canvas.height / 2);
    return;
  }
  
  // Show available sensors even if less than 10
  const displaySensors = topSensors.slice(0, 10);
  const labels = displaySensors.map(s => s.sensor);
  const counts = displaySensors.map(s => s.count);
  
  charts.sensorHeatmap = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Anomaly Contribution Count',
        data: counts,
        backgroundColor: '#58a6ff',
        borderWidth: 0
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { 
          ticks: { color: '#8b949e' },
          grid: { color: '#30363d' },
          title: { display: true, text: 'Contribution Frequency', color: '#c9d1d9' }
        },
        y: { ticks: { color: '#8b949e' }, grid: { display: false } }
      },
      plugins: {
        legend: { display: false },
          tooltips: { 
          backgroundColor: '#161b22',
          titleColor: '#c9d1d9',
          bodyColor: '#c9d1d9',
          borderColor: '#30363d',
          borderWidth: 1,
          callbacks: {
            afterLabel: (ctx) => {
              const sensor = displaySensors[ctx.dataIndex];
              const lines = [];
              if (sensor.avg_error !== undefined) lines.push(`Avg Error: ${sensor.avg_error.toFixed(4)}`);
              if (sensor.avg_shap !== undefined) lines.push(`Avg SHAP: ${sensor.avg_shap.toFixed(4)}`);
              return lines.length ? lines : [];
            }
          }
        }
      }
    }
  });
}// ============================================================
// ANOMALY EVENTS LIST (EXPANDABLE CARDS)
// ============================================================
function renderAnomalyEvents() {
  const anomalies = analysisData.anomalies || [];
  const listElem = document.getElementById('anomalyEventsList');
  
  if (anomalies.length === 0) {
    listElem.innerHTML = '<div style="color:#8b949e;text-align:center;padding:2rem;">No anomalies detected yet</div>';
    return;
  }
  
  listElem.innerHTML = anomalies.reverse().map((anom, idx) => {
    const confidence = anom.confidence || 0;
    const confidenceBadge = confidence >= 0.8 ? 'high' : confidence >= 0.5 ? 'medium' : 'low';
    const confidenceLabel = confidence >= 0.8 ? 'High' : confidence >= 0.5 ? 'Medium' : 'Low';
    
    const explanation = anom.explanation || {};
    const reasoning = explanation.reasoning || 'No detailed explanation available.';
    const topFeatures = explanation.top_features || [];
    
    const sensorHtml = Object.entries(anom.sensors || {}).slice(0, 12).map(([sensor, value]) => `
      <div class="sensor-tag">
        <div class="sensor-label">${sensor}</div>
        <div class="sensor-value">${typeof value === 'number' ? value.toFixed(3) : value}</div>
      </div>
    `).join('');
    
    return `
      <div class="event-card" id="event-${idx}" onclick="toggleEvent(${idx})">
        <div class="event-header">
          <span class="event-cycle">Cycle ${anom.cycle || 0}</span>
          <div class="event-confidence">
            <span class="badge ${confidenceBadge}">${confidenceLabel}</span>
            <span class="event-score">VAE: ${(anom.vae_score || 0).toFixed(4)}</span>
          </div>
        </div>
        <div class="event-details">
          <div class="event-explanation">${reasoning}</div>
          <div class="sensor-grid">${sensorHtml}</div>
        </div>
      </div>
    `;
  }).join('');
}

function toggleEvent(idx) {
  const card = document.getElementById(`event-${idx}`);
  card.classList.toggle('expanded');
}

// ============================================================
// EXPORT REPORT (CSV)
// ============================================================
function exportReport() {
  if (!analysisData) {
    alert('No data available to export');
    return;
  }
  
  // Generate CSV content
  let csv = 'ENGINE ANALYSIS REPORT\n';
  csv += `Engine ID: ${ENGINE_ID}\n`;
  csv += `Total Cycles: ${analysisData.total_cycles}\n`;
  csv += `Total Anomalies: ${analysisData.total_anomalies}\n`;
  csv += `Anomaly Rate: ${analysisData.anomaly_rate?.toFixed(2)}%\n`;
  csv += `Health Score: ${analysisData.health_score?.toFixed(2)}\n`;
  csv += `Avg VAE Score: ${analysisData.avg_vae_score?.toFixed(4)}\n\n`;
  
  csv += 'ANOMALY EVENTS\n';
  csv += 'Cycle,VAE Score,Confidence,Timestamp,Reasoning\n';
  (analysisData.anomalies || []).forEach(a => {
    const reasoning = (a.explanation?.reasoning || '').replace(/"/g, '""');
    csv += `${a.cycle},"${a.vae_score?.toFixed(4)}","${a.confidence?.toFixed(2)}","${a.timestamp || ''}","${reasoning}"\n`;
  });
  
  csv += '\nTOP CONTRIBUTING SENSORS\n';
  csv += 'Sensor,Count,Avg Error,Avg SHAP\n';
  (analysisData.top_sensors || []).forEach(s => {
    csv += `${s.sensor},${s.count},${s.avg_error?.toFixed(4)},${s.avg_shap?.toFixed(4)}\n`;
  });
  
  // Download CSV
  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `engine_${ENGINE_ID}_report.csv`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// ============================================================
// INITIALIZE ON PAGE LOAD
// ============================================================
document.addEventListener('DOMContentLoaded', () => {
  loadEngineAnalysis();
});
