// dashboard.js - modular frontend logic for Hybrid AI Sentinel
(function(){
  // Guard so multiple executions don't rebind
  if(window.__dashboardInitDone){ return; }
  window.__dashboardInitDone = false;
  // Predeclare global handlers (temporary placeholders) to avoid ReferenceError before init completes
  window.startSim = function(){ console.warn('Dashboard not initialized yet.'); };
  window.stopSim = function(){ console.warn('Dashboard not initialized yet.'); };
  window.resetSim = function(){ console.warn('Dashboard not initialized yet.'); };
  try {
  const API = 'http://localhost:5000/api';
  let eventSrc = null;
  const MAX_TIMELINE_POINTS = 300;
  const MAX_SENSOR_ROWS = 25;
  let sensorCols = [];
  let lastEnsemble = '-';
  let lastReasoningText = '';

  // Tabs
  const tabButtons = document.querySelectorAll('.tab');
  tabButtons.forEach(btn=>btn.addEventListener('click',()=>switchTab(btn)));
  function switchTab(btn){
    tabButtons.forEach(b=>{ b.classList.remove('active'); b.setAttribute('aria-selected','false'); });
    btn.classList.add('active'); btn.setAttribute('aria-selected','true');
    document.querySelectorAll('.tab-panel').forEach(p=>p.classList.remove('active'));
    document.getElementById('tab-'+btn.dataset.tab).classList.add('active');
  }

  // Chart defaults
  Chart.defaults.color = '#94a3b8';
  Chart.defaults.borderColor = '#2f435a';
  Chart.defaults.font.family = 'Inter, sans-serif';

  const mainChart = new Chart(document.getElementById('mainChart').getContext('2d'), {
    type: 'line',
    data: { labels: [], datasets: [
      { label: 'Anomaly Score', data: [], borderColor: '#0ea5e9', backgroundColor: 'rgba(14,165,233,0.15)', fill:true, tension:0.25, pointRadius:0 },
      { label: 'Threshold', data: [], borderColor: '#ef4444', borderDash:[5,4], pointRadius:0, borderWidth:1 }
    ]},
    options:{ responsive:true, maintainAspectRatio:false, animation:false, scales:{ x:{ grid:{color:'#1f3245'}}, y:{ grid:{color:'#1f3245'} } }, plugins:{legend:{labels:{color:'#e2e8f0'}}} }
  });

  const radarChart = new Chart(document.getElementById('radarChart').getContext('2d'), {
    type:'radar',
    data:{ labels:[], datasets:[{ label:'Impact', data:[], backgroundColor:'rgba(239,68,68,0.25)', borderColor:'#ef4444', pointRadius:2 }] },
    options:{ responsive:true, maintainAspectRatio:false, animation:false, scales:{ r:{ grid:{color:'#203042'}, angleLines:{color:'#203042'}, pointLabels:{color:'#94a3b8'} } } }
  });

  const reconChart = new Chart(document.getElementById('reconstructionChart').getContext('2d'), {
    type:'bar',
    data:{ labels:[], datasets:[
      { label:'Actual', data:[], backgroundColor:'#0ea5e9', borderWidth:0 },
      { label:'Expected', data:[], backgroundColor:'#10b981', borderWidth:0 }
    ]},
    options:{ responsive:true, maintainAspectRatio:false, animation:false, scales:{ x:{display:true}, y:{ grid:{color:'#1f3245'} } } }
  });

  const confChart = new Chart(document.getElementById('confidenceChart').getContext('2d'), {
    type:'doughnut',
    data:{ labels:['VAE','Kalman','ARIMA'], datasets:[{ data:[40,30,30], backgroundColor:['#0ea5e9','#10b981','#f59e0b'], borderWidth:0 }] },
    options:{ responsive:true, maintainAspectRatio:false, animation:false }
  });

  const contribChart = new Chart(document.getElementById('contribChart').getContext('2d'), {
    type:'bar',
    data:{ labels:[], datasets:[{ label:'Contribution', data:[], backgroundColor:'#6366f1' }] },
    options:{ responsive:true, maintainAspectRatio:false, animation:false, scales:{ x:{ grid:{display:false}}, y:{ grid:{color:'#1f3245'} } } }
  });

  const sensorHeaderRow = document.getElementById('sensorHeaderRow');
  const sensorTbody = document.getElementById('sensorTbody');

  function buildSensorHeader(cols){
    sensorHeaderRow.innerHTML = '<th>Cycle</th>' + cols.map(c=>`<th>${c.replace('sensor_','S')}</th>`).join('');
  }
  function appendSensorRow(cycle, sensors){
    if(!sensorCols.length){ sensorCols = Object.keys(sensors).filter(k=>k!=='cycle' && k!=='timestamp'); buildSensorHeader(sensorCols); }
    const row = document.createElement('tr');
    row.innerHTML = `<td>${cycle}</td>` + sensorCols.map(c=>`<td>${formatVal(sensors[c])}</td>`).join('');
    sensorTbody.appendChild(row);
    while(sensorTbody.children.length > MAX_SENSOR_ROWS){ sensorTbody.removeChild(sensorTbody.firstChild); }
  }
  function formatVal(v){ if(v===undefined||v===null) return '-'; return (+v).toFixed(3); }

  function initStream(){ if(eventSrc){ eventSrc.close(); } eventSrc = new EventSource(`${API}/stream`); eventSrc.onmessage = e => { const payload = JSON.parse(e.data); updateUI(payload); }; }

  function updateUI(p){
    document.getElementById('engineId').innerText = p.engine;
    document.getElementById('cycleCount').innerText = p.cycle;
    const windowSize = p.explainability ? (p.explainability.all_sensor_names?.length ? p.explainability.all_sensor_names.length : 30) : 30;
    document.getElementById('bufferStatus').innerText = `${p.buffer_fill}/${windowSize}`;
    if(p.sensors) appendSensorRow(p.cycle, p.sensors);
    if(p.anomaly) updateAnomaly(p.anomaly, p.cycle);
    if(p.explainability) updateExplain(p.explainability, p.anomaly ? p.anomaly.is_anomaly : false);
  }

  function updateAnomaly(a, cycle){
    const total = a.vae_score + a.kalman_score + a.arima_score + 1e-9;
    confChart.data.datasets[0].data = [ (a.vae_score/total*100)||0, (a.kalman_score/total*100)||0, (a.arima_score/total*100)||0 ];
    confChart.update();
    const alertBox = document.getElementById('anomalyAlert');
    if(a.is_anomaly){ 
      const reasoningHTML = lastReasoningText ? `<div style="font-size:11px;margin-top:6px;opacity:0.9;line-height:1.4;">${lastReasoningText}</div>` : '';
      alertBox.className='alert anom'; 
      alertBox.innerHTML='<i class="fa fa-exclamation-triangle"></i><div class="alert-text"><strong>ANOMALY DETECTED</strong><div>Confidence '+(a.confidence*100).toFixed(1)+'%</div>'+reasoningHTML+'</div>'; 
      document.getElementById('sysStatus').textContent='Anomaly'; 
      document.getElementById('sysStatus').className='status-dot err'; 
    }
    else { alertBox.className='alert normal'; alertBox.innerHTML='<i class="fa fa-check-circle"></i><div class="alert-text"><strong>SYSTEM NORMAL</strong></div>'; document.getElementById('sysStatus').textContent='Monitoring'; document.getElementById('sysStatus').className='status-dot ok'; lastReasoningText=''; }
    mainChart.data.labels.push('C'+cycle); mainChart.data.datasets[0].data.push(a.vae_score);
    if(mainChart.data.labels.length > MAX_TIMELINE_POINTS){ mainChart.data.labels.shift(); mainChart.data.datasets[0].data.shift(); mainChart.data.datasets[1].data.shift(); }
    mainChart.update('none');
    lastEnsemble = 'VAE:'+a.vae_score.toFixed(4)+' Kal:'+a.kalman_score.toFixed(3)+' ARIMA:'+a.arima_score.toFixed(3);
    document.getElementById('ensembleDetails').textContent = lastEnsemble;
  }

  function updateExplain(exData, isAnom){
    if(mainChart.data.datasets[1].data.length !== mainChart.data.datasets[0].data.length){ while(mainChart.data.datasets[1].data.length < mainChart.data.datasets[0].data.length){ mainChart.data.datasets[1].data.push(exData.threshold); } } else { mainChart.data.datasets[1].data[mainChart.data.datasets[1].data.length-1] = exData.threshold; }
    mainChart.update('none');
    const vaeDiv = document.getElementById('vaeDetails'); vaeDiv.textContent = exData.vae_analysis; document.getElementById('vaeAnalysis').style.display='block';
    radarChart.data.labels = exData.all_sensor_names.map(s=>s.replace('sensor_','S')); radarChart.data.datasets[0].data = exData.all_shap_values.length?exData.all_shap_values:exData.all_vae_errors; radarChart.update('none');
    
    // Build natural language reasoning for alert
    if(exData.top_features && exData.top_features.length>0){
      const topSensor = exData.top_features[0];
      const sensorName = topSensor.sensor.replace('sensor_','Sensor ');
      let reason = `${sensorName} shows ${topSensor.deviation_pct.toFixed(1)}% deviation. `;
      if(topSensor.vae_error > 0.01) reason += `High reconstruction error (${topSensor.vae_error.toFixed(4)}). `;
      if(Math.abs(topSensor.shap_value) > 0.001) reason += `SHAP indicates strong feature impact (${Math.abs(topSensor.shap_value).toFixed(4)}). `;
      if(topSensor.status==='CRITICAL') reason += '⚠️ Critical status.';
      else if(topSensor.status==='WARNING') reason += '⚠ Warning level.';
      lastReasoningText = reason;
    } else { lastReasoningText = 'Multiple sensors exceed thresholds.'; }
    
    const list = document.getElementById('evidenceList'); list.innerHTML=''; const contribLabels=[]; const contribValues=[]; const chartLabels=[]; const actuals=[]; const preds=[];
    exData.top_features.forEach(f=>{ 
      const div=document.createElement('div'); 
      const statClass=f.status==='CRITICAL'?'crit':f.status==='WARNING'?'warn':'minor'; 
      div.className='item '+statClass; 
      const humanReason = buildHumanReason(f);
      div.innerHTML=`<div><strong>${f.sensor.replace('sensor_','Sensor ')}</strong> <span class='meta'>${f.status} — Dev ${f.deviation_pct.toFixed(1)}%</span></div><div style='margin:4px 0;font-size:12px;line-height:1.4;'>${humanReason}</div><div class='meta'>Actual: ${f.actual.toFixed(3)} | Expected: ${f.reconstructed.toFixed(3)} | VAE err: ${f.vae_error.toFixed(4)} | SHAP: ${Math.abs(f.shap_value).toFixed(4)}</div>`; 
      list.appendChild(div); 
      contribLabels.push(f.sensor.replace('sensor_','S')); 
      contribValues.push(f.total_contrib); 
      if(chartLabels.length<3){ chartLabels.push(f.sensor.replace('sensor_','S')); actuals.push(f.actual); preds.push(f.reconstructed); } 
    });
    reconChart.data.labels = chartLabels; reconChart.data.datasets[0].data = actuals; reconChart.data.datasets[1].data = preds; reconChart.update('none');
    contribChart.data.labels = contribLabels; contribChart.data.datasets[0].data = contribValues; contribChart.update('none');
  }
  
  function buildHumanReason(f){
    let parts = [];
    if(f.vae_error > 0.008) parts.push(`VAE detected ${(f.vae_error*1000).toFixed(2)}‰ reconstruction error`);
    else if(f.vae_error > 0.003) parts.push(`Moderate VAE error (${(f.vae_error*1000).toFixed(2)}‰)`);
    if(Math.abs(f.shap_value) > 0.002) parts.push(`high SHAP attribution`);
    else if(Math.abs(f.shap_value) > 0.0005) parts.push(`moderate SHAP influence`);
    if(f.kalman_innov > 0.05) parts.push(`Kalman innovation ${f.kalman_innov.toFixed(3)}`);
    if(f.arima_res > 0.05) parts.push(`ARIMA residual ${f.arima_res.toFixed(3)}`);
    if(f.deviation_pct > 20) parts.push(`⚠️ ${f.deviation_pct.toFixed(0)}% deviation from expected`);
    else if(f.deviation_pct > 10) parts.push(`${f.deviation_pct.toFixed(0)}% off expected value`);
    return parts.length ? parts.join('; ') + '.' : 'Sensor reading deviates from trained pattern.';
  }

  // Expose handlers globally
  window.startSim = function(){ fetch(`${API}/start`,{method:'POST'}).then(()=>{ document.getElementById('sysStatus').textContent='Monitoring'; document.getElementById('sysStatus').className='status-dot ok'; initStream(); }); };
  window.stopSim = function(){ fetch(`${API}/stop`,{method:'POST'}).then(()=>{ document.getElementById('sysStatus').textContent='Stopped'; document.getElementById('sysStatus').className='status-dot warn'; if(eventSrc){ eventSrc.close(); eventSrc=null; } }); };
  window.resetSim = function(){ const id=document.getElementById('engineInput').value; fetch(`${API}/reset`,{method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({engine_id:parseInt(id)})}).then(()=>{ clearAll(); loadCompletedEngines(); }); };

  function clearAll(){ mainChart.data.labels=[]; mainChart.data.datasets[0].data=[]; mainChart.data.datasets[1].data=[]; mainChart.update(); reconChart.data.labels=[]; reconChart.data.datasets[0].data=[]; reconChart.data.datasets[1].data=[]; reconChart.update(); radarChart.data.labels=[]; radarChart.data.datasets[0].data=[]; radarChart.update(); contribChart.data.labels=[]; contribChart.data.datasets[0].data=[]; contribChart.update(); sensorTbody.innerHTML=''; document.getElementById('evidenceList').innerHTML=''; document.getElementById('vaeDetails').textContent='-'; document.getElementById('ensembleDetails').textContent='-'; }

  // ============================================================
  // COMPLETED ENGINES ARCHIVE
  // ============================================================
  function loadCompletedEngines() {
    fetch(`${API}/completed-engines`)
      .then(r => r.json())
      .then(data => {
        const list = document.getElementById('completedEnginesList');
        const engines = data.completed_engines || [];
        
        if (engines.length === 0) {
          list.innerHTML = '<div style="color:#8b949e;font-size:0.85rem;text-align:center;padding:1rem;">No completed engines yet</div>';
          return;
        }
        
        list.innerHTML = engines.map(e => `
          <div class="completed-engine-item" onclick="window.location.href='/engine/${e.engine_id}'">
            <div class="engine-info">
              <div class="engine-id-label">Engine ${e.engine_id}</div>
              <div class="engine-stats">${e.total_cycles} cycles • ${e.total_anomalies} anomalies</div>
            </div>
            <div class="engine-health ${e.status.toLowerCase()}">${e.health_score.toFixed(0)}</div>
          </div>
        `).join('');
      })
      .catch(err => console.error('Failed to load completed engines:', err));
  }

  // Poll for completed engines every 10 seconds
  setInterval(loadCompletedEngines, 10000);
  loadCompletedEngines(); // Initial load

  window.__dashboardInitDone = true;
  } catch(err){
    console.error('Dashboard initialization failed:', err);
    // Fallback handlers to avoid breaking buttons
    window.startSim = function(){ alert('Init error. See console for details.'); };
    window.stopSim = function(){};
    window.resetSim = function(){};
  }
})();
