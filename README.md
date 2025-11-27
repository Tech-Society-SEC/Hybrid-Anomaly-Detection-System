# Hybrid AI Sentinel Dashboard

A clean, real-time explainable anomaly detection dashboard powered by Flask + TensorFlow (VAE), Kalman, and ARIMA ensemble.

## 🚀 Quick Start

### Option 1: Docker (Recommended - Works on Any OS)
```bash
# Windows
docker-deploy.bat

# Linux/Mac
chmod +x docker-deploy.sh
./docker-deploy.sh
```

Access at: **http://localhost:5000**

### Option 2: Local Python Setup
```bash
pip install -r requirements.txt
python app.py
```

Visit: **http://127.0.0.1:5000**

---

## 🐳 Docker Deployment

**Complete cross-platform containerization:**
- ✅ Works identically on Windows, Linux, macOS
- ✅ No Python/TensorFlow installation required
- ✅ Hot-reload for development
- ✅ Production-ready configuration

See **[DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md)** for detailed instructions.

### Quick Commands
```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down

# Rebuild after changes
docker-compose up -d --build
```

---

## Features
- Live sensor feed (rolling window)
- Real-time anomaly timeline with threshold overlay
- Ensemble confidence doughnut (VAE / Kalman / ARIMA)
- Reconstruction view (actual vs expected for top sensors)
- **Per-Engine Detailed Analysis Dashboard**
  - Health score gauge with degradation tracking
  - Anomaly timeline with multi-model scores
  - Sensor contribution heatmap
  - Expandable event cards with root cause analysis
  - CSV export functionality
- **Completed Engines Archive**
  - Automatic tracking of finished engine cycles
  - Clickable archive in dashboard sidebar
  - Preserves analysis across navigation
- Explainability panel with:
  - Overview (VAE analysis + ensemble raw scores)
  - Fingerprint (Radar impact + bar contributions)
  - Feature list (ranked sensors with multi-model reasoning)
- Server-Sent Events stream (`/api/stream`) for low-latency updates
- Robust model loading with fallbacks if config artifacts missing

---

## 📁 Project Structure
```
CMPAS/
├── app.py                      # Flask backend with ML models
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker image definition
├── docker-compose.yml          # Multi-container orchestration
├── docker-deploy.sh/.bat       # Interactive deployment scripts
├── DOCKER_DEPLOYMENT.md        # Complete Docker guide
├── templates/
│   ├── dashboard.html          # Real-time monitoring UI
│   └── engine_detail.html      # Per-engine analytics
├── static/
│   ├── css/
│   │   ├── dashboard.css       # Main dashboard styling
│   │   └── engine_detail.css   # Detail page styling
│   └── js/
│       ├── dashboard.js        # SSE client + Chart.js
│       └── engine_detail.js    # Analytics visualization
├── *.keras                     # VAE encoder/decoder models
├── *.pkl                       # Model configs and data
└── *.npy                       # Training/test datasets
```

---

## 🔧 Development Workflow

### Docker Development (Hot Reload Enabled)
1. Start container: `docker-compose up -d`
2. Edit code in `app.py`, `templates/`, `static/`
3. Changes reflect immediately (no rebuild needed)
4. View logs: `docker-compose logs -f`

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run with auto-reload
python app.py
```

Press **Start** to begin simulation. **Reset** selects engine ID and clears buffers. **Stop** halts streaming loop.

---

## 🌐 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main dashboard |
| `/engine/<id>` | GET | Detailed engine analysis page |
| `/api/stream` | GET | SSE real-time data stream |
| `/api/engine/<id>/analysis` | GET | Engine analytics (JSON) |
| `/api/completed-engines` | GET | List of finished engines |
| `/api/start` | POST | Start simulation |
| `/api/stop` | POST | Stop simulation |
| `/api/reset` | POST | Reset to specific engine |
| `/api/explainability` | GET | Current explainability data |

---

## 🎨 Customization

### Thresholds
Adjust in `app.py`:
```python
vae_threshold = state.vae_config.get('anomaly_threshold', 0.005)
kalman_threshold = state.kalman_config.get('threshold', 12.0)
arima_threshold = state.arima_config.get('threshold', 0.065)
```

### Colors & Styling
Edit CSS variables in `static/css/dashboard.css`:
```css
:root {
  --accent: #0ea5e9;
  --danger: #ef4444;
  --ok: #10b981;
}
```

### Adding Visualizations
1. Add `<canvas>` in HTML
2. Create Chart.js instance in JS
3. Update in `updateExplain()` or `updateAnomaly()`

---

## 🐛 Troubleshooting

### Docker Issues
```bash
# Check logs
docker-compose logs cmpas-app

# Rebuild from scratch
docker-compose down -v
docker-compose up -d --build

# Verify port availability
netstat -ano | findstr :5000  # Windows
lsof -i :5000                  # Linux/Mac
```

### Local Python Issues
- **Missing window_size**: Ensure `vae_config.pkl` exists or encoder shape is inferrable
- **SHAP errors**: Provide `shap_background.pkl` or `normal_windows.npy`
- **Empty simulation**: Check `test_scaled.pkl` has `unit`, `cycle`, and sensor columns
- **TensorFlow warnings**: Set `TF_CPP_MIN_LOG_LEVEL=2` environment variable

---

## 📊 Features Highlights

### Real-Time Monitoring
- **SSE streaming** (no polling)
- **30-second rolling window** for VAE
- **Live anomaly alerts** with confidence scores
- **Multi-model ensemble** voting

### Explainability
- **SHAP values** for VAE attributions
- **Kalman innovation** breakdown
- **ARIMA residual** analysis
- **Human-readable reasoning** for each anomaly

### Analytics Dashboard
- **Health score degradation** tracking
- **Sensor contribution heatmap** (top 10)
- **Anomaly event log** with expand-for-details
- **CSV export** for reporting

---

## 🔒 Production Deployment

### Use Gunicorn (Production WSGI Server)
Modify `Dockerfile`:
```dockerfile
RUN pip install gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "app:app"]
```

### Environment Configuration
Create `.env`:
```env
FLASK_ENV=production
FLASK_DEBUG=0
TF_CPP_MIN_LOG_LEVEL=3
```

### Resource Limits
Add to `docker-compose.yml`:
```yaml
deploy:
  resources:
    limits:
      cpus: '2'
      memory: 4G
```

---

## 📦 Dependencies

Core:
- Flask 3.0+
- TensorFlow 2.15+
- NumPy, Pandas
- SHAP (explainability)
- PyKalman (Kalman Filter)
- statsmodels (ARIMA)

UI:
- Chart.js 4.4.4
- Font Awesome 6.5.0

---

## 📝 License

Internal / Demo use only. Add licensing as needed.

---

## 🤝 Contributing

1. Fork repository
2. Create feature branch
3. Make changes (test with Docker!)
4. Submit pull request

**Docker ensures your changes work everywhere!**
