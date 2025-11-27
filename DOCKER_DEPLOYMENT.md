# 🐳 Docker Deployment Guide

## Quick Start

### Windows Users
```powershell
# Run the interactive deployment script
.\docker-deploy.bat
```

### Linux/Mac Users
```bash
# Make script executable
chmod +x docker-deploy.sh

# Run the interactive deployment script
./docker-deploy.sh
```

---

## Manual Docker Commands

### 1. Build the Docker Image
```bash
docker build -t cmpas-anomaly-detection:latest .
```

### 2. Run with Docker Compose (Recommended)
```bash
# Start in foreground (see logs in real-time)
docker-compose up

# Start in background (detached mode)
docker-compose up -d

# View logs
docker-compose logs -f

# Stop application
docker-compose down
```

### 3. Run with Docker (Without Compose)
```bash
docker run -d \
  -p 5000:5000 \
  -v $(pwd):/app \
  --name cmpas-app \
  cmpas-anomaly-detection:latest
```

---

## Development Workflow

### Hot Reload for Code Changes
The `docker-compose.yml` is configured with volume mounts for hot-reload:
- Changes to `app.py`, `templates/`, `static/` are reflected immediately
- No need to rebuild image for code changes
- Flask debug mode is enabled by default

### Rebuild After Dependency Changes
```bash
# If you modify requirements.txt
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

---

## Cross-Platform Compatibility

### Windows
```powershell
# Use PowerShell or CMD
docker-compose up -d

# Or use the Windows deployment script
.\docker-deploy.bat
```

### Linux
```bash
# Standard docker-compose
docker-compose up -d

# Or use the shell script
chmod +x docker-deploy.sh
./docker-deploy.sh
```

### macOS
```bash
# Same as Linux
docker-compose up -d

# Or use the shell script
chmod +x docker-deploy.sh
./docker-deploy.sh
```

---

## Environment Variables

Configure via `docker-compose.yml` or pass directly:

```yaml
environment:
  - FLASK_ENV=development          # development or production
  - FLASK_DEBUG=1                   # Enable Flask debug mode
  - TF_CPP_MIN_LOG_LEVEL=2         # Suppress TensorFlow warnings
  - TF_ENABLE_ONEDNN_OPTS=0        # Disable oneDNN warnings
```

---

## Port Configuration

Default: `http://localhost:5000`

To change port, modify `docker-compose.yml`:
```yaml
ports:
  - "8080:5000"  # Map host port 8080 to container port 5000
```

Then access at `http://localhost:8080`

---

## Persistent Data

Model files and data are mounted as volumes:
- `*.keras` - VAE models
- `*.pkl` - Configuration and Kalman/ARIMA models
- `*.npy` - Training data

These persist across container restarts.

---

## Troubleshooting

### Container Won't Start
```bash
# Check logs
docker-compose logs cmpas-app

# Check if port 5000 is already in use
# Windows
netstat -ano | findstr :5000

# Linux/Mac
lsof -i :5000
```

### Model Files Not Found
```bash
# Ensure all model files exist in project directory
ls -la *.keras *.pkl *.npy

# If missing, volume mounts in docker-compose.yml will fail
```

### Permission Issues (Linux/Mac)
```bash
# Fix file permissions
chmod -R 755 static/ templates/
chmod 644 *.keras *.pkl *.npy
```

### Rebuild from Scratch
```bash
# Remove everything and start fresh
docker-compose down -v
docker rmi cmpas-anomaly-detection:latest
docker build --no-cache -t cmpas-anomaly-detection:latest .
docker-compose up -d
```

---

## Production Deployment

### 1. Use Production-Ready WSGI Server
Modify `Dockerfile` CMD:
```dockerfile
# Install gunicorn
RUN pip install gunicorn

# Replace CMD with
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "120", "app:app"]
```

### 2. Environment Configuration
Create `.env` file:
```env
FLASK_ENV=production
FLASK_DEBUG=0
TF_CPP_MIN_LOG_LEVEL=3
```

Reference in `docker-compose.yml`:
```yaml
env_file:
  - .env
```

### 3. Resource Limits
Add to `docker-compose.yml`:
```yaml
deploy:
  resources:
    limits:
      cpus: '2'
      memory: 4G
    reservations:
      cpus: '1'
      memory: 2G
```

---

## CI/CD Integration

### GitHub Actions Example
```yaml
name: Build and Deploy

on:
  push:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Build Docker image
        run: docker build -t cmpas-anomaly-detection:${{ github.sha }} .
      
      - name: Run tests
        run: docker run cmpas-anomaly-detection:${{ github.sha }} python -m pytest
      
      - name: Push to registry
        run: |
          docker tag cmpas-anomaly-detection:${{ github.sha }} registry.example.com/cmpas:latest
          docker push registry.example.com/cmpas:latest
```

---

## Multi-Device Setup

### 1. Clone Repository on New Device
```bash
git clone <repository-url>
cd CMPAS
```

### 2. Run Docker Setup
```bash
# Automatic (recommended)
./docker-deploy.sh  # or docker-deploy.bat on Windows

# Manual
docker-compose up -d
```

### 3. Access Dashboard
Open browser: `http://localhost:5000`

**No Python installation required!** Docker handles everything.

---

## Health Checks

Container includes automatic health monitoring:
```bash
# Check container health
docker inspect cmpas-anomaly-detection | grep Health -A 10

# Health check endpoint
curl http://localhost:5000/
```

---

## Backup & Migration

### Export Container State
```bash
# Create image from running container
docker commit cmpas-anomaly-detection cmpas-backup:$(date +%Y%m%d)

# Save to tar archive
docker save -o cmpas-backup.tar cmpas-backup:$(date +%Y%m%d)
```

### Restore on Another Machine
```bash
# Load image
docker load -i cmpas-backup.tar

# Run container
docker run -d -p 5000:5000 cmpas-backup:YYYYMMDD
```

---

## Support

For issues or questions:
1. Check logs: `docker-compose logs -f`
2. Verify model files are present
3. Ensure port 5000 is available
4. Check Docker Desktop is running (Windows/Mac)

**Container includes all dependencies - guaranteed to work identically across all platforms!**
