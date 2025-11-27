@echo off
REM CMPAS Docker Build & Deployment Script for Windows
REM Builds Docker image and provides quick start commands

setlocal enabledelayedexpansion

echo.
echo ╔════════════════════════════════════════╗
echo ║  CMPAS Anomaly Detection System       ║
echo ║  Docker Build ^& Deployment             ║
echo ╚════════════════════════════════════════╝
echo.

REM Check if Docker is installed
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker is not installed. Please install Docker Desktop first.
    pause
    exit /b 1
)

echo ✓ Docker is installed
echo.

REM Check for docker-compose
docker-compose --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️  docker-compose not found. Using 'docker compose' instead.
    set DOCKER_COMPOSE=docker compose
) else (
    set DOCKER_COMPOSE=docker-compose
)

:menu
cls
echo.
echo ╔════════════════════════════════════════╗
echo ║         CMPAS Deployment Menu         ║
echo ╚════════════════════════════════════════╝
echo.
echo 1) Build Docker image
echo 2) Start application (foreground)
echo 3) Start in detached mode (background)
echo 4) Stop application
echo 5) View logs
echo 6) Rebuild and restart
echo 7) Clean up (remove containers and images)
echo 8) Exit
echo.

set /p choice="Enter your choice [1-8]: "

if "%choice%"=="1" goto build
if "%choice%"=="2" goto start
if "%choice%"=="3" goto start_detached
if "%choice%"=="4" goto stop
if "%choice%"=="5" goto logs
if "%choice%"=="6" goto rebuild
if "%choice%"=="7" goto cleanup
if "%choice%"=="8" goto exit

echo ❌ Invalid option. Please try again.
timeout /t 2 >nul
goto menu

:build
echo.
echo 📦 Building Docker image...
docker build -t cmpas-anomaly-detection:latest .
if %errorlevel% equ 0 (
    echo ✓ Image built successfully!
) else (
    echo ❌ Build failed!
)
pause
goto menu

:start
echo.
echo 🚀 Starting application...
%DOCKER_COMPOSE% up
pause
goto menu

:start_detached
echo.
echo 🚀 Starting application in detached mode...
%DOCKER_COMPOSE% up -d
if %errorlevel% equ 0 (
    echo ✓ Application started!
    echo 📊 Access dashboard at: http://localhost:5000
    echo View logs with: %DOCKER_COMPOSE% logs -f
)
pause
goto menu

:stop
echo.
echo ⏹  Stopping application...
%DOCKER_COMPOSE% down
echo ✓ Application stopped
pause
goto menu

:logs
echo.
echo 📋 Viewing logs (Ctrl+C to exit)...
%DOCKER_COMPOSE% logs -f
pause
goto menu

:rebuild
echo.
echo 🔄 Rebuilding and restarting...
%DOCKER_COMPOSE% down
docker build -t cmpas-anomaly-detection:latest .
%DOCKER_COMPOSE% up -d
echo ✓ Application rebuilt and restarted!
echo 📊 Access dashboard at: http://localhost:5000
pause
goto menu

:cleanup
echo.
echo 🧹 Cleaning up...
%DOCKER_COMPOSE% down -v
docker rmi cmpas-anomaly-detection:latest 2>nul
echo ✓ Cleanup complete
pause
goto menu

:exit
echo.
echo 👋 Goodbye!
timeout /t 2 >nul
exit /b 0
