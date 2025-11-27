#!/bin/bash

# CMPAS Docker Build & Deployment Script
# Builds Docker image and provides quick start commands

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}╔════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  CMPAS Anomaly Detection System       ║${NC}"
echo -e "${GREEN}║  Docker Build & Deployment             ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════╝${NC}"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed. Please install Docker first.${NC}"
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo -e "${YELLOW}⚠️  docker-compose not found. Will use 'docker compose' instead.${NC}"
    DOCKER_COMPOSE="docker compose"
else
    DOCKER_COMPOSE="docker-compose"
fi

echo -e "${GREEN}✓ Docker is installed${NC}"
echo ""

# Function to display menu
show_menu() {
    echo -e "${YELLOW}Select an option:${NC}"
    echo "1) Build Docker image"
    echo "2) Start application (docker-compose up)"
    echo "3) Start in detached mode (background)"
    echo "4) Stop application"
    echo "5) View logs"
    echo "6) Rebuild and restart"
    echo "7) Clean up (remove containers and images)"
    echo "8) Exit"
    echo ""
}

# Build Docker image
build_image() {
    echo -e "${GREEN}📦 Building Docker image...${NC}"
    docker build -t cmpas-anomaly-detection:latest .
    echo -e "${GREEN}✓ Image built successfully!${NC}"
}

# Start with docker-compose
start_app() {
    echo -e "${GREEN}🚀 Starting application...${NC}"
    $DOCKER_COMPOSE up
}

# Start in detached mode
start_detached() {
    echo -e "${GREEN}🚀 Starting application in detached mode...${NC}"
    $DOCKER_COMPOSE up -d
    echo -e "${GREEN}✓ Application started!${NC}"
    echo -e "${GREEN}📊 Access dashboard at: http://localhost:5000${NC}"
    echo -e "${YELLOW}View logs with: $DOCKER_COMPOSE logs -f${NC}"
}

# Stop application
stop_app() {
    echo -e "${YELLOW}⏹  Stopping application...${NC}"
    $DOCKER_COMPOSE down
    echo -e "${GREEN}✓ Application stopped${NC}"
}

# View logs
view_logs() {
    echo -e "${GREEN}📋 Viewing logs (Ctrl+C to exit)...${NC}"
    $DOCKER_COMPOSE logs -f
}

# Rebuild and restart
rebuild_restart() {
    echo -e "${YELLOW}🔄 Rebuilding and restarting...${NC}"
    $DOCKER_COMPOSE down
    docker build -t cmpas-anomaly-detection:latest .
    $DOCKER_COMPOSE up -d
    echo -e "${GREEN}✓ Application rebuilt and restarted!${NC}"
    echo -e "${GREEN}📊 Access dashboard at: http://localhost:5000${NC}"
}

# Clean up
cleanup() {
    echo -e "${RED}🧹 Cleaning up...${NC}"
    $DOCKER_COMPOSE down -v
    docker rmi cmpas-anomaly-detection:latest 2>/dev/null || true
    echo -e "${GREEN}✓ Cleanup complete${NC}"
}

# Main loop
while true; do
    show_menu
    read -p "Enter your choice [1-8]: " choice
    
    case $choice in
        1)
            build_image
            ;;
        2)
            start_app
            ;;
        3)
            start_detached
            ;;
        4)
            stop_app
            ;;
        5)
            view_logs
            ;;
        6)
            rebuild_restart
            ;;
        7)
            cleanup
            ;;
        8)
            echo -e "${GREEN}👋 Goodbye!${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}❌ Invalid option. Please try again.${NC}"
            ;;
    esac
    
    echo ""
    read -p "Press Enter to continue..."
    clear
done
