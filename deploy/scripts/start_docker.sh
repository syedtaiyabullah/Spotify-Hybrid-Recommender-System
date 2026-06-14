#!/bin/bash
exec > /home/ubuntu/start_docker.log 2>&1
set -e

ECR_REGISTRY="891377050051.dkr.ecr.ap-south-1.amazonaws.com"
COMPOSE_FILE="/home/ubuntu/app/deploy/docker-compose.deploy.yml"

echo "Logging in to ECR..."
aws ecr get-login-password --region ap-south-1 | \
  docker login --username AWS --password-stdin "$ECR_REGISTRY"

echo "Pulling latest images..."
ECR_REGISTRY="$ECR_REGISTRY" docker compose -f "$COMPOSE_FILE" pull

echo "Stopping existing containers..."
ECR_REGISTRY="$ECR_REGISTRY" docker compose -f "$COMPOSE_FILE" down --remove-orphans || true

echo "Starting services..."
ECR_REGISTRY="$ECR_REGISTRY" docker compose -f "$COMPOSE_FILE" up -d

echo "Services started successfully."
