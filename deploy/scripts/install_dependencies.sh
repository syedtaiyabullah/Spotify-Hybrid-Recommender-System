#!/bin/bash
export DEBIAN_FRONTEND=noninteractive

sudo apt-get update -y

# Docker
sudo apt-get install -y docker.io
sudo systemctl start docker
sudo systemctl enable docker

# Docker Compose plugin
sudo apt-get install -y docker-compose-plugin

# Utilities
sudo apt-get install -y unzip curl

# AWS CLI v2
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "/home/ubuntu/awscliv2.zip"
unzip -o /home/ubuntu/awscliv2.zip -d /home/ubuntu/
sudo /home/ubuntu/aws/install

sudo usermod -aG docker ubuntu

rm -rf /home/ubuntu/awscliv2.zip /home/ubuntu/aws
