#!/bin/bash

# Get host IP address
HOST_IP=$(hostname -I | awk '{print $2}')
export HOST_IP

# Start the stack
cd /home/ubuntu/6g-monitoring
/usr/bin/docker compose up -d
