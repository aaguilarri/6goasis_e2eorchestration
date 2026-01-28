FROM ubuntu:22.04

# Identify the maintainer of an image
LABEL maintainer="hola@cttc.es"

# Set working directory
WORKDIR /home

# Set environment variables early
ENV DEBIAN_FRONTEND=noninteractive
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Update package lists and install dependencies
RUN cat /etc/resolv.conf
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y \
        python3 \
        python3-pip \
        python3-venv \
        python3-dev \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        libgstreamer1.0-0 \
        libgstreamer-plugins-base1.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python3 -m venv /home/myvenv
ENV PATH="/home/myvenv/bin:$PATH"

# Install Python packages in the virtual environment
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
        carla==0.9.14 \
        numpy \
        matplotlib \
        opencv-python \
        pillow \
        ultralytics \
        pygame \
        lapx \
        torch \
        torchvision \
        kafka-python

# Reset DEBIAN_FRONTEND
ENV DEBIAN_FRONTEND=dialog

# Copy application files
COPY ./generate_traffic.py .
COPY ./red_light_camera.py .
COPY ./connected_vehicle.py .
COPY ./yolov8n.pt .
COPY ./new_world.py .
COPY ./set_red_light.py .
COPY ./carla_agent_ns.py .
