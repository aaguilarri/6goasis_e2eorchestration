#
# Each instruction in this file generates a new layer that gets pushed to your local image cache
#
#
# Lines preceeded by # are regarded as comments and ignored
#
#
# The line below states we will base our new image on the Latest Official Ubuntu 
# Use Ubuntu as the base image
FROM ubuntu:22.04

# Set non-interactive mode for apt
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR home
# Update and install dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    wget \
    build-essential \
    libssl-dev \
    libffi-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncurses5-dev \
    libgdbm-dev \
    libnss3-dev \
    liblzma-dev \
    zlib1g-dev \
    ca-certificates \
    libgl1 \ 
    libglib2.0-0 \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y python3.10 python3.10-venv python3.10-dev python3-pip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Verify version
RUN python --version

# Install pip packages
RUN pip install --upgrade pip \
    && pip install carla==0.9.15 ultralytics kafka-python pygame lap opencv-python numpy pillow

ADD ./generate_traffic.py .
ADD ./red_light_camera.py .
ADD ./connected_vehicle.py .
ADD ./yolov8n.pt .
ADD ./new_world.py .
ADD ./set_red_light.py .
ADD ./carla_agent.py .
ADD ./carla_agentMWC.py .
ADD ./carla_agentSCN.py .


RUN apt-get update && apt-get install -y iputils-ping