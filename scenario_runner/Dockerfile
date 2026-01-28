# ===============================
# Dockerfile for CARLA 0.9.14 + ScenarioRunner + Python 3.8
# ===============================

FROM carlasim/carla:0.9.15

# Switch to root for package installation
USER root
# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CARLA_HOME=/home/carla
ENV SCENARIO_RUNNER_HOME=/home/carla/ScenarioRunner
# Add CARLA PythonAPI egg to Python path
#RUN ls $CARLA_HOME/PythonAPI/carla/dist/
#ENV PYTHONPATH="${CARLA_HOME}/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg"
ENV PYTHONPATH="${CARLA_HOME}/PythonAPI/carla:${CARLA_HOME}/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg:${SCENARIO_RUNNER_HOME}"





#ENV PYTHONPATH="${CARLA_HOME}/PythonAPI/carla:${SCENARIO_RUNNER_HOME}"


WORKDIR $CARLA_HOME

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    iputils-ping \
    libtiff5 \
    libjpeg8 && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.7 python3.7-distutils python3.7-dev python3-pip git wget unzip && \
    rm -rf /var/lib/apt/lists/*

# Set Python 3.8 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1 && \
    update-alternatives --set python3 /usr/bin/python3.7 && \
    python3 -m pip install --upgrade pip setuptools wheel


# Clone ScenarioRunner 0.9.14
RUN git clone -b v0.9.15 https://github.com/carla-simulator/scenario_runner.git $SCENARIO_RUNNER_HOME


# after cloning ScenarioRunner
RUN pip3 install --no-cache-dir \
    py-trees==0.8.3 \
    networkx==2.2 \
    Shapely==1.7.1 \
    psutil \
    xmlschema==1.0.18 \
    ephem \
    tabulate \
    opencv-python==4.2.0.32 \
    numpy \
    matplotlib \
    six \
    simple-watchdog-timer \
    antlr4-python3-runtime
    #pip3 install --no-cache-dir -r $SCENARIO_RUNNER_HOME/requirements.txt

# Install Python requirements
#RUN pip3 install --no-cache-dir -r $SCENARIO_RUNNER_HOME/requirements.txt

# Optional: install Vulkan/Xvfb for offscreen rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    mesa-vulkan-drivers vulkan-utils xvfb && \
    rm -rf /var/lib/apt/lists/*

# Switch to carla for running
USER carla
# Default command
#CMD ["bash", "-c", "echo 'CARLA ScenarioRunner with Python 3.8 ready. To run: python3 $SCENARIO_RUNNER_HOME/scenario_runner.py --help'"]

