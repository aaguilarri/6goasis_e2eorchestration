#
# Each instruction in this file generates a new layer that gets pushed to your local image cache
#
#
# Lines preceeded by # are regarded as comments and ignored
#
#
# The line below states we will base our new image on the Latest Official Ubuntu 
FROM ubuntu:22.04 
#
# Identify the maintainer of an image
LABEL maintainer="hola@cttc.es"
EXPOSE 9092
EXPOSE 9093
WORKDIR /home/
RUN apt update
RUN apt upgrade -y
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN apt-get update
RUN pip install kafka-python
RUN pip install numpy
RUN pip install pandas
RUN pip install matplotlib
RUN pip install scipy
RUN pip install kafka-python
RUN pip install pandas 
RUN pip install matplotlib
RUN pip install pygame
RUN pip install randomname
RUN pip install pymap3d
RUN pip install filterpy
ADD ./pfilt.py .
ADD ./carla_edge6.py .
ADD ./save_data2.py .
#ADD ./trajectory.py .
ADD ./delayer.py .
ADD ./plotter.py .
#ADD ./edge_commands.sh .
#ADD ./carla_edge6MWC.py .
ADD ./edge_experiment.py .
#RUN mkdir -p /home/edge_files
#RUN ls -la /home/edge_files/  
#ADD ./carandlight.json /home
#RUN cp /home/carandlight.json /home/edge_files/carandlight.json
#RUN ls -la /home/edge_files/  

 	

