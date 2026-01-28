#!/bin/bash
# Run container
IDX=$1
IP="10.1.24.50"  #CARLA server IP
GPUS="1"        #Available GPUs
GPUCV="0"       #GPU used by agent
GPURL="0"       #GPU used by red light
TICK="yes"      #is the ticking script
NO_TICK="no"    #for non-ticking scripts
PTH="/home/ubuntu/carlaClientVenv/bin/python "
PTHH="/home/ubuntu/6goasis/client/"

ISKAFKA="kafka" #kafka no_kafka
KAFKAIP="127.0.0.1" #"${KAFKA_HOST:-kafka-0.kafka.kafka.svc.cluster.local}" #"172.10.0.1"
KAFKAPORT="9092" #${KAFKA_PORT:-9092}" #"9092"
KAFKAFILE="kafka_in.json"
#PTH=/home/aaguilar/carlaclient/
IMG="6goasis-client"    #Docker image
CNT0="carla_server"     #server container name  
CNT="carla_client"      #client container name
MAP="Town02"            #Map to be used
LAT="41.274927054998706"    #Reference latitude to scale coordinates
LON="1.9865927764756361"    #Reference longitude
## Sakin machine
#SOCKET="/tmp/.X11-unix/X1"          #Configure X11 socket
#AUTHH="/run/user/1000/gdm/Xauthority" #"/home/aaguilar/.Xauthority"  #Xautohryt file in client
#AUTHC="/root/.Xauthority" 
##Docker iesc-gpu
#SOCKET="/tmp/.X11-unix/X0"          #Configure X11 socket
#AUTHH="/home/aaguilar/.Xauthority"  #Xautohryt file in client
#AUTHC="/root/.Xauthority"           #Xautority file in root
# Marco machine
SOCKET="/tmp/.X11-unix"          #Configure X11 socket
AUTHH="/run/user/1000/gdm/Xauthority" #"/home/aaguilar/.Xauthority"  #Xautohryt file in client
AUTHC="/root/.Xauthority" 
#base coords
CMX="0.0" 
CMY="0.0" 
CMZ="10.0"
CMR="0.0"
CMP="-20.00"
CMW="0.0"
NCARS="30"     #number of cars when traffic is invoked
NNPCS="30"      #number of pedestrians
SID="000"       #Station ID (changes with agent type)
#Town00
#expectator is  Transform(Location(x=-46.258762, y=143.985580, z=3.581266), Rotation(pitch=-2.999635, yaw=-51.196709, roll=0.000002))
if [ "$MAP" == "Town00" ]
then
CMX="-46.258762" 
CMY="143.985580" 
CMZ="3.581266"
CMR="0.000022"
CMP="-2.999635"
CMW="-51.196709"
fi

if [ "$MAP" == "Town02" ]
then
CMX="186.950058" 
CMY="176.769974" 
CMZ="4.0"
CMR="0.0000"
CMP="-20.0"
CMW="30.0"
fi
SCRIPTRL="set_red_light.py"
SCRIPTW="new_world.py"
SCRIPTA="carla_agent.py" 
SCRIPTT="generate_traffic.py"

echo $IDX 


if [ "$IDX" == "car_hero" ]
#run Carla client container
then
    #run Carla client with traffic script
    SID="9999"
    MPT="1883"
    echo "Starting hero"
    sudo /home/ubuntu/carlaClientVenv/bin/python $SCRIPTA $LAT $LON $IP $GPUCV $TICK $IDX $CMX $CMY $CMZ $CMR $CMP $CMW $SID $MPT $ISKAFKA $KAFKAIP $KAFKAPORT $KAFKAFILE
fi

if [ "$IDX" == "car_agent" ]
#run Carla client container
then

    SID="5555"
    MPT="1884"
    sudo /home/ubuntu/carlaClientVenv/bin/python $SCRIPTA $LAT $LON $IP $GPURL $TICK $IDX $CMX $CMY $CMZ $CMR $CMP $CMW $SID $MPT $ISKAFKA $KAFKAIP $KAFKAPORT $KAFKAFILE; exec bash

fi

if [ "$IDX" == "red_light" ]
#run Carla client
then
    SID="1111"
    MPT="1884"
    sudo /home/ubuntu/carlaClientVenv/bin/python $SCRIPTA $LAT $LON $IP $GPURL $NO_TICK $IDX $CMX $CMY $CMZ $CMR $CMP $CMW $SID $MPT $ISKAFKA $KAFKAIP $KAFKAPORT $KAFKAFILE; exec bash
fi



if [ "$IDX" == "traffic" ]
#run Carla client to configure red ligth cameras
then
    sudo /home/ubuntu/carlaClientVenv/bin/python $SCRIPTT --host $IP --number-of-vehicles $NCARS --number-of-walkers $NNPCS; exec bash
fi

if [ "$IDX" == "map" ]
#run Carla client
then
    sudo /home/ubuntu/carlaClientVenv/bin/python $SCRIPTW $MAP $IP; exec bash
fi
