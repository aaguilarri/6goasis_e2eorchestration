#!/bin/bash
# Run container
IDX=$1
IP="127.0.0.1"  #CARLA server IP
GPUS="0"        #Available GPUs
GPUCV="0"       #GPU used by agent
GPURL="0"       #GPU used by red light
TICK="yes"      #is the ticking script
NO_TICK="no"    #for non-ticking scripts
SYNCH="yes"     #synchronous mode?
ISKAFKA="kafka" #kafka no_kafka
KAFKAIP="192.168.169.102" #"${KAFKA_HOST:-kafka-0.kafka.kafka.svc.cluster.local}" #"172.10.0.1"
MAP="Town07"            #Map to be used
KAFKAPORT="9092" #${KAFKA_PORT:-9092}" #"9092"
KAFKAFILE="jsons/$MAP.json" #use always jsons/
#PTH=/home/aaguilar/carlaclient/
IMG="6goasis-client"    #Docker image
CNT0="carla_server"     #server container name  
CNT="carla_client"      #client container name
IMGSR="6goasis-srunner"
CNTSR="scenario_runner"
LAT="41.274927054998706"    #Reference latitude to scale coordinates
LON="1.9865927764756361"    #Reference longitude
# Sakin machine
SOCKET="/tmp/.X11-unix/X1"          #Configure X11 socket
AUTHH="/run/user/1000/gdm/Xauthority" #"/home/aaguilar/.Xauthority"  #Xautohryt file in client
AUTHC="/root/.Xauthority" 
PTH="/home/"
##Docker iesc-gpu
#SOCKET="/tmp/.X11-unix/X0"          #Configure X11 socket
#AUTHH="/home/aaguilar/.Xauthority"  #Xautohryt file in client
#AUTHC="/root/.Xauthority"           #Xautority file in root
#PTH="/home/aaguilar/6goasis/client"
#PTH2="/home/aaguilar/6goasis/client/jsons"
## Marco machine
#SOCKET="/tmp/.X11-unix"          #Configure X11 socket
#AUTHH="/run/user/1000/gdm/Xauthority" #"/home/aaguilar/.Xauthority"  #Xautohryt file in client
#AUTHC="/root/.Xauthority" 
#PTH="/home/marco/gitlab/6goasis/client"
#PTH2="/home/marco/gitlab/jsons"
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
SCRIPTA="carla_agentMWC.py" 
SCRIPTT="generate_traffic.py"

echo $IDX 
if [ "$IDX" == "server" ]

then
    echo    GPUS
    docker run --rm --privileged --gpus $GPUS  --net=host --name=$CNT0 carlasim/carla:0.9.15 /bin/bash ./CarlaUE4.sh -RenderOffScreen
    #docker run --rm --privileged --gpus 0  --net=host --name="carla_server" carlasim/carla:0.9.14 /bin/bash ./CarlaUE4.sh -RenderOffScreen
fi


if [ "$IDX" == "local_car_hero" ]
#run Carla client container
then
    ATP="autopilot" #autopilot manual
    SID="9999"
    MPT="1883"
    FPS="20"
    YOLO="yes"
    python $SCRIPTA $LAT $LON $IP $GPUCV $TICK $IDX $CMX $CMY $CMZ $CMR $CMP $CMW $SID $MPT $ISKAFKA $KAFKAIP $KAFKAPORT $KAFKAFILE $SYNCH $ATP $FPS $YOLO
           #$SCRIPTA $LAT $LON $IP $GPUCV $TICK $IDX $CMX $CMY $CMZ $CMR $CMP $CMW $SID $MPT $ISKAFKA $KAFKAIP $KAFKAPORT $KAFKAFILE $SYNCH $ATP $FPS $YOLO

fi

if [ "$IDX" == "scenario_runner" ]
#run scenario runner container
then
    docker run --net=host --name=$CNTSR -e DISPLAY=$DISPLAY -v $SOCKET:$SOCKET -v $AUTHH:$AUTHC -t -d $IMGSR /bin/bash
fi

if [ "$IDX" == "run_scenario" ]
#run a scenario
then
    SCN="FollowLeadingVehicle_1"
    docker exec -it  -w /home/carla/ScenarioRunner/ scenario_runner  python3  scenario_runner.py --scenario $SCN --reloadWorld  --output --host localhost --port 2000
fi

if [ "$IDX" == "list_scenario" ]
#list all scenarios
then
    docker exec -it  -w /home/carla/ScenarioRunner/ scenario_runner  python3  scenario_runner.py --list
fi

if [ "$IDX" == "car_hero" ]
#run Carla client container
then
    MY_CNT="${CNT}_hero"
    ATP="autopilot" #autopilot manual
    #run Carla client with traffic script
    echo "Starting hero"
    PTHC="/home/jsons"
    PTH2="/home/aaguilar/6goasis/csvs/"
    docker run --rm --privileged --runtime=nvidia --gpus all  --name=$MY_CNT --net=host -e DISPLAY=$DISPLAY -v $SOCKET:$SOCKET -v $AUTHH:$AUTHC -v $PTH2:$PTHC  -t -d $IMG;
    echo "Container started"
    sleep 1
    SID="9999"
    MPT="1883"
    FPS="60"
    YOLO="yes"
    docker exec -it $MY_CNT python3 /home/$SCRIPTA $LAT $LON $IP $GPUCV $TICK $IDX $CMX $CMY $CMZ $CMR $CMP $CMW $SID $MPT $ISKAFKA $KAFKAIP $KAFKAPORT $KAFKAFILE $SYNCH  $ATP $FPS $YOLO; exec bash

fi

if [ "$IDX" == "car_helper" ]
#run Carla client container
then
    MY_CNT="${CNT}_helper"
    ATP="autopilot" #autopilot manual
    #run Carla client with traffic script
    echo "Starting helper"
    docker run --rm --privileged --runtime=nvidia --gpus all  --name=$MY_CNT --net=host -e DISPLAY=$DISPLAY -v $SOCKET:$SOCKET -v $AUTHH:$AUTHC -v $PTH2:$PTHC -t -d $IMG;
    echo "Container started"
    sleep 1
    SID="8888"
    MPT="1882"
    FPS="20"
    YOLO="yes"
    docker exec -it $MY_CNT python3 /home/$SCRIPTA $LAT $LON $IP $GPUCV $NO_TICK $IDX $CMX $CMY $CMZ $CMR $CMP $CMW $SID $MPT $ISKAFKA $KAFKAIP $KAFKAPORT $KAFKAFILE $SYNCH $ATP $FPS $YOLO; exec bash

fi

if [ "$IDX" == "car_agent" ]
#run Carla client container
then
    MY_CNT="${CNT}_agent"
    #run Carla client with traffic script
    docker run --rm --privileged --runtime=nvidia --gpus all  --name=$MY_CNT --net=host -e DISPLAY=$DISPLAY -v $SOCKET:$SOCKET -v $AUTHH:$AUTHC -t -d $IMG
    sleep 1
    SID="5555"
    MPT="1884"
    FPS="20"
    docker exec -it $MY_CNT python3 /home/$SCRIPTA $LAT $LON $IP $GPURL $TICK $IDX $CMX $CMY $CMZ $CMR $CMP $CMW $SID $MPT $ISKAFKA $KAFKAIP $KAFKAPORT $KAFKAFILE $SYNCH; exec bash

fi

if [ "$IDX" == "red_light" ]
#run Carla client
then
    MY_CNT="${CNT}_red_light"
    docker run --rm --privileged --runtime=nvidia --gpus all  --name=$MY_CNT --net=host -e DISPLAY=$DISPLAY -v $SOCKET:$SOCKET -v $AUTHH:$AUTHC -t -d $IMG
    sleep 1
    SID="1111"
    MPT="1884"
    FPS="20"
    ATP="manual"
    YOLO="yes"
    docker exec -it $MY_CNT python3 /home/$SCRIPTA $LAT $LON $IP $GPURL $NO_TICK $IDX $CMX $CMY $CMZ $CMR $CMP $CMW $SID $MPT $ISKAFKA $KAFKAIP $KAFKAPORT $KAFKAFILE $SYNCH $ATP $FPS $YOLO; exec bash
fi

if [ "$IDX" == "client" ]
#run Carla client container
then
    #run Carla client with traffic script
    docker run --rm --privileged --runtime=nvidia --gpus all  --name=$CNT --net=host -e DISPLAY=$DISPLAY -v $SOCKET:$SOCKET -v $AUTHH:$AUTHC -t -d $IMG; exec bash
fi



if [ "$IDX" == "map" ]
#run Carla client to configure red ligth cameras
then
    MAXSST="1"
    SSTDT="0.02"
    #echo "Running client"
    #docker run --rm --privileged --runtime=nvidia --gpus all  --name=$CNT --net=host -e DISPLAY=$DISPLAY -v $SOCKET:$SOCKET -v $AUTHH:$AUTHC -t -d $IMG
    #sleep 3
    echo "setting sync"
    docker exec -it $CNT python3 /home/$SCRIPTW $MAP $IP $MAXSST $SSTDT $SYNCH
    #sleep 3
    #echo "setting traffic"
    #docker exec -it $CNT python3 /home/$SCRIPTT --host $IP --number-of-vehicles $NCARS --number-of-walkers $NNPCS; exec bash
fi

if [ "$IDX" == "traffic" ]
#run Carla client to configure red ligth cameras
then
    #echo "Running client"
    #docker run --rm --privileged --runtime=nvidia --gpus all  --name=$CNT --net=host -e DISPLAY=$DISPLAY -v $SOCKET:$SOCKET -v $AUTHH:$AUTHC -t -d $IMG
    #sleep 3
    #echo "setting sync"
    #docker exec -it $CNT python3 /home/$SCRIPTWT $MAP
    #sleep 3
    echo "setting traffic"
    docker exec -it $CNT python3 /home/$SCRIPTT --host $IP --number-of-vehicles $NCARS --number-of-walkers $NNPCS; exec bash
fi

# Currently the following scenarios are supported:
# OtherLeadingVehicle_1
# OtherLeadingVehicle_2
# OtherLeadingVehicle_3
# OtherLeadingVehicle_4
# OtherLeadingVehicle_5
# OtherLeadingVehicle_6
# OtherLeadingVehicle_7
# OtherLeadingVehicle_8
# OtherLeadingVehicle_9
# OtherLeadingVehicle_10
# ControlLoss_1
# ControlLoss_2
# ControlLoss_3
# ControlLoss_4
# ControlLoss_5
# ControlLoss_6
# ControlLoss_7
# ControlLoss_8
# ControlLoss_9
# ControlLoss_10
# ControlLoss_11
# ControlLoss_12
# ControlLoss_13
# ControlLoss_14
# ControlLoss_15
# FreeRide_1
# FreeRide_2
# FreeRide_3
# FreeRide_4
# MultiEgo_1
# MultiEgo_2
# SignalizedJunctionLeftTurn_1
# SignalizedJunctionLeftTurn_2
# SignalizedJunctionLeftTurn_3
# SignalizedJunctionLeftTurn_4
# SignalizedJunctionLeftTurn_5
# SignalizedJunctionLeftTurn_6
# FollowLeadingVehicle_1
# FollowLeadingVehicleWithObstacle_1
# FollowLeadingVehicle_2
# FollowLeadingVehicleWithObstacle_2
# FollowLeadingVehicle_3
# FollowLeadingVehicleWithObstacle_3
# FollowLeadingVehicle_4
# FollowLeadingVehicleWithObstacle_4
# FollowLeadingVehicle_5
# FollowLeadingVehicleWithObstacle_5
# FollowLeadingVehicle_6
# FollowLeadingVehicleWithObstacle_6
# FollowLeadingVehicle_7
# FollowLeadingVehicleWithObstacle_7
# FollowLeadingVehicle_8
# FollowLeadingVehicleWithObstacle_8
# FollowLeadingVehicle_9
# FollowLeadingVehicleWithObstacle_9
# FollowLeadingVehicle_10
# FollowLeadingVehicleWithObstacle_10
# FollowLeadingVehicle_11
# FollowLeadingVehicleWithObstacle_11
# StationaryObjectCrossing_1
# DynamicObjectCrossing_1
# StationaryObjectCrossing_2
# DynamicObjectCrossing_2
# StationaryObjectCrossing_3
# DynamicObjectCrossing_3
# StationaryObjectCrossing_4
# DynamicObjectCrossing_4
# StationaryObjectCrossing_5
# DynamicObjectCrossing_5
# StationaryObjectCrossing_6
# DynamicObjectCrossing_6
# StationaryObjectCrossing_7
# DynamicObjectCrossing_7
# StationaryObjectCrossing_8
# DynamicObjectCrossing_8
# DynamicObjectCrossing_9
# ConstructionSetupCrossing
# NoSignalJunctionCrossing
# OppositeVehicleRunningRedLight_1
# OppositeVehicleRunningRedLight_2
# OppositeVehicleRunningRedLight_3
# OppositeVehicleRunningRedLight_4
# OppositeVehicleRunningRedLight_5
# ManeuverOppositeDirection_1
# ManeuverOppositeDirection_2
# ManeuverOppositeDirection_3
# ManeuverOppositeDirection_4
# CutInFrom_left_Lane
# CutInFrom_right_Lane
# VehicleTurningRight_1
# VehicleTurningLeft_1
# VehicleTurningRight_2
# VehicleTurningLeft_2
# VehicleTurningRight_3
# VehicleTurningLeft_3
# VehicleTurningRight_4
# VehicleTurningLeft_4
# VehicleTurningRight_5
# VehicleTurningLeft_5
# VehicleTurningRight_6
# VehicleTurningLeft_6
# VehicleTurningRight_7
# VehicleTurningLeft_7
# VehicleTurningRight_8
# VehicleTurningLeft_8
# SignalizedJunctionRightTurn_1
# SignalizedJunctionRightTurn_2
# SignalizedJunctionRightTurn_3
# SignalizedJunctionRightTurn_4
# SignalizedJunctionRightTurn_5
# SignalizedJunctionRightTurn_6
# SignalizedJunctionRightTurn_7
# ChangeLane_1
# ChangeLane_2
# CARLA:FollowLeadingVehicle (OpenSCENARIO)
# CARLA:PedestrianCrossing (OpenSCENARIO)
# CARLA:ControllerExample (OpenSCENARIO)
# CARLA:Slalom (OpenSCENARIO)
# CARLA:CyclistCrossing (OpenSCENARIO)
# CARLA:SynchronizeIntersectionEntry (OpenSCENARIO)
# CARLA:ChangingWeatherExample (OpenSCENARIO)
# CARLA:PedestrianCrossing (OpenSCENARIO)
# CARLA:LaneChangeSimple (OpenSCENARIO)


