#!/bin/bash

IDX=$1
MCH=$2
if [ "$MCH" == "sakin" ]
then
PT="9092"
VF="/home/sakin/Code/gitlab/6goasis/edge_data/"
fi
if [ "$MCH" == "aa" ]
then
PT="9092"
VF="/home/aa/gitlab/6goasis/edge_data/"
fi
if [ "$MCH" == "iesc-gpu" ]
then
PT="9094"
VF="/home/aaguilar/6goasis/edge_data/"
fi
IMG="edge" #edge service image name
CNT="edge" #edge container name
SPT="carla_edge6.py" #edge script
SVP="save_data2.py" #data collector script
DLP="delayer.py"  #artificial delayer script
PLT="plotter.py"  #makes plots
IP="kafka-broker"  #IP from kafka server
#Sakin
#PT="9092"        #port
#Sakin
#PT="9092"        #port
#Sakin
#PT="9092"        #port
FN="OppositeVehicleRunningRedLight4" #"OppositeVehicleRunningRedLight5"  #"carandlightdelay_150_ms.json" #"carandlight.json"   #filename to save data
MD="kafka"  #kafka/no-kafka, iow, get data online or offline
PTH="./" #to save file
#Sakin
#VF="/home/sakin/Code/gitlab/6goasis/edge_data/"
#AA
#VF="/home/aa/gitlab/6goasis/edge_data/"
#iesc-gpu
#VF="/home/aaguilar/6goasis/edge_data/"
 #"/home/sakin/Code/gitlab/6goasis/edge_data/" #"/media/aa/bodega/6GOASIScode/edge_files/" #"/home/aa/6GOASIScode/edge_files/" #"/media/aa/bodega/6GOASIScode/edge_files/" #volume path in host
DF="/home/jsons/"  #volume path in container
DL="100"  #delay
NP="500"  #number of particles
LKL="300" #likelihood threshold
NST="1"  
#DDL="100"
DST="9999"
LGD="delays"
DSC="discarding"
MST="no-master"
NT="kafka-network"

if [ "$IDX" == "build" ]; then
    docker --debug build -t "$IMG" .
elif [ "$IDX" == "start" ]; then
    echo VF
    echo DF
    docker  run -it --name="$CNT" --net=$NT -v "$VF":"$DF" -t -d "$IMG" /bin/bash
elif [ "$IDX" == "run" ]; then
    sleep 1
    docker exec -it "$CNT" python3 "/home/$SPT" "$IP" "$PT" "$FN" "$MD" "$DF" "$DL" "$NP" "$LKL" "$DSC"
elif [ "$IDX" == "experiment" ]; then
    sleep 1
    EXP="edge_experiment.py"
    TYP="kalman"
    docker exec -it "$CNT" python3 "/home/$EXP" "$DF" "$TYP" >> $VF'edge_experiment_log.txt'
elif [ "$IDX" == "local_experiment" ]; then
    sleep 1
    EXP="edge_experiment.py"
    TYP="kalman"
    python $EXP "$VF" "$TYP"
elif [ "$IDX" == "save_data" ]; then
    docker exec -it "$CNT" python3 "/home/$SVP" "$FN" "$IP" "$PT" "$DF" >> $VF'save_data_log.txt'
elif [ "$IDX" == "delayer" ]; then
    ALL="yes" # "yes" "no"
    docker exec -it "$CNT" python3 "/home/$DLP" "$FN" "$DF" "$DST" "$DL" "$MST" "$ALL"
elif [ "$IDX" == "local_delayer" ]; then
    ALL="yes" # "yes" "no"
    python $DLP "$FN" "$VF" "$DST" "$DL" "$MST" "$ALL"
elif [ "$IDX" == "delete" ]; then
    docker stop "$CNT"
    sleep 1
    docker rm -f "$CNT"
elif [ "$IDX" == "plot" ]; then
    docker exec -it "$CNT" python3 "/home/$PLT" "$DF" "$DF" "$LGD"
else
    echo "Invalid option: $IDX"
    exit 1
fi

