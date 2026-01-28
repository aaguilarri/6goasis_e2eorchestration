#!/bin/bash

NTT="kafka-network"
CNT="test_kafka"
IMG="test_kafka"
FNM="kafka_test.py"
PTH="/app/home/"
docker run  -d -t --net=$NTT --name=$CNT $IMG /bin/bash
docker exec -it $CNT  python3 "$PTH$FNM"; exec bash  
