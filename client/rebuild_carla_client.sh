#!/bin/bash
IMG0="6goasis-client"
IMG="6goasis-client:latest"
CNT0="carla_server"
CNT="carla_client"
docker rm -f $CNT
docker image rm -f $IMG
docker build -t $IMG0 .