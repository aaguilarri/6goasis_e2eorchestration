#/bin/bash

IDX=$1
MCH=$2
if [ "$MCH" == "sakin" ]
then
PT="9092"
fi
if [ "$MCH" == "aa" ]
then
PT="9092"
fi
if [ "$MCH" == "iesc-gpu" ]
then
PT="9094"
fi
IP="kafka-broker"
#PT="9092"
CNT="kafka-broker"
IMG="carla-kafka"
TPC="tracklets"
NT="kafka-network"

if [ "$IDX" == "run" ]
then
docker run -d  -p $PT:$PT -p 2181:2181 --net=$NT --name $CNT $IMG; exec bash
fi

if [ "$IDX" == "stop" ]
then
echo "Stopping Kafka"
docker stop $CNT
docker rm -f $CNT
echo "Kafka Stopped."
fi

if [ "$IDX" == "topic" ]
then
docker exec -it $CNT /opt/kafka/bin/kafka-topics.sh --create --topic $TPC --bootstrap-server $IP:$PT --partitions 1 --replication-factor 1
docker exec -it $CNT /opt/kafka/bin/kafka-topics.sh --list --bootstrap-server $IP:$PT
fi
