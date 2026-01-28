#!/bin/bash
set -e

# Start ZooKeeper
${KAFKA_HOME}/bin/zookeeper-server-start.sh ${KAFKA_HOME}/config/zookeeper.properties &

# Start Kafka
${KAFKA_HOME}/bin/kafka-server-start.sh ${KAFKA_HOME}/config/server.properties

#docker run -p 9092:9092 -p 9093:9093 -p 9094:9094 -p 2181:2181 --name kafka carla-kafka