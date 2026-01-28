FROM openjdk:11-jre-slim

# Install required packages
RUN apt-get update && apt-get install -y \
    wget \
    iputils-ping \
    net-tools \
    apt-transport-https \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Set Kafka version
ENV KAFKA_VERSION=3.4.0
ENV SCALA_VERSION=2.13

# Copy Kafka tarball into the image
COPY kafka_${SCALA_VERSION}-${KAFKA_VERSION}.tgz /tmp/kafka.tgz

# Extract Kafka
RUN set -eux; \
    tar -xzf /tmp/kafka.tgz -C /opt; \
    rm /tmp/kafka.tgz; \
    ln -s /opt/kafka_${SCALA_VERSION}-${KAFKA_VERSION} /opt/kafka

# Set Kafka environment variables
ENV KAFKA_HOME="/opt/kafka"
ENV PATH=${PATH}:${KAFKA_HOME}/bin

# Copy custom configuration files
COPY server.properties /opt/kafka/config/
COPY start-kafka.sh /usr/bin/

# Expose Kafka ports
EXPOSE 9092 9093 9094 2128

# Define Kafka listeners
ENV KAFKA_LISTENERS=PLAINTEXT://0.0.0.0:9092
ENV KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://kafka-broker:9092
ENV KAFKA_LISTENER_SECURITY_PROTOCOL_MAP=PLAINTEXT:PLAINTEXT
ENV KAFKA_INTER_BROKER_LISTENER_NAME=PLAINTEXT

# Set the entry point
CMD ["start-kafka.sh"]


