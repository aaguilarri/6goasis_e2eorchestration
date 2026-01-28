import json
from kafka import KafkaProducer

# Kafka broker address
bootstrap_servers = 'localhost:9092'

# Kafka topic
topic_name = 'your-topic-name'

# The path value to send
path_value = '/some/specific/file/or/data/path'

# Initialize Kafka producer with JSON serialization
producer = KafkaProducer(
    bootstrap_servers=bootstrap_servers,
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Create the JSON payload
message = {
    'path': path_value
}

# Send the message
producer.send(topic_name, value=message)
producer.flush()

print(f"Sent message to topic '{topic_name}': {message}")
