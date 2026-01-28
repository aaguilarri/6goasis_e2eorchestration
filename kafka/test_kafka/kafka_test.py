from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError
import time

# Kafka configuration
bootstrap_servers = 'kafka-broker:9092'
topic_name = 'tracklets'

def create_producer():
    return KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: v.encode('utf-8')
    )

def create_consumer():
    return KafkaConsumer(
        topic_name,
        bootstrap_servers=bootstrap_servers,
        auto_offset_reset='earliest',
        value_deserializer=lambda x: x.decode('utf-8')
    )

def produce_messages(producer, messages):
    for message in messages:
        future = producer.send(topic_name, value=message)
        try:
            future.get(timeout=10)
            print(f"Message '{message}' sent successfully.")
        except KafkaError as e:
            print(f"Failed to send message '{message}': {e}")

def consume_messages(consumer, num_messages):
    for i, message in enumerate(consumer):
        print(f"Received message: {message.value}")
        if i + 1 >= num_messages:
            break

if __name__ == "__main__":
    producer = create_producer()
    consumer = create_consumer()

    test_messages = [f"Test message {i}" for i in range(5)]

    # Produce test messages
    print("Producing messages...")
    produce_messages(producer, test_messages)
    time.sleep(2)

    # Consume test messages
    print("Consuming messages...")
    consume_messages(consumer, len(test_messages))

    producer.close()
    consumer.close()
