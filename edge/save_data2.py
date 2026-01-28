import sys
import json 
import time
from kafka import KafkaConsumer
from kafka.errors import KafkaError


filename = 'exp1car1light.json'
pathos =' .'

def create_consumer(server,port,topic_name):
    bootstrap_servers=server+':'+port
    print(f"topic si {topic_name} server {bootstrap_servers}")
    return KafkaConsumer(
        topic_name,
        bootstrap_servers=bootstrap_servers,
        auto_offset_reset='latest',
        value_deserializer=lambda x: x.decode('utf-8')
    )

def consume_messages(consumer):
    global filename
    global pathos
    print(f"we have {consumer}")
    for i, message in enumerate(consumer):
        print(f"Received message!")
        try:
            tracklet_dict = json.loads(message.value)
        except:
            print(f"message {message.value} is not json")
            continue
        tracklet_dict['sys_time'] = time.time()
        tracklet_j = json.dumps(tracklet_dict, indent=4)
        print(f"Tracklet is {tracklet_j}")
        try:
            with open(pathos + filename, 'r') as json_file:
                kafka_messages = json.load(json_file)
        except:
            kafka_messages = []
        if tracklet_j is None:
            return
        kafka_messages.append(tracklet_j)
        with open(pathos + filename, 'w') as json_file:
            json.dump(kafka_messages, json_file)
            print(f"Tracklet saved to {filename}")


def main():
    args = sys.argv
    global filename
    global pathos
    filename = args[1]
    filename = 'jsons/' + filename + '_ts_' + str(int(time.time()*10**7)) + '.json'
    broker_url = args[2]  # "localhost"
    broker_port = args[3]
    pathos = args[4]
    my_topic = "tracklets"
    print('connecting to ', broker_url, ' port ', broker_port)
    print(f"to be saves at {pathos}{filename}")
    consumer = create_consumer(broker_url,broker_port,my_topic)
    print('Connected')
    while True:
        try:
            consume_messages(consumer)
            time.sleep(1)
        except KeyboardInterrupt:
            print("Stopping...")
            consumer.close()
            break
    print("This is the end of the route. Goodbye!")

if __name__ == '__main__':
    main()