import json
import signal
import yaml
from prometheus_client import start_http_server, Counter, Gauge
from confluent_kafka import Consumer, KafkaException, KafkaError
import time

# Prometheus Metrics
METRICS = {
    # Cell-level metrics (unchanged)
    'latency': Gauge('network_latency_seconds', 'End-to-end latency', ['cell_id']),
    'ues': Gauge('network_ue_count', 'Active UEs', ['cell_id']),
    
    # UE-level metrics (updated with oasis_edge label)
    'snr': Gauge('radio_snr_db', 'Signal-to-Noise Ratio', ['cell_id', 'ue_id', 'oasis_edge']),
    'cqi': Gauge('radio_cqi', 'Channel Quality Indicator', ['cell_id', 'ue_id', 'oasis_edge']),
    'prb_avail_dl': Gauge('rru_prb_avail_dl', 'Available DL PRBs', ['cell_id', 'ue_id', 'oasis_edge']),
    'prb_avail_ul': Gauge('rru_prb_avail_ul', 'Available UL PRBs', ['cell_id', 'ue_id', 'oasis_edge']),
    'prb_thr_dl': Gauge('rru_prb_thr_dl', 'Throughput DL PRBs', ['cell_id', 'ue_id', 'oasis_edge']),
    'prb_thr_ul': Gauge('rru_prb_thr_ul', 'Throughput UL PRBs', ['cell_id', 'ue_id', 'oasis_edge']),
    'air_if_delay': Gauge('air_if_delay', 'Air Interface Delay', ['cell_id', 'ue_id', 'oasis_edge']),
    'packet_success': Counter('drb_packet_success_total', 'Successful packets', ['cell_id', 'ue_id', 'oasis_edge']),
}

class Kafka6GConsumer:
    def __init__(self, config_path):
        self.load_config(config_path)
        self.running = True
        self.consumer = Consumer({
            'bootstrap.servers': self.kafka_config['bootstrap.servers'],
            'group.id': self.kafka_config['group.id'],
            'auto.offset.reset': self.kafka_config['auto.offset.reset']
        })
        
    def load_config(self, config_path):
        with open(config_path) as f:
            config = yaml.safe_load(f)
            self.kafka_config = config['kafka']
            
    def process_message(self, msg):
        try:
            data = json.loads(msg.value().decode('utf-8'))
            cell_id = msg.key().decode('utf-8')
            
            # Process cell-level metrics (unchanged)
            METRICS['latency'].labels(cell_id).set(data.get('latency', 0))
            METRICS['ues'].labels(cell_id).set(data.get('UEs_number', 0))
            
            # Process UE-level metrics
            for key in data:
                if key.startswith("UE") and isinstance(data[key], dict):
                    ue_data = data[key]
                    ue_id = str(ue_data.get('ID', 'unknown'))
                    
                    # Determine oasis_edge value based on ue_id
                    oasis_edge_value = ''
                    if ue_id == "0":
                        oasis_edge_value = 'd7d3204a-181a-4f33-a931-29b49d018c8c'
                    elif ue_id == "1":
                        oasis_edge_value = 'ebb53258-8ce4-4692-82c9-f0cab07aba5b'
                    
                    # Update UE-specific metrics with oasis_edge label
                    METRICS['snr'].labels(cell_id, ue_id, oasis_edge_value).set(ue_data.get('RSRQ', 0))
                    METRICS['cqi'].labels(cell_id, ue_id, oasis_edge_value).set(ue_data.get('CQI', 0))
                    METRICS['prb_avail_dl'].labels(cell_id, ue_id, oasis_edge_value).set(ue_data.get('RRU.PrbAvailDl', 0))
                    METRICS['prb_avail_ul'].labels(cell_id, ue_id, oasis_edge_value).set(ue_data.get('RRU.PrbAvailUl', 0))
                    
                    thr_dl = ue_data.get('DRB.UEThpDl', 0) / 1024
                    thr_ul = ue_data.get('DRB.UEThpUl', 0) / 1024
                    METRICS['prb_thr_dl'].labels(cell_id, ue_id, oasis_edge_value).set(thr_dl)
                    METRICS['prb_thr_ul'].labels(cell_id, ue_id, oasis_edge_value).set(thr_ul)
                    
                    METRICS['air_if_delay'].labels(cell_id, ue_id, oasis_edge_value).set(ue_data.get('DRB.AirIfDelayUl', 0))
                    METRICS['packet_success'].labels(cell_id, ue_id, oasis_edge_value).inc(ue_data.get('DRB.PacketSuccessRateUlgNBUu', 0))
                    
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
        except Exception as e:
            print(f"Error processing message: {e}")

    def run(self):
        self.consumer.subscribe([self.kafka_config['topic']])
        
        try:
            while self.running:
                msg = self.consumer.poll(1.0)
                
                if msg is None:
                    continue
                if msg.error():
                    if msg.error().code() == KafkaError.UNKNOWN_TOPIC_OR_PART:
                       print("Topic not available, retrying in 5 seconds...")
                       time.sleep(5)
                       continue
                    elif msg.error().code() == KafkaError._PARTITION_EOF:
                       continue
                    else:
                       raise KafkaException(msg.error())
                self.process_message(msg)
                
        except KeyboardInterrupt:
            pass
        finally:
            self.consumer.close()

def shutdown(signum, frame):
    print("\nShutting down...")
    consumer.running = False

if __name__ == "__main__":
    # Start metrics server
    start_http_server(8000)
    
    # Set up signal handler
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    
    # Start consumer
    consumer = Kafka6GConsumer('config.yaml')
    print("Starting 6G monitoring consumer...")
    consumer.run()
