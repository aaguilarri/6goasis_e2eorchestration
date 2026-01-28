import json
import signal
import yaml
from prometheus_client import start_http_server, Counter, Gauge
from confluent_kafka import Consumer, KafkaException

# Prometheus Metrics
METRICS = {
    # Cell-level metrics
    'latency': Gauge('network_latency_seconds', 'End-to-end latency', ['cell_id']),
    'ues': Gauge('network_ue_count', 'Active UEs', ['cell_id']),
    
    # UE-level metrics (with both cell_id and ue_id labels)
    'snr': Gauge('radio_snr_db', 'Signal-to-Noise Ratio', ['cell_id', 'ue_id']),
    'cqi': Gauge('radio_cqi', 'Channel Quality Indicator', ['cell_id', 'ue_id']),
    'prb_avail_dl': Gauge('rru_prb_avail_dl', 'Available DL PRBs', ['cell_id', 'ue_id']),
    'prb_avail_ul': Gauge('rru_prb_avail_ul', 'Available UL PRBs', ['cell_id', 'ue_id']),
    'prb_thr_dl': Gauge('rru_prb_thr_dl', 'Throughput DL PRBs', ['cell_id', 'ue_id']),
    'prb_thr_ul': Gauge('rru_prb_thr_ul', 'Throughput UL PRBs', ['cell_id', 'ue_id']),
    'air_if_delay': Gauge('air_if_delay', 'Air Interface Delay', ['cell_id', 'ue_id']),
    'packet_success': Counter('drb_packet_success_total', 'Successful packets', ['cell_id', 'ue_id']),
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
            print(data)
            cell_id = msg.key().decode('utf-8')
            
            # Process cell-level metrics
            METRICS['latency'].labels(cell_id).set(data.get('latency', 0))
            METRICS['ues'].labels(cell_id).set(data.get('UEs_number', 0))
            
            # Process UE-level metrics
            for key in data:
                if key.startswith("UE") and isinstance(data[key], dict):
                    ue_data = data[key]
                    ue_id = str(ue_data.get('ID', 'unknown'))
                    
                    # Update UE-specific metrics
                    METRICS['snr'].labels(cell_id, ue_id).set(ue_data.get('RSRQ', 0))
                    METRICS['cqi'].labels(cell_id, ue_id).set(ue_data.get('CQI', 0))
                    METRICS['prb_avail_dl'].labels(cell_id, ue_id).set(ue_data.get('RRU.PrbAvailDl', 0))
                    METRICS['prb_avail_ul'].labels(cell_id, ue_id).set(ue_data.get('RRU.PrbAvailUl', 0))
                    
                    # Convert throughput from kbps to mbps
                    thr_dl = ue_data.get('DRB.UEThpDl', 0) / 1000  # kbps to mbps
                    thr_ul = ue_data.get('DRB.UEThpUl', 0) / 1000
                    METRICS['prb_thr_dl'].labels(cell_id, ue_id).set(thr_dl)
                    METRICS['prb_thr_ul'].labels(cell_id, ue_id).set(thr_ul)
                    
                    METRICS['air_if_delay'].labels(cell_id, ue_id).set(ue_data.get('DRB.AirIfDelayUl', 0))
                    METRICS['packet_success'].labels(cell_id, ue_id).inc(ue_data.get('DRB.PacketSuccessRateUlgNBUu', 0))
                    
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
