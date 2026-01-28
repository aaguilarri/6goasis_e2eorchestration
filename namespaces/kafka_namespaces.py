import ctypes
import os
import json
import time
import threading
from kafka import KafkaProducer
from kafka.errors import KafkaError
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load libc for namespace operations
libc = ctypes.CDLL("libc.so.6", use_errno=True)
CLONE_NEWNET = 0x40000000

setns = libc.setns
setns.argtypes = [ctypes.c_int, ctypes.c_int]
setns.restype = ctypes.c_int

class NamespaceKafkaProducer:
    def __init__(self, kafka_config=None):
        """
        Initialize the Kafka producer with namespace control
        
        Args:
            kafka_config (dict): Kafka configuration parameters
        """
        self.kafka_config = kafka_config or {
            'bootstrap_servers': ['localhost:9092'],
            'value_serializer': lambda x: json.dumps(x).encode('utf-8'),
            'key_serializer': lambda x: x.encode('utf-8') if x else None,
            'acks': 'all',
            'retries': 3,
            'retry_backoff_ms': 1000
        }
        
        self.producer = None
        self.current_namespace = "default"
        self._lock = threading.Lock()
    
    def switch_namespace(self, namespace_name):
        """
        Switch to a specific network namespace
        
        Args:
            namespace_name (str): Name of the namespace or 'default'
        
        Returns:
            bool: True if successful, False otherwise
        """
        with self._lock:
            if namespace_name == self.current_namespace:
                logger.info(f"Already in namespace '{namespace_name}'")
                return True
            
            # Close existing producer if it exists
            if self.producer:
                logger.info("Closing existing Kafka producer before namespace switch")
                self.producer.close()
                self.producer = None
            
            # Switch namespace
            if namespace_name == "default":
                ns_path = "/proc/1/ns/net"
            else:
                ns_path = f"/var/run/netns/{namespace_name}"
            
            try:
                fd = os.open(ns_path, os.O_RDONLY)
                if setns(fd, CLONE_NEWNET) != 0:
                    errno = ctypes.get_errno()
                    raise OSError(errno, os.strerror(errno))
                os.close(fd)
                
                self.current_namespace = namespace_name
                logger.info(f"Successfully switched to namespace: {namespace_name}")
                return True
                
            except FileNotFoundError:
                logger.error(f"Namespace '{namespace_name}' does not exist")
                return False
            except Exception as e:
                logger.error(f"Failed to switch to namespace '{namespace_name}': {e}")
                return False
    
    def _ensure_producer(self):
        """Ensure Kafka producer is initialized in current namespace"""
        if self.producer is None:
            try:
                logger.info(f"Creating Kafka producer in namespace '{self.current_namespace}'")
                self.producer = KafkaProducer(**self.kafka_config)
                logger.info("Kafka producer created successfully")
            except Exception as e:
                logger.error(f"Failed to create Kafka producer: {e}")
                raise
    
    def send_message(self, topic, message, key=None, namespace=None):
        """
        Send a message to Kafka, optionally switching namespace first
        
        Args:
            topic (str): Kafka topic
            message (dict): Message to send
            key (str, optional): Message key
            namespace (str, optional): Namespace to switch to before sending
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Switch namespace if requested
            if namespace and namespace != self.current_namespace:
                if not self.switch_namespace(namespace):
                    return False
            
            # Ensure producer exists
            self._ensure_producer()
            
            # Add namespace info to message
            enhanced_message = {
                'namespace': self.current_namespace,
                'timestamp': time.time(),
                'data': message
            }
            
            # Send message
            future = self.producer.send(topic, value=enhanced_message, key=key)
            
            # Wait for confirmation (optional - for reliability)
            record_metadata = future.get(timeout=10)
            
            logger.info(f"Message sent successfully to {topic} from namespace '{self.current_namespace}' "
                       f"(partition: {record_metadata.partition}, offset: {record_metadata.offset})")
            return True
            
        except KafkaError as e:
            logger.error(f"Kafka error sending message: {e}")
            return False
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            return False
    
    def send_batch_messages(self, messages_config):
        """
        Send multiple messages, potentially from different namespaces
        
        Args:
            messages_config (list): List of dicts with 'topic', 'message', 'key', 'namespace'
        """
        results = []
        
        for config in messages_config:
            topic = config['topic']
            message = config['message']
            key = config.get('key')
            namespace = config.get('namespace')
            
            success = self.send_message(topic, message, key, namespace)
            results.append({
                'topic': topic,
                'namespace': namespace or self.current_namespace,
                'success': success
            })
        
        return results
    
    def close(self):
        """Close the Kafka producer"""
        with self._lock:
            if self.producer:
                logger.info("Closing Kafka producer")
                self.producer.close()
                self.producer = None

# Example usage and testing
def main():
    # Initialize the producer
    producer = NamespaceKafkaProducer({
        'bootstrap_servers': ['localhost:9092'],  # Adjust as needed
        'value_serializer': lambda x: json.dumps(x).encode('utf-8'),
        'acks': 'all'
    })
    
    try:
        # Example 1: Send from default namespace
        logger.info("=== Sending from default namespace ===")
        success = producer.send_message(
            topic='test-topic',
            message={'sensor': 'temperature', 'value': 25.5, 'location': 'default'},
            key='sensor-1'
        )
        
        # Example 2: Send from ue1 namespace
        logger.info("=== Sending from ue1 namespace ===")
        success = producer.send_message(
            topic='test-topic',
            message={'sensor': 'humidity', 'value': 60.0, 'location': 'ue1'},
            key='sensor-2',
            namespace='ue1'
        )
        
        # Example 3: Send from ue2 namespace
        logger.info("=== Sending from ue2 namespace ===")
        success = producer.send_message(
            topic='test-topic',
            message={'sensor': 'pressure', 'value': 1013.25, 'location': 'ue2'},
            key='sensor-3',
            namespace='ue2'
        )
        
        # Example 4: Batch sending from multiple namespaces
        logger.info("=== Batch sending from multiple namespaces ===")
        batch_config = [
            {
                'topic': 'sensor-data',
                'message': {'type': 'batch1', 'value': 100},
                'key': 'batch-1',
                'namespace': 'ue1'
            },
            {
                'topic': 'sensor-data',
                'message': {'type': 'batch2', 'value': 200},
                'key': 'batch-2',
                'namespace': 'ue2'
            },
            {
                'topic': 'sensor-data',
                'message': {'type': 'batch3', 'value': 300},
                'key': 'batch-3',
                'namespace': 'default'
            }
        ]
        
        results = producer.send_batch_messages(batch_config)
        
        logger.info("Batch results:")
        for result in results:
            logger.info(f"  {result}")
            
        # Example 5: Stay in a namespace and send multiple messages
        logger.info("=== Staying in ue1 for multiple messages ===")
        producer.switch_namespace('ue1')
        
        for i in range(3):
            producer.send_message(
                topic='continuous-data',
                message={'sequence': i, 'data': f'message-{i}'},
                key=f'seq-{i}'
            )
            time.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f
