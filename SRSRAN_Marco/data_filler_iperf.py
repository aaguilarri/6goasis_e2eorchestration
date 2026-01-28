import sys
import os
import socket
import threading
import psutil
import random
import json
from data_store import influxdb_functions 
from ue_report_class import SRSMetrics
from proto_dir.path_loss_pb2 import *
from proto_dir.noise_amplitude_pb2 import *

GRPC_SEND = True

PATH_LOSS_SERVER_PORT = 11154
NOISE_AMPLITUDE_SERVER_PORT = 11155

### "ue_id": "port"
master_dict = {}

#### SLAVE FUNCTION
def generate_random_port():
    return random.randint(43000, 65535)

#### SLAVE FUNCTION
def is_port_open_os(port):
    for conn in psutil.net_connections(kind="inet"):
        if conn.laddr.port == port:
            return True
    return False

#### SLAVE FUNCTION
def find_unoccupied_port():
    while True:
        port = generate_random_port()
        if not is_port_open_os(port):
            return port

def load_session_counters():
    if os.path.exists("influx_session_counters.json"):
        with open("influx_session_counters.json", "r") as file:
            return json.load(file)
    else:
        return {}

#### SLAVE FUNCTION
def create_ue_instance_update_message(ue_nr, port):
    message = {
        "type": "UE_INSTANCE_UPDATE",
        "content": {
            "ue_nr": ue_nr,
            "port": port
        }
    }
    return json.dumps(message)

def is_socket_open(host, master_port):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.bind((host, master_port))
        return False 
    except OSError:
        return True

def server_thread_slave(port, ue_object):
    print(f"Slave thread opening server on port {port}")
    slave_server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    slave_server_socket.bind(("127.0.0.1", port))
    print(f"***************************************    port => {port}      ***********************************")
    while True:
        """if port == PATH_LOSS_SERVER_PORT:
            print("\n\n\n\n\nREADY TO RECEIVE PATH LOSS MESSAGES\n\n\n\n\n")
            data, addr = slave_server_socket.recvfrom(8192)
            received_message = PathLoss()
            received_message.ParseFromString(data)
            path_loss = received_message.path_loss
            ue_nr = received_message.ue_id
        
            ue_object.update_path_loss(path_loss)"""
        #elif port == NOISE_AMPLITUDE_SERVER_PORT: -- Find a way to define if this is getting path loss or an but try to get both (try/expect on protobuf creation object)
        data, addr = slave_server_socket.recvfrom(8192)
        print("RECEIVED UPDATE MESSAGE ---SLAVE---")
        received_message = NoiseAmplitude()
        received_message.ParseFromString(data)
        noise_amplitude = received_message.noise_amplitude
        ue_nr = received_message.ue_id

        ue_object.update_noise_amplitude(noise_amplitude)

def server_thread(ue_object, influx_db_writer, server_port, server_host='localhost'):
    if is_socket_open(server_host, server_port):
        # Slave
        print("Server socket is already in use.")
        path_loss_receive_info_port = find_unoccupied_port()
        print("--->", path_loss_receive_info_port)
        msg = str(create_ue_instance_update_message(ue_object.ue_nr, path_loss_receive_info_port))
        slave_thread = threading.Thread(target=server_thread_slave, args=(path_loss_receive_info_port,ue_object,))
        slave_thread.start()
        msg_bytes = msg.encode('utf-8')
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.sendto(msg_bytes, ("127.0.0.1", server_port))
        return

    # Master
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print("SERVER MASTER PORT => ", server_port)
    server_socket.bind((server_host, server_port))
    print("Waiting for data...")
    while True:
        data, addr = server_socket.recvfrom(8192)
        try:
            print("***************$$$$$$$$$$$$$$$$$$$$$*******************")
            decoded_data = data.decode('utf-8')  
            print(f"DECODED DATA => {decoded_data}")
            received_message = json.loads(decoded_data, strict=False)  
            received_message.get("type") == "UE_INSTANCE_UPDATE"
            print(f"DATA => {data}")
            print(f"MASTER DICT => {master_dict}")
            print(f"RECEIVED MESSAGE => {received_message}")
            print(f"RECEIVED MESSAGE [UE NR] => {received_message.get('content')['ue_nr']}")
            print(f"RECEIVED MESSAGE [PORT] => {received_message.get('content')['port']}")
            master_dict[received_message.get("content")["ue_nr"]] = received_message.get("content")["port"]
            #print(master_dict)
        except:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

            if server_port == PATH_LOSS_SERVER_PORT:
                print ("******** PATH LOSS ********")
                received_message = PathLoss()
                received_message.ParseFromString(data)
                path_loss = round(float(received_message.path_loss),2)
                print(f"PATH LOSS => {path_loss}")
                print(f"UE NR => {received_message.ue_id}")
                ue_nr = received_message.ue_id
            elif server_port == NOISE_AMPLITUDE_SERVER_PORT:
                print ("******** NOISE AMPLITUDE ********")
                received_message = NoiseAmplitude()
                received_message.ParseFromString(data)
                noise_amplitude = round(float(received_message.noise_amplitude),2)
                print(f"NOISE AMPLITUDE => {noise_amplitude}")
                print(f"UE NR => {received_message.ue_id}")
                ue_nr = received_message.ue_id

            print(f"SENDING TO UE NR = {int(ue_nr)+1}")    #, on port {master_dict[ue_nr]}")
            ### Forward to slave the new path_loss
            
            try:
                print("Vou mandar para alguem no dicionario:")
                print(f"MASTER DICT => {master_dict}")
                print(f"PORT TO FORWARD => {master_dict[int(ue_nr)+1]}")
                sock.sendto(data, ("127.0.0.1", master_dict[int(ue_nr)+1]))
            except: #Path loss master update
                if server_port == PATH_LOSS_SERVER_PORT:
                    ue_object.update_path_loss(path_loss)
                elif server_port == NOISE_AMPLITUDE_SERVER_PORT:
                    ue_object.update_noise_amplitude(noise_amplitude)
            """ue_object = SRSMetrics.get_object_by_ue_nr(ue_nr)
            if ue_nr == ue_object.ue_nr:
                ue_object.update_path_loss(path_loss)"""


## Since this is the server running on local machine (UE Side) stats must be marked as downlink in JSON to write in Influx
def process_iperf_output(ue_object, influx_db_writer, direction, test_name):

    session_counters = load_session_counters()
    #its 0 because the iperf only runs on srsran and there is no session definition
    #

    print("TEST NAME => ", test_name)
    print("SESSION COUNTERS => ", session_counters)
    if test_name not in session_counters:
        session_counters[test_name] = 1
    test_number = session_counters[test_name]
    print("TEST NUMBER => ", test_number)

    if direction == "downlink": ## read directly from the terminal
        for line in sys.stdin:
            print(line)
            line = line.strip()
            if line.startswith("[") and line[2:] != "ID]":
                parts = line.split()
                if len(parts) == 12 and parts[4].isdigit():  
                    transfer = float(parts[4])
                    bitrate = float(parts[6])
                    jitter = float(parts[8])
                    lost_percentage_str = parts[11][1::]

                    perctg_idx = lost_percentage_str.find('%')
                    numero_int_percentagem = lost_percentage_str[:perctg_idx]
                    lost_percentage = float(float(numero_int_percentagem) / 100)
                    json_metrics = ue_object.update_metrics_from_iperf(transfer, bitrate, jitter, lost_percentage)
                    #print("Número de instâncias de SRSMetrics:", SRSMetrics.get_instance_count())
                    print("json_metrics==>", json_metrics)
                    #influx_db_writer.write('iperf_'+test_name, dict(json.loads(json_metrics)), 0, test_number)    
    elif direction == "uplink":
        UDP_IP = '10.42.0.144'#'10.42.0.144'#'10.43.78.214'
        UDP_PORT = json.load(open('iperf_udp_ports.json'))[str(ue_object.ue_nr)]
        print(UDP_PORT)
        #UDP_PORT = 54559
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((UDP_IP, UDP_PORT))
        while True:
            data, addr = sock.recvfrom(1024)
            metrics_json = json.loads(data.decode("utf-8"))
            json_metrics = ue_object.update_metrics_from_iperf(metrics_json["transfer"], metrics_json["bitrate"], metrics_json["jitter"], metrics_json["lost_percentage"])
            print("json_metrics==>", json_metrics)
            json_iperf = dict(json.loads(json_metrics))
            if json_iperf['transfer'] == 0 and json_iperf['bitrate'] == 0:
                print("Will not save this entry") 
            else:
                influx_db_writer.write('iperf_'+test_name, json_iperf, 0, test_number)  



def main():
    if len(sys.argv) != 8 or not all(arg.startswith("ue_nr=") or arg.startswith("bandwidth_required=") or arg.startswith("initial_an") or arg.startswith("direction") or arg.startswith("test_name") or arg.startswith("fading") or arg.startswith("prb") for arg in sys.argv[1:]):
        print("Usage: python3 -u script.py ue_nr=<number> bandwidth_required=<bandwidth> initial_an=<initial_an> direction=<up/downlink> test_name=<test_name> fading=<fading_gnu_set> prb=<prb_init_value<")
        return

    try:
        print("TUDO => ", sys.argv)
        print(sys.argv[2].split("=")[1])
        ue_nr = int(sys.argv[1].split("=")[1])
        print("UE NR => ", ue_nr)
        bandwidth_required = sys.argv[2].split("=")[1]
        print("bandwidth_required=> ", bandwidth_required)
        initial_an = float(sys.argv[3].split("=")[1])
        print("initial_an=> ", initial_an)
        direction = sys.argv[4].split("=")[1]
        print("direction=> ", direction)
        test_name = sys.argv[5].split("=")[1]
        print("test_name=> ", test_name)
        fading = sys.argv[6].split("=")[1]
        print("fading=> ", fading)
        prb = sys.argv[7].split("=")[1]
        print("prb=> ", prb)
        
        if direction != "downlink" and direction != "uplink":
            print("Invalid direction argument. You must type downlink or uplink")
            return
    except ValueError:
        print("Invalid arguments")
        return
    
    print(f"Starting application for UE number {ue_nr} with required bandwidth {bandwidth_required}...")
    
    token = os.environ.get("INFLUXDB_TOKEN")
    print("TOKEN => " + token)
    org = "oasis"
    url = "http://localhost:8086"

    BUCKET='srsran_metrics_split'

    influx_db_writer = influxdb_functions.InfluxDBFunctions(url, token, org, 'iperf_'+test_name, BUCKET)
    ue_object = SRSMetrics(ue_nr, bandwidth_required, initial_an, fading, prb)

    server_thread_instance = threading.Thread(target=server_thread, args=(ue_object, influx_db_writer, NOISE_AMPLITUDE_SERVER_PORT))
    server_thread_instance.start()

    process_iperf_output(ue_object, influx_db_writer, direction, test_name)


if __name__ == "__main__":
    main()
