import socket
import sys
import csv
import json
import os
from data_store import influxdb_functions 


GRPC_SEND = True

def load_ip_map(csv_file):
    ip_map = {}
    with open(csv_file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            ip_map[row[-1]] = row[0]
    return ip_map

def load_session_counters():
    if os.path.exists("influx_session_counters.json"):
        with open("influx_session_counters.json", "r") as file:
            return json.load(file)
    else:
        return {}

def udp_server(influx_db_writer, test_number, test_name):
    udp_host = '10.3.10.114'  
    udp_port = 54558
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind((udp_host, udp_port))
        while True:
            data, addr = s.recvfrom(1024)
            dict_json_save = {
                "ue_id": data.decode().split(",")[0].split(":")[1],
                "seq_nr": data.decode().split(",")[1].split(":")[1],
                "time_latency": data.decode().split(",")[2].split(":")[1]
            }
            influx_db_writer.write('latency_'+test_name, dict_json_save, 0, test_number)    


def record_ping(influx_db_writer, test_number, uplink_ue_nr, test_name):
    ip_map = load_ip_map('docker_clone_subscriber_db.csv')
    ue_id = None
    for line in sys.stdin:
        parts = line.split()
        print(parts)
        if parts[0] == 'PING' or parts[0] == 'sudo':
            print("OUTRA LINHA")
        else:
            if(parts[4] != None and parts[6] != None):
                seq_nr = parts[4].split('=')[1]
                time = parts[6].split('=')[1]  # ms
                
                print("NÚMERO DE SEQ ==> ", seq_nr)
                print("TEMPO ==> ", time)
                
                dict_json_save = {
                    "ue_id": uplink_ue_nr,
                    "seq_nr": seq_nr,
                    "time_latency": time
                }
                influx_db_writer.write('latency_' + test_name, dict_json_save, 0, test_number) 

        


if __name__ == '__main__':

    token = os.environ.get("INFLUXDB_TOKEN")
    #print("TOKEN => " + token)
    org = "oasis"
    url = "http://localhost:8086"

    BUCKET='srsran_metrics_split'

    #run with python3 -u
    if sys.argv[1].split('=')[1] == 'downlink': ## vai receber pelo socket porque o ping e iniciado pelo core
        test_name = sys.argv[2].split('=')[1]
        influx_db_writer = influxdb_functions.InfluxDBFunctions(url, token, org, 'latency'+test_name, BUCKET)
        session_counters = load_session_counters()
        if test_name not in session_counters:
            session_counters[test_name] = 1
        test_number = session_counters[test_name]
        udp_server(influx_db_writer, test_number, test_name)
    elif sys.argv[1].split('=')[1] == 'uplink':
        #need to specify when downlink because the target ip will always be 
        downlink_ue_nr = sys.argv[2].split('=')[1]
        if downlink_ue_nr is None:
        	print("You have to pass the number of UE netns via argv")
        	exit()
        test_name = sys.argv[3].split('=')[1]
        if test_name is None:
            print("You have to insert the name of the test that you're performing")
            exit()
        #sudo ip ro add 10.45.0.0/16 via 10.53.1.2
        #sudo ip netns exec ue1 ping -i 0.1 10.45.1.1
        session_counters = load_session_counters()
        if test_name not in session_counters:
            session_counters[test_name] = 1
        test_number = session_counters[test_name]
        influx_db_writer = influxdb_functions.InfluxDBFunctions(url, token, org, 'latency_'+test_name, BUCKET)
        record_ping(influx_db_writer, test_number, downlink_ue_nr, test_name)
    else:
        print("Need to pass as argument the direction of traffic\n Example: data_filler_latency.py direction=downlink\n")

    
