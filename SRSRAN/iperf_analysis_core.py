import socket
import sys
import json



def main():
    UDP_IP = '10.42.0.144'  #'10.42.0.144'
    UDP_PORT = json.load(open('iperf_udp_ports.json'))[sys.argv[1].split('=')[1]]
    print("UDP PORT => ", UDP_PORT)

    for line in sys.stdin:
        #print(line)
        line = line.strip()
        
        if line.startswith("[") and line[2:] != "ID]":
            parts = line.split()
            print("parts => ", parts)

            if len(parts) == 12 and parts[4].replace('.', '', 1).isdigit():  
                transfer = float(parts[4])
                bitrate = float(parts[6])
                jitter = float(parts[8])
                lost_percentage_str = parts[11][1::]

                perctg_idx = lost_percentage_str.find('%')
                numero_int_percentagem = lost_percentage_str[:perctg_idx]
                lost_percentage = float(float(numero_int_percentagem) / 100)
                if parts[5] == "KBytes":
                    #print("transfer pre conv", transfer)
                    transfer = round(float(transfer/1000),3)
                    #print("transfer rate now", transfer)
                if parts [7] == "Kbits/sec":
                    #print("bitrate pre conv", bitrate)
                    bitrate = round(float(bitrate/1000),2)
                    #print("bitrate pos conv", bitrate)
                json_iperf_metrics = {
                        "transfer": transfer,
                        "bitrate": bitrate, #Mbytes
                        "jitter": jitter, #ms
                        "lost_percentage": lost_percentage
                }
                #print("JSON Before send => ", json_iperf_metrics)
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                try:
                    sock.sendto(bytes(json.dumps(json_iperf_metrics), encoding="utf-8"), (UDP_IP, UDP_PORT))
                    #print("Data sent via socket")
                except:
                    print("Error sending data")

if __name__ == "__main__":
    main()
