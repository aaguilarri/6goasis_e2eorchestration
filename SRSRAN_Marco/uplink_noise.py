#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: srsRAN_multi_UE
# GNU Radio version: 3.10.1.1

from packaging.version import Version as StrictVersion
import numpy as np
import threading
import random
from time import sleep
from proto_dir.path_loss_pb2 import * 
from proto_dir.noise_amplitude_pb2 import *
import socket 


NUMBER_OF_UES = -1
MIN_AN = -1000000   ## Here we'll increase AN each 2 min
MAX_AN = -17.2
### With the CHANGE_ALL variable if set as True every UE will change at the same time the noise amplitude value. (Test purpose)
CHANGE_ALL = True
RANDOM_AN_VALUES = True
#MAX_DOPPLER = -1

if __name__ == '__main__':
    import ctypes
    import sys

    try:
        NUMBER_OF_UES = int(sys.argv[1].split('=')[1])
    except:
        NUMBER_OF_UES = 1
    """try:
        MAX_DOPPLER = int(sys.argv[2].split('=')[1])
    except:
        MAX_DOPPLER = 0"""

    if sys.platform.startswith('linux'):
        try:
            x11 = ctypes.cdll.LoadLibrary('libX11.so')
            x11.XInitThreads()
        except:
            print("Warning: failed to XInitThreads()")

if __name__ == '__main__':
    import ctypes
    import sys
    if sys.platform.startswith('linux'):
        try:
            x11 = ctypes.cdll.LoadLibrary('libX11.so')
            x11.XInitThreads()
        except:
            print("Warning: failed to XInitThreads()")



from gnuradio import analog
from gnuradio import blocks
from gnuradio import channels
from gnuradio import gr
from gnuradio.filter import firdes
from gnuradio.fft import window
import sys
import signal
from PyQt5 import Qt
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import zeromq
from gnuradio.qtgui import Range, RangeWidget
from PyQt5 import QtCore



from gnuradio import qtgui

#FADING = False

class multi_ue_scenario(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "srsRAN_multi_UE")

        self.settings = Qt.QSettings("GNU Radio", "multi_ue_scenario")

        try:
            if StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
                self.restoreGeometry(self.settings.value("geometry").toByteArray())
            else:
                self.restoreGeometry(self.settings.value("geometry"))
        except:
            pass

        ##################################################
        # Variables
        ##################################################
        self.path_loss_variables = {}
        self.an_ch_variables = {}

        uniform_distribution = np.random.uniform(low=MIN_AN, high=MAX_AN, size=NUMBER_OF_UES)
        an_ch_values = np.around(uniform_distribution, decimals=2)

        self.zmq_timeout = zmq_timeout = 100
        self.zmq_hwm = zmq_hwm = -1

        for i in range(NUMBER_OF_UES):
            self.path_loss_variables[f'ue_{i+1}_path_loss_db'] = 0
        for i in range(NUMBER_OF_UES):
            ### Commented set random values at the beggining
            #self.an_ch_variables[f'An_ch{i+1}'] = an_ch_values[i]
            self.an_ch_variables[f'An_ch{i+1}'] = MIN_AN
            #setattr(self, f'An_ch{i+1}', an_ch_values[i])
            setattr(self, f'An_ch{i+1}', MIN_AN)
            #print(f"An_ch{i+1} => {an_ch_values[i]} dB")
            print("NUMBER OF UES => ", NUMBER_OF_UES)
            print(f"An_ch{i+1} => {MIN_AN} dB")


        self.slow_down_ratio = slow_down_ratio = 1
        self.samp_rate = samp_rate = 11520000
        #self.max_doppler_frequency = max_doppler_frequency = MAX_DOPPLER
        #print(f"Max doppler frequency set to {self.max_doppler_frequency}")
        """if FADING:
            self.fD = fD = 125
        else:
            self.fD = fD = 125
        print(f"Max Doppler Frequency set to {self.fD}")"""

        ##################################################
        # Blocks
        ##################################################
        for i in range(1, NUMBER_OF_UES + 1):
            #range_value = path_loss_values_list[i-1] 
            ue_path_loss_db_range = Range(0, 100, 1, 0, 200)
            setattr(self, f"_ue{i}_path_loss_db_range", ue_path_loss_db_range)
            
        #### VERIFY
        for i in range(1, NUMBER_OF_UES + 1):
            range_value = an_ch_values[i-1] 
            An_ch_range = Range(-30, -17.4, 0.2, MIN_AN, 200)
            setattr(self, f"_An_ch{i}_range", An_ch_range)
        
        self._slow_down_ratio_range = Range(1, 32, 1, 1, 200)
        #self._slow_down_ratio_win = RangeWidget(self._slow_down_ratio_range, self.set_slow_down_ratio, "Time Slow Down Ratio", "counter_slider", float, QtCore.Qt.Horizontal)
        #self.top_layout.addWidget(self._slow_down_ratio_win)

        for i in range(1, NUMBER_OF_UES + 1):
            #path_loss_db = path_loss_values_list[i-1] ### TO MAKE PATH LOSS DYNAMIC AT START  
            #path_loss_db = 0
            #setattr(self, f"blocks_multiply_const_vxx_0_{i}", blocks.multiply_const_cc(10**(-1.0*path_loss_db/20.0)))
            #setattr(self, f"blocks_multiply_const_vxx_0_1_{i}", blocks.multiply_const_cc(10**(-1.0*path_loss_db/20.0)))
            #setattr(self, f"blocks_add_an_{i}", blocks.add_vcc(1))
            
            base_port = 2100 + (i - 1) * 100 
            
            if base_port == 3000:
                setattr(self, f"zeromq_req_source_ue_{i}", zeromq.req_source(gr.sizeof_gr_complex, 1, f'tcp://127.0.0.1:{base_port + 1}', zmq_timeout, False, zmq_hwm)) 
                setattr(self, f"zeromq_req_sink_ue_{i}", zeromq.rep_sink(gr.sizeof_gr_complex, 1, f'tcp://127.0.0.1:{base_port + 2}', 100, False, zmq_hwm)) 
            else:
                setattr(self, f"zeromq_req_source_ue_{i}", zeromq.req_source(gr.sizeof_gr_complex, 1, f'tcp://127.0.0.1:{base_port + 1}', zmq_timeout, False, zmq_hwm)) 
                setattr(self, f"zeromq_req_sink_ue_{i}", zeromq.rep_sink(gr.sizeof_gr_complex, 1, f'tcp://127.0.0.1:{base_port}', 100, False, zmq_hwm)) 
        

        for i in range(1, NUMBER_OF_UES + 1):
            #### 10**(channel_noise_db/20.0)
            #setattr(self, f"channel_fading_model_ue_{i}", channels.fading_model( 8, (self.max_doppler_frequency/self.samp_rate), False, 4.0, i-1 ))
            #setattr(self, f"channel_model_ue_{i}", channels.channel_model(noise_voltage=10.0**((getattr(self, f"An_ch{i}"))/20),frequency_offset=0.0,epsilon=1.0,taps=[1.0 + 1.0j],noise_seed=int(i-1),block_tags=False))
            setattr(self, f"analog_noise_source_ue_ul_{i}", channels.channel_model(noise_voltage=10.0**(-100.0/20.0),frequency_offset=0.0,epsilon=1.0,taps=[1.0 + 1.0j],noise_seed=int(i-1),block_tags=False))       
            #setattr(self, f"analog_noise_source_ue_ul_{i}",  analog.noise_source_c(analog.GR_GAUSSIAN, 10**(MIN_AN/20.0), 0))
            #setattr(self, f"blocks_add_noise_ue_ul_{i}", blocks.add_vcc(1))  

            setattr(self, f"analog_noise_source_ue_dl_{i}", channels.channel_model(noise_voltage=10.0**(-100.0/20.0),frequency_offset=0.0,epsilon=1.0,taps=[1.0 + 1.0j],noise_seed=int(i-1),block_tags=False))
            #setattr(self, f"blocks_add_noise_ue_dl_{i}", blocks.add_vcc(1))  
            
            # IF SAMPLE RATE DIFFERENT => #setattr(self, f"throttle_samp_rate_ue_{i}", blocks.throttle(gr.sizeof_gr_complex*1, 1.0*getattr(self, f"", -1)/(1.0*slow_down_ratio),True))
            #setattr(self, f"throttle_samp_rate_ue_{i}", blocks.throttle(gr.sizeof_gr_complex*1, 1.0*self.samp_rate/(1.0*self.slow_down_ratio),True))
        ##GNB
        self.zeromq_req_source_0 = zeromq.req_source(gr.sizeof_gr_complex, 1, 'tcp://127.0.0.1:2000', zmq_timeout, False, zmq_hwm)

        ##GNB 
        self.zeromq_rep_sink_0_1 = zeromq.rep_sink(gr.sizeof_gr_complex, 1, 'tcp://127.0.0.1:2001', zmq_timeout, False, zmq_hwm)

        self.blocks_add_xx_0 = blocks.add_vcc(1)

        ##################################################
        # Connections
        #       req source e sink igual
        #       blocks_multiply_const_vxx_0_{i} e blocks_multiply_const_vxx_0_1_{i} igual
        #       blocks_add_an_{i} => novos para dar add do ruido
        #       blocks_add_xx_0 => o antigo para o sink (blocks_add_xx_0 no codigo org)
        #       analog_noise_source_ue_{i} => analog_noise_source_x_0
        ##################################################

        for ue in range(1, NUMBER_OF_UES + 1):
            ### DOWNLINK
            self.connect( (getattr(self, "zeromq_req_source_0"), 0),  (getattr(self, f"analog_noise_source_ue_dl_{ue}"), 0))
            self.connect( (getattr(self, f"analog_noise_source_ue_dl_{ue}"), 0),  (getattr(self, f"zeromq_req_sink_ue_{ue}"), 0))
            #self.connect( (getattr(self, "zeromq_req_source_0"), 0), (getattr(self, f"blocks_add_noise_ue_dl_{ue}") ,0) )
            #self.connect( (getattr(self, f"analog_noise_source_ue_dl_{ue}"), 0), (getattr(self, f"blocks_add_noise_ue_dl_{ue}"), 1))

            #self.connect( (getattr(self, f"blocks_add_noise_ue_dl_{ue}"), 0), (getattr(self, f"zeromq_req_sink_ue_{ue}"), 0))

            ### UPLINK
            #self.connect( (getattr(self, f"zeromq_req_source_ue_{ue}"), 0), (getattr(self, f"blocks_add_noise_ue_ul_{ue}") ,0) )
            #self.connect( (getattr(self, f"analog_noise_source_ue_ul_{ue}"), 0), (getattr(self, f"blocks_add_noise_ue_ul_{ue}"), 1))
            self.connect( (getattr(self, f"zeromq_req_source_ue_{ue}"), 0), (getattr(self, f"analog_noise_source_ue_ul_{ue}") ,0) )
            self.connect( (getattr(self, f"analog_noise_source_ue_ul_{ue}"), 0), (getattr(self, "blocks_add_xx_0"), int(ue-1) ))
            #self.connect( (getattr(self, f"zeromq_req_source_ue_{ue}"), 0), (getattr(self, "blocks_add_xx_0"), int(ue-1) ))
        # single conn uplink
        self.connect((self.blocks_add_xx_0, 0), (self.zeromq_rep_sink_0_1, 0))
        

    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "multi_ue_scenario")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_zmq_timeout(self):
        return self.zmq_timeout

    def set_zmq_timeout(self, zmq_timeout):
        self.zmq_timeout = zmq_timeout

    def get_zmq_hwm(self):
        return self.zmq_hwm

    def set_zmq_hwm(self, zmq_hwm):
        self.zmq_hwm = zmq_hwm

    def get_ue3_path_loss_db(self):
        return self.ue3_path_loss_db

    def set_ue3_path_loss_db(self, ue3_path_loss_db):
        self.ue3_path_loss_db = ue3_path_loss_db
        self.blocks_multiply_const_vxx_0_0_0.set_k(10**(-1.0*self.ue3_path_loss_db/20.0))
        self.blocks_multiply_const_vxx_0_1_0.set_k(10**(-1.0*self.ue3_path_loss_db/20.0))

    def get_ue2_path_loss_db(self):
        return self.ue2_path_loss_db

    def set_ue2_path_loss_db(self, ue2_path_loss_db):
        self.ue2_path_loss_db = ue2_path_loss_db
        self.blocks_multiply_const_vxx_0_0.set_k(10**(-1.0*self.ue2_path_loss_db/20.0))
        self.blocks_multiply_const_vxx_0_1_1.set_k(10**(-1.0*self.ue2_path_loss_db/20.0))

    def get_ue1_path_loss_db(self):
        return self.ue1_path_loss_db

    def set_ue1_path_loss_db(self, ue1_path_loss_db):
        self.ue1_path_loss_db = ue1_path_loss_db
        self.blocks_multiply_const_vxx_0.set_k(10**(-1.0*self.ue1_path_loss_db/20.0))
        self.blocks_multiply_const_vxx_0_1.set_k(10**(-1.0*self.ue1_path_loss_db/20.0))

    def get_slow_down_ratio(self):
        return self.slow_down_ratio

    def set_slow_down_ratio(self, slow_down_ratio):
        self.slow_down_ratio = slow_down_ratio
        self.blocks_throttle_0.set_sample_rate(1.0*self.samp_rate_ue1/(1.0*self.slow_down_ratio))
        self.blocks_throttle_0_0.set_sample_rate(1.0*self.org_samp_rate/(1.0*self.slow_down_ratio))
        self.blocks_throttle_0_0_0.set_sample_rate(1.0*self.samp_rate_ue3/(1.0*self.slow_down_ratio))

    def get_samp_rate_ue3(self):
        return self.samp_rate_ue3

    def set_samp_rate_ue3(self, samp_rate_ue3):
        self.samp_rate_ue3 = samp_rate_ue3
        self.blocks_throttle_0_0_0.set_sample_rate(1.0*self.samp_rate_ue3/(1.0*self.slow_down_ratio))

    def get_samp_rate_ue2(self):
        return self.samp_rate_ue2

    def set_samp_rate_ue2(self, samp_rate_ue2):
        self.samp_rate_ue2 = samp_rate_ue2

    def get_samp_rate_ue1(self):
        return self.samp_rate_ue1

    def set_samp_rate_ue1(self, samp_rate_ue1):
        self.samp_rate_ue1 = samp_rate_ue1
        self.blocks_throttle_0.set_sample_rate(1.0*self.samp_rate_ue1/(1.0*self.slow_down_ratio))

    def get_org_samp_rate(self):
        return self.org_samp_rate

    def set_org_samp_rate(self, org_samp_rate):
        self.org_samp_rate = org_samp_rate
        self.blocks_throttle_0_0.set_sample_rate(1.0*self.org_samp_rate/(1.0*self.slow_down_ratio))

    def get_An_ch3(self):
        return self.An_ch3

    def set_An_ch3(self, An_ch3):
        self.An_ch3 = An_ch3
        self.analog_noise_source_x_0.set_amplitude(10**(self.An_ch3/20.0))

    def get_An_ch2(self):
        return self.An_ch2

    def set_An_ch2(self, An_ch2):
        self.An_ch2 = An_ch2
        self.analog_noise_source_x_0_0.set_amplitude(10**(self.An_ch2/20.0))

    def get_An_ch1(self):
        return self.An_ch1

    def set_An_ch1(self, An_ch1):
        self.An_ch1 = An_ch1
        self.analog_noise_source_x_0_0_0.set_amplitude(10**(self.An_ch1/20.0))

    def send_noise_amplitude_message(self, ue_nr, noise_amplitude): 
        noise_amplitude_message = NoiseAmplitude()
        noise_amplitude_message.ue_id = ue_nr
        noise_amplitude_message.noise_amplitude = noise_amplitude
        serialized_message = noise_amplitude_message.SerializeToString()

        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            try:
                host = 'localhost'
                port = 11155
                socket_address = (host, port)
                s.sendto(serialized_message, socket_address)
            except Exception as e:
                print("Error sending serialized message:", str(e))  

    def set_an(self, ue_nr, an_value) :
        setattr(self, f"an_ch{ue_nr}_value", an_value)
        analog_noise_source_ul = getattr(self, f"analog_noise_source_ue_ul_{ue_nr}")
        analog_noise_source_dl = getattr(self, f"analog_noise_source_ue_dl_{ue_nr}")
        #print("ANALOG NOISE SOURCE OBJECT => ", analog_noise_source)
        analog_noise_source_ul.set_noise_voltage(10**(an_values[ue_nr]/20.0))
        analog_noise_source_dl.set_noise_voltage(10**(an_values[ue_nr]/20.0))
        #analog_noise_source.set_noise_voltage(10**(self.an_value/20.0))
        print(f"An {ue_nr} updated => {an_value} dB")
        self.send_noise_amplitude_message(ue_nr, an_value)
    
    def set_all_an(self, an_values):
        print(f"An VALUES => [{an_values}]")
        for ue_nr in range(0, NUMBER_OF_UES):
            setattr(self, f"an_ch{ue_nr+1}_value", an_values[ue_nr])
            analog_noise_source_ul = getattr(self, f"analog_noise_source_ue_ul_{ue_nr+1}")
            analog_noise_source_dl = getattr(self, f"analog_noise_source_ue_dl_{ue_nr+1}")
            #print("ANALOG NOISE SOURCE OBJECT => ", analog_noise_source)
            #analog_noise_source.set_noise_voltage(10**(an_values[ue_nr]/20.0))
            
            analog_noise_source_ul.set_noise_voltage(10**(an_values[ue_nr]/20.0))
            analog_noise_source_dl.set_noise_voltage(10**(an_values[ue_nr]/20.0))
            print(f"An {ue_nr+1} updated => {an_values[ue_nr]} dB")
            self.send_noise_amplitude_message(ue_nr, an_values[ue_nr])

class noise_thread(threading.Thread): 
    def __init__(self, multi_ue_scenario, number_of_ues): 
        threading.Thread.__init__(self) 
        self.number_of_ues = number_of_ues 
        self.min_an = -28.0
        self.max_an = -16.0
        if RANDOM_AN_VALUES:
            self.time_sleep = 60 #60 or 120
        else:
            #self.time_sleep = 120
            self.time_sleep = 30
        self.mean = 100
        self.stddev = 3
        if NUMBER_OF_UES == 1:
            self.fixed_values = np.arange(-30, -15.9, 0.5)
            #self.fixed_values = np.arange(30, 50, 0.5) => se for so uplink
        else:
            self.fixed_values = [-28.0, -26.0, -24.0, -22.0, -20.0, -18.0, -17.8, -17.6, -17.4, -17.2]
        self.pos_fixed = 0

        self.multi_ue_scenario = multi_ue_scenario

    def run(self): 
        #sleep(self.time_sleep) ## Time to connect everything -- with an = -100
        sleep(1)
        self.multi_ue_scenario.set_all_an( np.full(NUMBER_OF_UES, -20.0))
        sleep(self.time_sleep)
        while True:
            if CHANGE_ALL:
                if RANDOM_AN_VALUES:
                    an_values = [round(np.random.uniform(low=self.min_an, high=self.max_an), 1) for _ in range(NUMBER_OF_UES)]
                else:
                    an_values = np.full(NUMBER_OF_UES, self.fixed_values[self.pos_fixed])
                    self.pos_fixed = self.pos_fixed + 1
                self.multi_ue_scenario.set_all_an(an_values)
                sleep(self.time_sleep)
            else:
                if RANDOM_AN_VALUES:
                    new_an = round(np.random.uniform(low=MIN_AN, high=MAX_AN), 2)
                else:
                    new_an = self.fixed_values[self.pos_fixed]
                    self.pos_fixed = self.pos_fixed + 1
                self.multi_ue_scenario.set_an(random.randint(1, self.number_of_ues), new_an)
                sleep(self.time_sleep)



def main(top_block_cls=multi_ue_scenario, options=None):
    if StrictVersion("4.5.0") <= StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
        style = gr.prefs().get_string('qtgui', 'style', 'raster')
        Qt.QApplication.setGraphicsSystem(style)
    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()

    tb.start()

    t_noise = noise_thread(tb, NUMBER_OF_UES)
    t_noise.run()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    qapp.exec_()
if __name__ == '__main__':
    main()
