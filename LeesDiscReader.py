#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 11:57:14 2019

@author: pranavaddepalli
"""

import serial

serial_port = '/dev/cu.usbmodem14101';
baud_rate = 9600; #In arduino, Serial.begin(baud_rate)

output_file = open("Lees20Lines.txt", "w+");
ser = serial.Serial(serial_port, baud_rate)

while True:
    line = ser.readline();
    line = line.decode("utf-8") #ser.readline returns a binary, convert to string
    print(line);
    output_file.write(line);
    
    