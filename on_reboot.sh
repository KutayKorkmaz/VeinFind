#!/bin/bash

export DISPLAY=:0 && xhost +localhost
cd /home/pi/Desktop/VeinFind
python threaddeneme.py
