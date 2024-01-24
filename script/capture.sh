#!/bin/bash

cd /home/pi/workspace/parking_v1
echo "cam.capture script"
libcamera-jpeg -o /home/pi/workspace/parking_v1/image/raw.jpg --rotation 180 --width 1920 --height 1080 -n -t100
