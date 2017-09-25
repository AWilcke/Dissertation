#!/bin/bash

nvidia-docker build -t "pytorch" .
nvidia-docker run \
    --device /dev/nvidia0:/dev/nvidia0 \
    --device /dev/nvidiactl:/dev/nvidiactl \
    --device /dev/nvidia-modeset:/dev/nvidia-modeset \
    --device /dev/nvidia-uvm:/dev/nvidia-uvm \
    --device /dev/nvidia-uvm-tools:/dev/nvidia-uvm-tools \
    -v ~/Documents/Coursework/Dissertation/:/home \
    -it pytorch
