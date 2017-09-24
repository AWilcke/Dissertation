#!/bin/bash

docker build -t "pytorch" .
docker run \
    -v ~/Documents/Coursework/Dissertation/:/home \
    -it pytorch
