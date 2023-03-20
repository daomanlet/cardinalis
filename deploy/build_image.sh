#!/bin/bash
cp -rp ../app ./
sudo docker rmi -f rouynxia/cardinalis:latest
sudo docker build --no-cache --force-rm -t rouynxia/cardinalis:latest .
