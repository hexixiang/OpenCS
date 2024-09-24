#!/bin/bash

docker build -f ./docker/docker-cuda/Dockerfile \
    -t llamafactory:latest .