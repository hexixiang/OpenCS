#!/bin/bash
# -v ./data:/app/data \
    # -v ./output:/app/output \
docker run -it --gpus=all \
    -v /data/hxx/model:/model \
    -v /home/hxx/LLaMA-Factory:/app \
    -p 7860:7860 \
    -p 8000:8000 \
    --shm-size 512G \
    --name llamafactory_py310 \
    llamafactory:latest

# pip uninstall flash-attn