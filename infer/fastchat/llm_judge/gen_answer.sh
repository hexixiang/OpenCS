#!/bin/bash

commands=(
"CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python gen_model_answer.py --model-path /path/to/model --model-id model_name --dtype bfloat16 --max-new-token 512 --save-path /path/to/save/answer"
)

for i in "${!commands[@]}"; do
    if ! eval "${commands[$i]}"; then
        echo "Command on line $((i+1)) failed: ${commands[$i]}"
    fi
done
