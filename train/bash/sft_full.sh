#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 llamafactory-cli train examples/train_full/llama2-7b_full_sft_ds3.yaml