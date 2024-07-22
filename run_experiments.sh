#!/bin/bash

# Export the GPU device to use
export CUDA_VISIBLE_DEVICES=7

# List of datasets
datasets=("lego_8v" "materials_8v" "mic_8v" "ship_8v")

# Iterate over each dataset
for dataset in "${datasets[@]}"; do
    # Training step
    python run_nerf_uncertainty_NF.py \
        --config "configs/${dataset}.txt" \
        --expname "${dataset}" \
        --N_rand 512 \
        --N_samples 128 \
        --n_flows 4 \
        --h_alpha_size 64 \
        --h_rgb_size 64 \
        --K_samples 32 \
        --n_hidden 128 \
        --type_flows "triangular" \
        --beta1 0.01 \
        --depth_lambda 0.01 \
        --netdepth 8 \
        --netwidth 512 \
        --model "NeRF_Flows" \
        --index_step -1 \
        --is_train > "${dataset}_train.log" 2>&1
    
    # Rendering step
    python run_nerf_uncertainty_NF.py \
        --config "configs/${dataset}.txt" \
        --expname "${dataset}" \
        --N_rand 512 \
        --N_samples 128 \
        --n_flows 4 \
        --h_alpha_size 64 \
        --h_rgb_size 64 \
        --K_samples 32 \
        --n_hidden 128 \
        --type_flows "triangular" \
        --beta1 0.01 \
        --depth_lambda 0.01 \
        --netdepth 8 \
        --netwidth 512 \
        --model "NeRF_Flows" \
        --index_step -1 \
        --render_only \
        --is_train \
        --render_test > "${dataset}_render.log" 2>&1
done
