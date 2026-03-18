#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="1"
export NVIDIA_VISIBLE_DEVICES="1"
export MUJOCO_EGL_DEVICE_ID="1"
export NV_GPU="1"
export EGL_DEVICE_ID="1"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYOPENGL_PLATFORM=egl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
conda activate uarl
cd ~/uncertainty_aware_residual_rl/examples/residual_sim
seed=(0 10 20 30 40)
for i in {0..4}
do
    python sync_residual_diffusion.py "$@" \
        --env RobomimicState \
        --env_name PickPlaceCan \
        --max_traj_length 300 \
        --eval_n_trajs 20 \
        --burn_in_steps 10000 \
        --batch_size 64 \
        --eval_period 50000 \
        --max_steps 1_000_000 \
        --replay_buffer_capacity 1000000 \
        --robomimic_checkpoint_path ~/uncertainty_aware_residual_rl/base_policies/can/can_robomimic.pth \
        --normalization_path ~/uncertainty_aware_residual_rl/base_policies/can/normalization.npz \
        --data_file  ~/uncertainty_aware_residual_rl/base_policies/can/train.npz \
        --diffuision_config_path ~/uncertainty_aware_residual_rl/examples/cfg/can_robomimic.yaml \
        --exp_name can_diffusion_distance_to_data_${seed[$i]} \
        --wandb_group_name can_diffusion_distance_to_data \
        --seed ${seed[$i]} \
        --uncertainty_threshold=4.5e-5 \
        --uncertainty_decay_rate=400000 \
        --checkpoint_path ~/test_checkpoints/can_diffusion_distance_to_data_${seed[$i]} &> ~/test_checkpoints/logs/can_diffusion_distance_to_data_${seed[$i]}.out &
done
wait