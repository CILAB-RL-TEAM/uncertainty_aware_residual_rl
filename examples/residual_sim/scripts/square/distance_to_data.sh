#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYOPENGL_PLATFORM=egl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
conda activate uarl
cd ~/uncertainty_aware_residual_rl/examples/residual_sim
seed=(0 10 20)
for i in {0..2}
do
    python sync_residual_diffusion.py "$@" \
        --env RobomimicState \
        --env_name NutAssemblySquare \
        --max_traj_length 400 \
        --eval_n_trajs 20 \
        --burn_in_steps 10000 \
        --batch_size 256 \
        --eval_period 50000 \
        --max_steps 2_000_000 \
        --replay_buffer_capacity 1000000 \
        --robomimic_checkpoint_path ~/uncertainty_aware_residual_rl/base_policies/square/square_robomimic.pth \
        --normalization_path ~/uncertainty_aware_residual_rl/base_policies/square/normalization.npz \
        --data_file  ~/uncertainty_aware_residual_rl/base_policies/square/train.npz \
        --diffuision_config_path ~/uncertainty_aware_residual_rl/examples/cfg/square_robomimic.yaml \
        --exp_name square_diffusion_distance_to_data_seed_${seed[$i]} \
        --wandb_group_name square_diffusion_distance_to_data \
        --seed ${seed[$i]} \
        --uncertainty_threshold=4.5e-5 \
        --uncertainty_decay_rate=1000000 \
        --checkpoint_path ~/test_checkpoints/square_diffusion_distance_to_data_seed_${seed[$i]} &> ~/test_checkpoints/logs/learner_square_diffusion_distance_to_data_seed_${seed[$i]}.out &
done
wait