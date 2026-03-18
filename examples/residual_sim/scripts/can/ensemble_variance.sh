#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYOPENGL_PLATFORM=egl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
conda activate uarl
cd ~/uncertainty_aware_residual_rl/examples/residual_sim
seed=(0 10 20 30 40)
for i in {0..1}
do
    python sync_residual_diffusion_ensemble.py "$@" \
        --env RobomimicState \
        --env_name PickPlaceCan \
        --learner_update 1 \
        --utd_ratio 4 \
        --max_traj_length 300 \
        --eval_n_trajs 20 \
        --burn_in_steps 10000 \
        --batch_size 64 \
        --eval_period 50000 \
        --max_steps 1_000_000 \
        --replay_buffer_capacity 1000000 \
        --robomimic_checkpoint_path ~/uncertainty_aware_residual_rl/base_policies/can/can_robomimic.pth \
        --normalization_path ~/uncertainty_aware_residual_rl/base_policies/can/normalization.npz \
        --diffusion_config_dir_path ~/uncertainty_aware_residual_rl/examples/cfg/ensemble/can \
        --exp_name can_diffusion_ensemble_seed_${seed[$i]} \
        --wandb_group_name can_diffusion_ensemble \
        --seed ${seed[$i]} \
        --uncertainty_threshold=0.2 \
        --uncertainty_decay_rate=500000 \
        --checkpoint_path ~/test_checkpoints/can_diffusion_ensemble_seed_${seed[$i]} &> ~/test_checkpoints/logs/learner_can_diffusion_ensemble_seed_${seed[$i]}.out &
done
wait