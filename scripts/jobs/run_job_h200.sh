#!/bin/bash
#SBATCH --job-name=transformer_1_3b
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --exclude=dcc-h200-gpu-04,dcc-h200-gpu-05
#SBATCH --nodelist=dcc-h200-gpu-01,dcc-h200-gpu-02,dcc-h200-gpu-03,dcc-h200-gpu-06,dcc-h200-gpu-07
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=h200ea
#SBATCH --output=../../logs/%x-%j.out
#SBATCH --error=../../logs/%x-%j.err
#SBATCH --account=h200ea

cd ../../

if [ -f "./scripts/env/env.sh" ]; then
    source ./scripts/env/env.sh
fi

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
RANDOM_OFFSET=$((100 + RANDOM % 100))
export MASTER_PORT=$((29500 + RANDOM_OFFSET))

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_NTASKS_PER_NODE: $SLURM_NTASKS_PER_NODE"
echo "SLURM_PROCID: $SLURM_PROCID"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc_per_node=$SLURM_NTASKS_PER_NODE \
  --node_rank=$SLURM_PROCID \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  main.py "$@"
