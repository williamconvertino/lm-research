#!/bin/bash

# Optional arguments [MODEL_NAME] [NUM_GPUS] [NUM_NODES]
NUM_GPUS=${1:-4}                # default 4 GPUs per node
NUM_NODES=${2:-1}               # default 1 node

# Clear first 2 args to avoid hydra issues
shift 2

# CPUs and memory automatically determined 
CPUS_PER_TASK=8                             # 8 CPUs per GPU
TOTAL_CPUS=$((NUM_GPUS * CPUS_PER_TASK))
TOTAL_MEM=$((NUM_GPUS * 32))                # 32gb ram per GPU


# Extract model name from hydra overrides
MODEL_NAME="transformer"
for arg in "$@"; do
  if [[ "$arg" == model=* ]]; then
    MODEL_NAME="${arg#model=}"
    break
  fi
done

JOB_NAME="${MODEL_NAME}_train_1_3b"

echo "Launching training job: $JOB_NAME"
echo "Nodes: ${NUM_NODES}, GPUs per node: ${NUM_GPUS}"
echo "Resources per node: ${TOTAL_CPUS} CPUs, ${TOTAL_MEM} GB memory"

SBATCH_ARGS="--job-name=${JOB_NAME} \
             --nodes=${NUM_NODES} \
             --gres=gpu:h200:${NUM_GPUS} \
             --ntasks-per-node=${NUM_GPUS} \
             --cpus-per-task=${CPUS_PER_TASK} \
             --mem=${TOTAL_MEM}G \
             --output=../../logs/${MODEL_NAME}/%x-%j.out \
             --error=../../logs/${MODEL_NAME}/%x-%j.err"

sbatch $SBATCH_ARGS ../jobs/run_job_h200.sh \
    size=1_3b \
    training=1_3b \
    dataset=slimpajama_1m \
    "$@"