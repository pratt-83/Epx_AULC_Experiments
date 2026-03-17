#!/bin/bash
#SBATCH --cluster=ub-hpc
#SBATCH --partition=general-compute
#SBATCH --qos=general-compute
#SBATCH --gpus-per-node=a100-pcie-40gb:1
#SBATCH --constraint=AVX512
#SBATCH --account=vaalbert
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --mem=80G
#SBATCH --job-name=run_train_aulc
#SBATCH --output=run_train_aulc.out
#SBATCH --error=run_train_aulc.err
#SBATCH --mail-user=nbpratt@buffalo.edu
#SBATCH --mail-type=ALL

##SBATCH --requeue
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
cd $SLURM_SUBMIT_DIR

echo "working directory = "$SLURM_SUBMIT_DIR
ulimit -s unlimited

# Modules needed for torch + CUDA + matplotlib on CCR
module load gcc/11.2.0 openmpi/4.1.1
module load pytorch/1.13.1-CUDA-11.8.0
module load matplotlib/3.5.2

# Sanity checks: fail fast if the environment is not consistent
python - <<'PY'
import sys
print("python:", sys.executable)

import torch
print("torch:", torch.__version__, "cuda?", torch.cuda.is_available())
if torch.cuda.is_available():
	print("gpu:", torch.cuda.get_device_name(0))

import matplotlib
print("matplotlib:", matplotlib.__version__)
PY

mkdir -p logs metrics figures/20k

python train.py \
	--data_dir data/processed_20k \
	--seq_len 2048 \
	--steps 12000 \
	--batch_size 16 \
	--eval_every 200 \
	--eval_batches 50 \
	--log_csv logs/loss_curves_20k.csv \
	--save_ckpt

python calc_aulc.py \
	--csv logs/loss_curves_20k.csv \
	--out_json metrics/aulc_20k.json \
	--out_dir figures/20k
