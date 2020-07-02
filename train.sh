#!/bin/sh

#SBATCH -J ResNet50_P4M_Mixup_on_CIFAR
#SBATCH -o ResNet50_P4M_Mixup_on_CIFAR.%j.out
#SBATCH -p gpu-titanxp
#SBATCH -t 48:00:00

#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=1

cd  $SLURM_SUBMIT_DIR

echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
echo "CUDA_HOME=$CUDA_HOME"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "CUDA_VERSION=$CUDA_VERSION"

srun -l /bin/hostname
srun -l /bin/pwd
srun -l /bin/date

python3 train.py --model ResNet50 --dataset CIFAR --checkpoint ResNet50_P4M_Mixup_on_CIFAR --mixup

date

squeue  --job  $SLURM_JOBID

echo  "##### END #####"
