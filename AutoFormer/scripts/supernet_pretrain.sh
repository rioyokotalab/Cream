#!/bin/bash
#$ -cwd
#$ -l rt_F=64
#$ -l h_rt=24:00:00
#$ -j y
#$ -o output/o.$JOB_ID

# ======== Pyenv/ ========
export PYENV_ROOT=$HOME/.pyenv
export PATH=$PYENV_ROOT/bin:$PATH
eval "$(pyenv init -)"

# ======== Modules ========
source /etc/profile.d/modules.sh
module load openmpi cuda/10.2/10.2.89 cudnn nccl/2.7/2.7.8-1

export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep inet | cut -d " " -f 6 | cut -d "/" -f 1)

export NGPUS=256
export NPERNODE=4
mpirun -npernode $NPERNODE -np $NGPUS \
python supernet_train.py \
    --data-path /groups/gcd50691/datasets/ImageNet21k/train \
    --data-set others --nb-classes 21841 \
    --without-eval --gp \
    --change-qkv --mode super --relative-position \
    --cfg ./experiments/supernet/supernet-B.yaml --epochs 120 --warmup-epochs 10 \
    --lr 4e-3 --scaled-lr \
    --output-dir /groups/gcc50533/acc12016yi/AutoFormer/output --batch-size 32 \
    --hold-epoch 20 \
    --log-wandb --experiment supernet_pretrain_base_imnet21k --group pretrain \
    --resume /groups/gcc50533/acc12016yi/AutoFormer/output/supernet_pretrain_base_imnet21k/last.pth \
    --resume-mode resume_train --no-amp
    