#!/bin/bash
#$ -cwd
#$ -l rt_F=32
#$ -l h_rt=72:00:00
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

export NGPUS=128
export NPERNODE=4
mpirun -npernode $NPERNODE -np $NGPUS \
python supernet_train.py \
    --data-path /groups/gcd50691/datasets/ImageNet21k/train \
    --data-set others --nb_classes 21841 \
    --without_eval --gp \
    --change_qkv --mode super --relative_position \
    --cfg ./experiments/supernet/supernet-S.yaml --epochs 120 --warmup-epochs 10 \
    --lr 4e-3 --scaled-lr \
    --output_dir /groups/gcc50533/acc12016yi/AutoFormer/output --batch-size 64 \
    --hold_epoch 20 \
    --log_wandb --experiment supernet_pretrain_small_imnet21k --group pretrain
    