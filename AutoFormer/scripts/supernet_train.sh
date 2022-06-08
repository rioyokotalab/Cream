#!/bin/bash
#$ -cwd
#$ -l rt_F=32
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

export NGPUS=128
export NPERNODE=4
mpirun -npernode $NPERNODE -np $NGPUS \
python supernet_train.py \
    --data-path /groups/gcd50691/datasets/ImageNet --gp \
    --change-qkv --mode super --dist-eval --relative-position \
    --cfg ./experiments/supernet/supernet-B.yaml --epochs 500 --warmup-epochs 20 \
    --output-dir /groups/gcc50533/acc12016yi/AutoFormer/output --batch-size 32 \
    --lr 1e-3 --scaled-lr \
    --hold-epoch 50 --ckp-hist 10 \
    --log-wandb --experiment supernet_train_base_imnet1k_scratch_lr1e-3 \
    --resume /groups/gcc50533/acc12016yi/AutoFormer/output/supernet_train_base_imnet1k_scratch_lr1e-3/last.pth \
    --resume-mode resume_train 
    # --eval-fixed-model \