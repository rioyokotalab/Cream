#!/bin/bash
#$ -cwd
#$ -l rt_F=8
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

export NGPUS=8
export NPERNODE=4
mpirun -npernode $NPERNODE -np $NGPUS \
python supernet_train.py \
    --data-path /groups/gcd50691/datasets/ImageNet --gp \
    --change_qkv --mode super --dist-eval --relative_position \
    --cfg ./experiments/supernet/supernet-S.yaml --epochs 500 --warmup-epochs 20 \
    --output_dir /groups/gcc50533/acc12016yi/AutoFormer/output --batch-size 128 \
    --lr 1e-3 --scaled-lr \
    --hold_epoch 50 --ckp_hist 10 \
    --log_wandb --experiment test_supernet_train_small_imnet1k_scratch_lr1e-3
    # --resume /groups/gcc50533/acc12016yi/AutoFormer/output/supernet_train_tiny_imnet1k_from_supernet_imnet21k_lr5e-4/last.pth \
    # --resume_mode resume_train 
    # --eval_fixed_model \