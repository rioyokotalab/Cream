#!/bin/bash
#$ -cwd
#$ -l rt_F=8
#$ -l h_rt=32:00:00
#$ -j y
#$ -o output/o.$JOB_ID

# ======== Pyenv/ ========
export PYENV_ROOT=$HOME/.pyenv
export PATH=$PYENV_ROOT/bin:$PATH
eval "$(pyenv init -)"

# ======== Modules ========
source /etc/profile.d/modules.sh
module load cuda/10.2/10.2.89 cudnn openmpi nccl/2.7/2.7.8-1

export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep inet | cut -d " " -f 6 | cut -d "/" -f 1)

export NGPUS=32
export NPERNODE=4
mpirun -npernode $NPERNODE -np $NGPUS \
python supernet_train.py \
    --data-path /groups/gcd50691/datasets/ImageNet --gp \
    --change_qkv --relative_position --mode retrain --dist-eval \
    --cfg ./experiments/subnet/AutoFormer-5.8M-imnet21k.yaml --epochs 300 --warmup-epochs 5 \
    --output_dir /groups/gcc50533/acc12016yi/AutoFormer/output --batch-size 32 \
    --log_wandb --experiment retrain_tiny_imnet1k_from_imnet21k \
    --resume /groups/gcc50533/acc12016yi/AutoFormer/output/retrain_tiny_imnet1k_from_imnet21k/last.pth \
    --resume_mode resume_train
#     --resume /groups/gcc50533/acc12016yi/AutoFormer/output/pretrain_autoformer_tiny_imnet21k_qkvbias/model_best.pth.tar \
#     --no_resume_opt --start_epoch 0 --resume_mode load_timm_pretrain
