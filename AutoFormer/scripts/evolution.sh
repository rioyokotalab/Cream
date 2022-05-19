#!/bin/bash
#$ -cwd
#$ -l rt_F=8
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

export NGPUS=32
export NPERNODE=4
mpirun -npernode $NPERNODE -np $NGPUS \
python evolution.py \
    --data-path /groups/gcc50533/imnet/ILSVRC2012 --gp \
    --change_qkv --relative_position --dist-eval \
    --cfg ./experiments/supernet/supernet-S.yaml \
    --resume /groups/gcc50533/acc12016yi/AutoFormer/output/supernet_train_small_imnet1k_from_supernet_imnet21k/last.pth \
    --min-param-limits 20 --param-limits 23 --data-set EVO_IMNET \
    --output_dir /groups/gcc50533/acc12016yi/AutoFormer/output \
    --log_wandb --experiment evolution_small_imnet1k_from_imnet21k_20to23
#     --load_cp /groups/gcc50533/acc12016yi/AutoFormer/output/evolution_tiny_imnet1k_bs4096_3to6/checkpoint-16.pth.tar
