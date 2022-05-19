import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import json
import yaml
import os
import wandb
import re
import collections
from pathlib import Path
from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler
from lib.datasets import build_dataset
from supernet_engine import train_one_epoch, evaluate
from lib.samplers import RASampler
from lib import utils
from lib.config import cfg, update_config_from_file
from model.supernet_transformer import Vision_TransformerSuper


def get_args_parser():
    parser = argparse.ArgumentParser('AutoFormer training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    # config file
    parser.add_argument('--cfg',help='experiment configure file name',required=True,type=str)

    # custom parameters
    parser.add_argument('--platform', default='pai', type=str, choices=['itp', 'pai', 'aml'],
                        help='Name of model to train')
    parser.add_argument('--teacher_model', default='', type=str,
                        help='Name of teacher model to train')
    parser.add_argument('--relative_position', action='store_true')
    parser.add_argument('--gp', action='store_true')
    parser.add_argument('--change_qkv', action='store_true')
    parser.add_argument('--qkv_bias', action='store_true', default=True)
    parser.add_argument('--max_relative_position', type=int, default=14, help='max distance in relative position embedding')
    parser.add_argument('--without_eval', action='store_true', help='train without evaluation')
    parser.add_argument('--eval_fixed_model', action='store_true',
                        help='do evaluation on fixed model in addition, use EVAL in config file')

    # Model parameters
    parser.add_argument('--model', default='', type=str, metavar='MODEL',
                        help='Name of model to train')
    # AutoFormer config
    parser.add_argument('--mode', type=str, default='super', choices=['super', 'retrain'], help='mode of AutoFormer')
    parser.add_argument('--input-size', default=224, type=int)
    parser.add_argument('--patch_size', default=16, type=int)

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                        help='Drop block rate (default: None)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    # parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')
    parser.add_argument('--rpe_type', type=str, default='bias', choices=['bias', 'direct'])
    parser.add_argument('--post_norm', action='store_true')
    parser.add_argument('--no_abs_pos', action='store_true')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--scaled-lr', action='store_true',
                        help='set a scaled lr manually, and skip lr scaling in the program')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--lr-power', type=float, default=1.0,
                        help='power of the polynomial lr scheduler')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')


    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Dataset parameters
    parser.add_argument('--data-path', default='./data/imagenet/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19', 'others'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--nb_classes', default=None, type=int,
                        help='set class numbers (only supported when choose others data-set)')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='./',
                        help='path where to save, empty for no saving')
    parser.add_argument('--ckp_hist', type=int, default=10,
                        help='number of checkpoints to keep')
    parser.add_argument('--hold_epoch', type=int, default=None,
                        help='keep checkpoints every x epochs')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--no_resume_opt', action='store_true',
                        help='prevent resume of optimizer state when resuming model')
    parser.add_argument('--resume_mode', type=str, choices=['resume_train', 'load_timm_pretrain', 'load_pretrain_diff_key'],
                        help='change method loading checkpoint')
    parser.add_argument('--start_epoch', default=None, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--pause_epoch', default=None, type=int,
                        help='pause learning at the epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str, help='distributed backend')

    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--no-amp', action='store_false', dest='amp')
    parser.set_defaults(amp=True)

    # wandb parameters
    parser.add_argument('--log_wandb', action='store_true')
    parser.add_argument('--project_name', default='AutoFormer', type=str)
    parser.add_argument('--experiment', default='', type=str)
    parser.add_argument('--group', default='supernet_train', type=str)

    return parser


def main(args):

    args.distributed = int(os.getenv('OMPI_COMM_WORLD_SIZE', '1')) > 1
    args.local_rank = 0
    args.world_size = 1
    args.rank = 0  # global rank
    if args.distributed:
        # initialize torch.distributed using MPI
        master_addr = os.getenv("MASTER_ADDR", default="localhost")
        master_port = os.getenv('MASTER_PORT', default='8888')
        method = "tcp://{}:{}".format(master_addr, master_port)
        rank = int(os.getenv('OMPI_COMM_WORLD_RANK', '0'))  # global rank
        world_size = int(os.getenv('OMPI_COMM_WORLD_SIZE', '1'))

        ngpus_per_node = torch.cuda.device_count()
        node = rank // ngpus_per_node
        args.local_rank = rank % ngpus_per_node
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend=args.dist_backend, init_method=method, world_size=world_size,
                                             rank=rank)
        args.rank = rank
        args.world_size = world_size
        print('Training in distributed mode with multiple processes, 1 GPU per process. Process %d:%d, total %d.'
              % (args.local_rank, node, args.world_size))
    else:
        print('Training with a single process on 1 GPUs.')
    assert args.rank >= 0

    # utils.init_distributed_mode(args)
    update_config_from_file(args.cfg)

    if args.rank == 0:
        print(args)
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)

    device = torch.device(args.local_rank)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)
    cudnn.benchmark = True

    if args.log_wandb and args.rank == 0:
        wandb.init(project=args.project_name, entity='yokota-vit', name=args.experiment, group=args.group, config=args)

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    if not args.without_eval:
        dataset_val, _ = build_dataset(is_train=False, args=args)
    if args.rank == 0:
        print(f'dataset loaded from {args.data_path}, with {args.nb_classes} classes')

    if args.distributed:
        num_tasks = world_size
        global_rank = dist.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if not args.without_eval:
            if args.dist_eval:
                if args.rank == 0 and len(dataset_val) % num_tasks != 0:
                    print(
                        'Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                        'This will slightly alter validation results as extra duplicate entries are added to achieve '
                        'equal num of samples per-process.')
                sampler_val = torch.utils.data.DistributedSampler(
                    dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
            else:
                sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        if not args.without_eval:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    if not args.without_eval:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, batch_size=int(2 * args.batch_size),
            sampler=sampler_val, num_workers=args.num_workers,
            pin_memory=args.pin_mem, drop_last=False
        )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    if args.rank == 0:
        print(f"Creating SuperVisionTransformer")
        print(cfg)
    # default set qkv_bias to True!
    # set to False once for debug
    model = Vision_TransformerSuper(img_size=args.input_size,
                                    patch_size=args.patch_size,
                                    embed_dim=cfg.SUPERNET.EMBED_DIM, depth=cfg.SUPERNET.DEPTH,
                                    num_heads=cfg.SUPERNET.NUM_HEADS,mlp_ratio=cfg.SUPERNET.MLP_RATIO,
                                    qkv_bias=args.qkv_bias, drop_rate=args.drop,
                                    drop_path_rate=args.drop_path,
                                    gp=args.gp,
                                    num_classes=args.nb_classes,
                                    max_relative_position=args.max_relative_position,
                                    relative_position=args.relative_position,
                                    change_qkv=args.change_qkv, abs_pos=not args.no_abs_pos)

    choices = {'num_heads': cfg.SEARCH_SPACE.NUM_HEADS, 'mlp_ratio': cfg.SEARCH_SPACE.MLP_RATIO,
               'embed_dim': cfg.SEARCH_SPACE.EMBED_DIM , 'depth': cfg.SEARCH_SPACE.DEPTH}

    model.to(device)
    if args.teacher_model:
        teacher_model = create_model(
            args.teacher_model,
            pretrained=True,
            num_classes=args.nb_classes,
        )
        teacher_model.to(device)
        teacher_loss = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        teacher_model = None
        teacher_loss = None

    model_ema = None

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if args.rank == 0:
        print('number of params:', n_parameters)

    if not args.scaled_lr:
        linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
        args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()
    lr_scheduler, _ = create_scheduler(args, optimizer)

    # criterion = LabelSmoothingCrossEntropy()

    if args.mixup > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    output_dir = Path(f'{args.output_dir}/{args.experiment}')

    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    # save config for later experiments
    if args.rank == 0:
        with open(output_dir / "config.yaml", 'w') as f:
            f.write(args_text)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')

        if args.resume_mode == 'load_pretrain_diff_key':
            ###### deal with key name ######
            model_ckp = checkpoint['state_dict']
            new_model_ckp = collections.OrderedDict()
            for k in model_ckp:
                if re.match('head', k):
                    continue
                elif re.match('patch_embed', k):
                    k1 = k.replace('patch_embed', 'patch_embed_super')
                elif re.match('blocks', k):
                    if re.search('norm1', k):
                        k1 = k.replace('norm1', 'attn_layer_norm')
                    elif re.search('norm2', k):
                        k1 = k.replace('norm2', 'ffn_layer_norm')
                    elif re.search('mlp', k):
                        k1 = k.replace('mlp.fc', 'fc')
                    else:
                        k1 = k
                else:
                    k1 = k
                new_model_ckp[k1] = model_ckp[k]
            ###### ------------------ ######
            model_without_ddp.load_state_dict(new_model_ckp, strict=False)
        elif args.resume_mode == 'load_timm_pretrain':
            model_ckp = checkpoint['state_dict']
            if args.nb_classes != model_ckp['head.weight'].shape[0]:
                del model_ckp['head.weight']
                del model_ckp['head.bias']
            model_without_ddp.load_state_dict(model_ckp, strict=False)
        elif args.resume_mode == 'resume_train':
            model_ckp = checkpoint['model']
            if args.nb_classes != model_ckp['head.weight'].shape[0]:
                del model_ckp['head.weight']
                del model_ckp['head.bias']
            model_without_ddp.load_state_dict(model_ckp, strict=False)

        if not args.eval and not args.no_resume_opt and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            if args.start_epoch is None:
                args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
        if args.rank == 0:
            print(f'Load checkpoint from {args.resume}')
            print(f'Start at epoch {args.start_epoch}')

    retrain_config = None
    eval_config = None
    if args.mode == 'retrain' and "RETRAIN" in cfg:
        retrain_config = {'layer_num': cfg.RETRAIN.DEPTH, 'embed_dim': [cfg.RETRAIN.EMBED_DIM]*cfg.RETRAIN.DEPTH,
                          'num_heads': cfg.RETRAIN.NUM_HEADS,'mlp_ratio': cfg.RETRAIN.MLP_RATIO}
    if args.eval:
        test_stats = evaluate(data_loader_val, model, device,  mode = args.mode, retrain_config=retrain_config, rank=args.rank)
        if args.rank == 0:
            print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return

    if args.rank == 0:
        print("Start training")
    start_time = time.time()

    if not args.without_eval:
        # use test acc1 as metric
        best_metric = 0.0
    else:
        # use train loss as metric
        best_metric = 10**5
    best_metric_epoch = -1

    if args.start_epoch is None:
        args.start_epoch = 0
    
    if args.output_dir:
        # for saving checkpoints
        ckp_list = []
        buf_best_ckp = -1
        last_path = f'{output_dir}/last.pth'
        best_path = f'{output_dir}/best.pth'

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, model_ema, mixup_fn,
            amp=args.amp, teacher_model=teacher_model,
            teach_loss=teacher_loss,
            choices=choices, mode = args.mode, retrain_config=retrain_config, rank=args.rank,
        )

        lr_scheduler.step(epoch)

        if not args.without_eval:
            test_stats = evaluate(data_loader_val, model, device, amp=args.amp, choices=choices, mode=args.mode, retrain_config=retrain_config, rank=args.rank)
            if args.rank == 0:
                print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            if args.eval_fixed_model:
                test_fixed_stats = evaluate(data_loader_val, model, device, amp=args.amp, choices=choices, mode='retrain', retrain_config=eval_config, rank=args.rank)
                if args.rank == 0:
                    print(f"Accuracy of the fixed network: {test_fixed_stats['acc1']:.1f}%")
            best_metric = max(best_metric, test_stats["acc1"])
            if best_metric == test_stats["acc1"]:
                best_metric_epoch = epoch
            if args.rank == 0:
                print(f'Max accuracy: {best_metric:.2f}%, epoch: {best_metric_epoch}')
        else:
            best_metric = min(best_metric, train_stats['loss'])
            if best_metric == train_stats['loss']:
                best_metric_epoch = epoch
            if args.rank == 0:
                print(f'Min train loss: {best_metric:.4f}, epoch: {best_metric_epoch}')

        # save and cleanup checkpoints
        if args.output_dir and args.rank == 0:
            ckp_list.append(epoch)
            checkpoint_path = f'{output_dir}/checkpoint-{epoch:03}.pth'
            if os.path.exists(last_path):
                os.unlink(last_path)
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                # 'model_ema': get_state_dict(model_ema),
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }, checkpoint_path)
            os.link(checkpoint_path, last_path)
            if epoch == best_metric_epoch:
                if os.path.exists(best_path):
                    os.unlink(best_path)
                os.link(checkpoint_path, best_path)
                if buf_best_ckp != -1:
                    ckp_del = f'{output_dir}/checkpoint-{buf_best_ckp:03}.pth'
                    os.remove(ckp_del)
                    buf_best_ckp = -1
            if len(ckp_list) > args.ckp_hist:
                to_del = ckp_list.pop(0)
                if to_del == best_metric_epoch:
                    buf_best_ckp = to_del
                elif not args.hold_epoch or to_del % args.hold_epoch != 0:
                    ckp_del = f'{output_dir}/checkpoint-{to_del:03}.pth'
                    os.remove(ckp_del)

        if not args.without_eval:
            if args.eval_fixed_model:
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'test_{k}': v for k, v in test_stats.items()},
                             **{f'test_fixed_{k}': v for k, v in test_fixed_stats.items()},
                             'epoch': epoch,
                             'n_parameters': n_parameters}
            else:
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'test_{k}': v for k, v in test_stats.items()},
                             'epoch': epoch,
                             'n_parameters': n_parameters}
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}
        if args.log_wandb and args.rank == 0:
            wandb.log(log_stats)

        if args.output_dir and args.rank == 0:
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        if args.pause_epoch is not None and epoch == args.pause_epoch:
            if args.rank == 0:
                print(f'Pause training at epoch {epoch}')
            break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if args.rank == 0:
        print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('AutoFormer training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
