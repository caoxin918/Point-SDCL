from pathlib import Path
import argparse
import json
import math
import os
import sys
import time
import wandb
import numpy as np
import torch
from torch import nn
import torch.optim as opt
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import utils
import builtins
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from data.modelnet40 import ModelNet40
from model.pointnet import PointNet
from model.pct import pctModel
from model.dgcnn import DGCNN
from model.PointTransformer import PointTransformer
import torch.backends.cudnn as cudnn
def get_arguments():
    parser = argparse.ArgumentParser(description="Pretrain a pointnet model with HOME", add_help=False)

    # Running
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')

    # Distributed
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--rank', default=0, type=int)
    parser.add_argument('--dist-url', default='tcp://localhost:10001',
                        help='url used to set up distributed training')

    # Data
    # parser.add_argument("--data-dir", type=Path, default="./datasets/cifar10",
    #                     help='Path to the image net dataset')

    # Checkpoints
    parser.add_argument("--exp-dir", type=Path, default="../exp/KDCL_mn40_L8_outdim8192_e0.9998_b16_epo200_pt_optadamw_lr1.25e-4_teaTemp0.07_CosLR_v0",
                        help='Path to the experiment folder, where all logs/checkpoints will be stored')
    parser.add_argument("--log-freq-time", type=int, default=100,
                        help='Print logs to the stats.txt file every [log-freq-time] seconds')

    # Optim
    parser.add_argument("--epochs", type=int, default=250,
                        help='Number of epochs')
    parser.add_argument("--batch-size", type=int, default=12,
                        help='Effective batch size (per worker batch size is [batch-size] / world-size)')
    parser.add_argument("--base-lr", type=float, default=0.000125,
                        help='Base learning rate, effective learning after warmup is [base-lr] * [batch-size] / 256')
    parser.add_argument("--wd", type=float, default=1e-6,
                        help='Weight decay')
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--min_lr', type=float, default=1e-7, help="""Target LR at the
           end of optimization. We use a cosine LR schedule with linear warmup.""")

    # Model
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.07, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=10, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')
    parser.add_argument('--momentum_teacher', default=0.9999, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--out_dim', default=8192, type=int, help="""Dimensionality of
        the head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--embed_dim', default=768, type=int, help="""Dimensionality of
        the head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument('--clip_grad', type=float, default=1.0, help="""Maximal parameter
         gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
         help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--trans_dim', default=384, type=int,
                        help='classifier base learning rate')
    parser.add_argument('--depth', default=12, type=int,
                        help='depth')
    parser.add_argument('--drop_path_rate', default=0.1, type=float,
                        help='print frequency')
    parser.add_argument('--cls_dim', default=40, type=int,
                        metavar='DIR', help='path to checkpoint directory')
    parser.add_argument('--num_heads', default=6, type=int,
                        help='classifier base learning rate')
    parser.add_argument('--group_size', default=32, type=int,
                        help='depth')
    parser.add_argument('--num_group', default=64, type=float,
                        help='print frequency')
    parser.add_argument('--encoder_dims', default=384, type=int,
                        metavar='DIR', help='path to checkpoint directory')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--batch_size_per_gpu', default=8, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--saveckp_freq', default=5, type=int, help='Save checkpoint every x epochs.')
    return parser



def main(args):
    args.distributed = True

    ngpus_per_node = torch.cuda.device_count()
    print(ngpus_per_node)
    # Since we have ngpus_per_node processes per node, the total world_size
    # needs to be adjusted accordingly
    args.world_size = ngpus_per_node * args.world_size
    # Use torch.multiprocessing.spawn to launch distributed processes: the
    # main_worker process function
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    # main_worker()

def main_worker(args):
    # wandb.init(project="Train_home_v3", name="exp_v27")  # bu xu yao tiao zheng ,xia yi ci xun lian
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k,\
            v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True
    # if "SLURM_NODEID" in os.environ:
    #     args.rank = int(os.environ["SLURM_NODEID"])
    #
    # # suppress printing if not first GPU on each node
    # if args.gpu != 0 or args.rank != 0:
    #     def print_pass(*args):
    #         pass
    #
    #     builtins.print = print_pass
    #
    # if args.gpu is not None:
    #     print("Use GPU: {} for training".format(args.gpu))

    # For multiprocessing distributed training, rank needs to be the
    # global rank among all the processes
    # if "MASTER_PORT" in os.environ:
    #     args.dist_url = 'tcp://{}:{}'.format(args.dist_url, int(os.environ["MASTER_PORT"]))
    # print(args.dist_url)

    # print(args.rank, args.gpu)
    # args.rank = args.rank * ngpus_per_node + gpu
    # dist.init_process_group(backend='nccl', init_method=args.dist_url,
    #                         world_size=args.world_size, rank=args.rank)
    # torch.distributed.barrier()

    if args.rank == 0:
        args.exp_dir.mkdir(parents=True, exist_ok=True)
        stats_file = open(args.exp_dir / 'stats.txt', 'a', buffering=1)
        print(' '.join(sys.argv))
        print(' '.join(sys.argv), file=stats_file)
    print(args.rank,'rank')
    dataset = ModelNet40(args.num_points, partition='train', local=args.local_crops_number)

    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
    per_device_batch_size = int(args.batch_size / args.world_size)
    # print(args.batch_size, args.world_size, per_device_batch_size)
    loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    # single GPU
    # loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
    #                     pin_memory=True,shuffle=True)
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs,\
                                                       len(loader))
    print(momentum_schedule.shape)
    student = PointTransformer(drop_path_rate=args.drop_path_rate)
    student = utils.MultiCropWrapper(student, Head(
        args.embed_dim,
        args.out_dim,
    ))
    student = wrapper(student,args.out_dim,args.out_dim).cuda()
    teacher = PointTransformer()
    teacher = utils.MultiCropWrapper(teacher, Head(
        args.embed_dim,
        args.out_dim,
    )).cuda()
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher,\
                device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student,\
            device_ids=[args.gpu])
    # print(model)
    # single
    # teacher_without_ddp = teacher
    teacher_without_ddp.load_state_dict(student.module.model.state_dict())
    # wandb.watch(student)
    for p in teacher.parameters():
        p.requires_grad = False
    params_groups = utils.get_params_groups(student)
    optimizer = torch.optim.AdamW(params_groups)
    lr_schedule = utils.cosine_scheduler(
        args.base_lr ,  # linear scaling rule
        args.min_lr,
        args.epochs, len(loader),
        warmup_epochs=10,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(loader),
    )
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0, last_epoch=-1, verbose=True)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3,
    #                             momentum=0.9, weight_decay=5e-4)

    # if (args.exp_dir / "model.pth").is_file():
    #     if args.rank == 0:
    #         print("resuming from checkpoint")
    #     ckpt = torch.load(args.exp_dir / "model.pth", map_location="cpu")
    #     start_epoch = ckpt["epoch"]
    #     msg = student.load_state_dict(ckpt["model"])
    #     print(msg)
    #     optimizer.load_state_dict(ckpt["optimizer"])
    # else:

    start_epoch = 0
    aucm_loss = Loss(
        args.out_dim,
        args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).cuda()
    train_losses = AverageMeter()
    start_time = last_logging = time.time()
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)

        for step, ((y1), _) in enumerate(loader, start=epoch * len(loader)):
            metric_logger = utils.MetricLogger(delimiter="  ")
            # print(len(y1))
            # print(y1[0].shape)
            batch_size = y1[0].shape[0]
            # print(y1[0].shape)
            y1 = [im.cuda(non_blocking=True) for im in y1]
            for i, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = lr_schedule[step]
                if i == 0:  # only the first group is regularized
                    param_group["weight_decay"] = wd_schedule[step]
            teacher_output = teacher(y1[:2])  # only the 2 global views pass through the teacher
            student_output,student_mean,student_var = student(y1)
            loss,loss_auc,loss_pro = aucm_loss(student_output, teacher_output, epoch,student_mean,student_var)
            # lr = adjust_learning_rate(args, optimizer, loader, step)
            # lr_scheduler.step()

            optimizer.zero_grad()
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()

            with torch.no_grad():
                m = momentum_schedule[step]  # momentum parameter
                for param_q, param_k in zip(student.parameters(), \
                                            teacher_without_ddp.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

            train_losses.update(loss.item(), batch_size)

            current_time = time.time()

            if utils.is_main_process() and current_time - last_logging > args.log_freq_time:
                stats = dict(
                    epoch=epoch,
                    step=step,
                    loss=loss.item(),
                    loss_aucm = loss_auc.item(),
                    loss_prom = loss_pro.item(),
                    time=int(current_time - start_time),
                    # lr=lr,
                )
                print(json.dumps(stats))
                print(json.dumps(stats), file=stats_file)
                last_logging = current_time
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'loss': aucm_loss.state_dict(),
        }
        utils.save_on_master(save_dict, os.path.join(args.exp_dir, \
                                                     'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.exp_dir,\
                    f'checkpoint{epoch:04}.pth'))

    if utils.is_main_process():
        torch.save(save_dict, args.exp_dir / "model_final.pth")



def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.base_lr
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr

class Loss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9,l=2.4):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))
        self.out = out_dim
        self.dist = torch.distributions.Normal(loc=torch.zeros(self.out, device="cuda"),
                                               scale=torch.ones(self.out, device="cuda")
                                               )
        self.l = l
        self.n_sample = 4
    def forward(self, student_output, teacher_output, epoch, mean_st,var_st):
        """
        Cross-enropy between softmax outputs of the teacher and
        student networks.
        """
        self.batch = mean_st.shape[0]
        samples = self.dist.rsample(sample_shape=torch.tensor([self.n_sample, self.batch]))
        student_sample = (
                (samples * var_st.unsqueeze(0)) + mean_st.unsqueeze(0))
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out1 = (teacher_output - self.center)
        teacher_out1 = teacher_out1.detach().chunk(2)
        teacher_out2 = F.softmax((teacher_output - self.center) \
                / temp, dim=-1)

        teacher_out2 = teacher_out2.detach().chunk(2)
        #teacher_out = teacher_output.detach().chunk(2)

        total_loss = 0
        loss_auc = 0
        loss_pro = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out2):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                logits = torch.einsum('nc, mc->nm', [q, student_out[v]])
                logits = 0.5*(logits + 1)
                N = logits.shape[0]
                mask = torch.eye(N).to(logits.device)
                posvals = logits.masked_select(mask.bool())
                negvals = logits.masked_select(~mask.bool()).reshape(N, -1)
                p = 32/N

                #loss = torch.sum(-q * F.log_softmax(student_out[v],\
                #        dim=-1), dim=-1)
                loss = torch.mean((posvals - 10)**2 - 2*posvals) +\
                    p*torch.mean(torch.sum((negvals - 0)**2 +\
                    2*negvals, dim=-1)) + 1

                loss_auc += loss.mean()
                n_loss_terms += 1
        loss_auc /= n_loss_terms

        student_sample_chunk = student_sample.chunk(self.ncrops, dim=1)
        loss_second_part = 0
        loss_first_part = 0
        n_loss = 0

        for i in range(self.n_sample):
            for iq, q in enumerate(teacher_out1):
                for iv in range(len(student_sample_chunk)):
                    loss_first_part += 2 * F.l1_loss(student_sample_chunk[iv][i, :, :], q)
                    n_loss += 1

            if i != self.n_sample - 1:
                for j in range(i + 1, self.n_sample):
                    loss_second_part += - (1 / (self.n_sample - 1)) * F.l1_loss(student_sample[i, :, :],
                                                                                student_sample[j, :, :])

        loss_pro = (
                (loss_first_part + 0.000009) / n_loss + self.l * (loss_second_part + 0.000009) / self.n_sample)

        self.update_center(teacher_output)
        total_loss = loss_auc + loss_pro
        return total_loss,loss_auc,loss_pro

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) *\
                dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum +\
                batch_center * (1 - self.center_momentum)

class Head(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=False, nlayers=3, hidden_dim=2048, bottleneck_dim=512):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            utils.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x

class wrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """

    def __init__(self, model, out_dim, m_dim):
        super(wrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        self.model = model
        self.mean_layer = nn.Linear(m_dim, m_dim, bias=False)
        self.var_layer = nn.Linear(m_dim, m_dim, bias=False)
        self.pred_layer1 = nn.Linear(out_dim, m_dim)
        self.nonlinear = nn.GELU()
        self.pred_layer2 = nn.Linear(out_dim, m_dim)
        self.non_linear = nn.ReLU()


    def forward(self, x):
        proj = self.model(x)

        pred1 = self.pred_layer1(proj)

        pred1 = self.pred_layer2(self.nonlinear(pred1))
        pred = nn.functional.normalize(pred1, dim=-1, p=2)
        mu = self.mean_layer(self.non_linear(pred))
        var = torch.exp(self.var_layer(self.non_linear(pred)) * 0.5)
        return proj, mu, var


class PointHome(nn.Module):
    """
    feature encoder: PointNet
    projector: 3 layer mlp
    loss :loss(home) + loss(reg)
    """
    def __init__(self, args):
        super(PointHome, self).__init__()
        self.args = args
        # feature encoder
        self.backbone = PointNet()
        # self.proj = Head(in_dim=1024,out_dim=512)
        # self.backbone = DGCNN()
        # projector
        # sizes = [2048] + list(map(int, args.projector.split('-')))  # 2048-DGCNN 1024-pointnet
        sizes = [1024] + list(map(int, args.projector.split('-')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.proj= nn.Sequential(*layers)
        # normalization layer
    def forward(self, x1):
        z1 = self.proj(self.backbone(x1))




        return z1





def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.environ["SLURM_JOB_ID"]}')
    exit()


def handle_sigterm(signum, frame):
    pass

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == "__main__":
    parser = argparse.ArgumentParser('HOME training script', parents=[get_arguments()])
    args = parser.parse_args()
    main_worker(args)
