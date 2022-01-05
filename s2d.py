from __future__ import absolute_import
from __future__ import division
import argparse
import logging
import os
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from apex import amp

from config import cfg, assert_and_infer_cfg
from s2d_src.ae import Unet
from utils.eval_misc_fullSeg_noSkip_3class import AverageMeter, prep_experiment, evaluate_eval, fast_hist1, warpgrid
import datasets
import loss_fullSeg as loss
import network
import optimizer_two5_SharedEnc_depth as optimizer
import cv2,sys
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

from torch import nn
from layers import disp_to_depth

# Argument Parser
parser = argparse.ArgumentParser(description='Semantic Segmentation')
parser.add_argument('--lr', type=float, default=0.00001)
parser.add_argument('--arch', type=str, default='network.deepv3_noBG_Paralleltask_depth_noSeman.DeepWV3Plus',
                    help='Network architecture. We have DeepSRNX50V3PlusD (backbone: ResNeXt50) \
                    and deepWV3Plus (backbone: WideResNet38).')

parser.add_argument('--dataset', type=str, default='OmniAudio_noBG_Paralleltask_depth_noSeman_modified',
                    help='cityscapes, mapillary, camvid, kitti, OmniAudio')

parser.add_argument("--iter_for_report", type=int, default=1000, help='Number of iterations to log')
parser.add_argument("--log_dir", type=str, default='log/default', help='Directory for logging')

parser.add_argument('--cv', type=int, default=None,
                    help='cross-validation split id to use. Default # of splits set to 3 in config')

parser.add_argument('--class_uniform_pct', type=float, default=0.5,
                    help='What fraction of images is uniformly sampled')
parser.add_argument('--class_uniform_tile', type=int, default=1024,
                    help='tile size for class uniform sampling')
parser.add_argument('--coarse_boost_classes', type=str, default=None,
                    help='use coarse annotations to boost fine data with specific classes')

parser.add_argument('--img_wt_loss', action='store_true', default=True,
                    help='per-image class-weighted loss')
parser.add_argument('--batch_weighting', action='store_true', default=True,
                    help='Batch weighting for class (use nll class weighting using batch stats')

parser.add_argument('--jointwtborder', action='store_true', default=False,
                    help='Enable boundary label relaxation')
parser.add_argument('--strict_bdr_cls', type=str, default='',
                    help='Enable boundary label relaxation for specific classes')
parser.add_argument('--rlx_off_epoch', type=int, default=-1,
                    help='Turn off border relaxation after specific epoch count')
parser.add_argument('--rescale', type=float, default=1.0,
                    help='Warm Restarts new learning rate ratio compared to original lr')
parser.add_argument('--repoly', type=float, default=1.5,
                    help='Warm Restart new poly exp')

parser.add_argument('--apex', action='store_true', default=False,
                    help='Use Nvidia Apex Distributed Data Parallel')
parser.add_argument('--fp16', action='store_true', default=False,
                    help='Use Nvidia Apex AMP')

parser.add_argument('--local_rank', default=0, type=int,
                    help='parameter used by apex library')

parser.add_argument('--sgd', action='store_true', default=False)
parser.add_argument('--adam', action='store_true', default=True)
parser.add_argument('--amsgrad', action='store_true', default=False)

parser.add_argument('--freeze_trunk', action='store_true', default=False)
parser.add_argument('--hardnm', default=0, type=int,
                    help='0 means no aug, 1 means hard negative mining iter 1,' +
                    '2 means hard negative mining iter 2')

parser.add_argument('--trunk', type=str, default='resnet101',
                    help='trunk model, can be: resnet101 (default), resnet50')
parser.add_argument('--max_epoch', type=int, default=180)
parser.add_argument('--max_cu_epoch', type=int, default=100000,
                    help='Class Uniform Max Epochs')
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--color_aug', type=float,
                    default=0.25, help='level of color augmentation')
parser.add_argument('--gblur', action='store_true', default=True,
                    help='Use Guassian Blur Augmentation')
parser.add_argument('--bblur', action='store_true', default=False,
                    help='Use Bilateral Blur Augmentation')
parser.add_argument('--lr_schedule', type=str, default='poly',
                    help='name of lr schedule: poly')
parser.add_argument('--poly_exp', type=float, default=1.0,
                    help='polynomial LR exponent')
parser.add_argument('--bs_mult', type=int, default=2,
                    help='Batch size for training per gpu')
parser.add_argument('--bs_mult_val', type=int, default=1,
                    help='Batch size for Validation per gpu')
parser.add_argument('--crop_size', type=int, default=720,
                    help='training crop size')
parser.add_argument('--pre_size', type=int, default=None,
                    help='resize image shorter edge to this before augmentation')
parser.add_argument('--scale_min', type=float, default=0.5,
                    help='dynamically scale training images down to this size')
parser.add_argument('--scale_max', type=float, default=2.0,
                    help='dynamically scale training images up to this size')
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--snapshot', type=str, default=None)

parser.add_argument('--restore_optimizer', action='store_true', default=False)
parser.add_argument('--exp', type=str, default='default',
                    help='experiment directory name')
parser.add_argument('--tb_tag', type=str, default='',
                    help='add tag to tb dir')
parser.add_argument('--ckpt', type=str, default='logs/ckpt',
                    help='Save Checkpoint Point')
parser.add_argument('--tb_path', type=str, default='logs/tb',
                    help='Save Tensorboard Path')
parser.add_argument('--syncbn', action='store_true', default=False,
                    help='Use Synchronized BN')
parser.add_argument('--dump_augmentation_images', action='store_true', default=False,
                    help='Dump Augmentated Images for sanity check')
parser.add_argument('--test_mode', action='store_true', default=False,
                    help='Minimum testing to verify nothing failed, ' +
                    'Runs code for 1 epoch of train and val')
parser.add_argument('-wb', '--wt_bound', type=float, default=1.0,
                    help='Weight Scaling for the losses')
parser.add_argument('--maxSkip', type=int, default=0,
                    help='Skip x number of  frames of video augmented dataset')
parser.add_argument('--scf', action='store_true', default=False,
                    help='scale correction factor')
args = parser.parse_args()
args.best_record = {'epoch': -1, 'iter': 0, 'val_loss': 1e10, 'acc': 0,
                    'acc_cls': 0, 'mean_iu': 0, 'fwavacc': 0}

# Enable CUDNN Benchmarking optimization
torch.backends.cudnn.benchmark = True
args.world_size = 1

# Test Mode run two epochs with a few iterations of training and val
if args.test_mode:
    args.max_epoch = 2

if 'WORLD_SIZE' in os.environ and args.apex:
    args.apex = int(os.environ['WORLD_SIZE']) > 1
    args.world_size = int(os.environ['WORLD_SIZE'])
    print("Total world size: ", int(os.environ['WORLD_SIZE']))

if args.apex:
    # Check that we are running with cuda as distributed is only supported for cuda.
    torch.cuda.set_device(args.local_rank)
    print('My Rank:', args.local_rank)
    # Initialize distributed communication
    torch.distributed.init_process_group(backend='nccl', init_method='env://')

def spectrogram_for_SOP(mag_img):
    B = mag_img.shape[0]
    T = mag_img.shape[3]
    grid_warp = torch.from_numpy(
                warpgrid(B, 256, T, warp=True)).cuda()
    mag_img = F.grid_sample(mag_img, grid_warp)
    log_mag_img = torch.log(mag_img).detach()
    return log_mag_img

def main():
    """
    Main Function
    """

    # Set up the Arguments, Tensorboard Writer, Dataloader, Loss Fn, Optimizer
    assert_and_infer_cfg(args)
    args.ngpu = torch.cuda.device_count()
    # writer = prep_experiment(args, parser)
    writer = SummaryWriter(args.log_dir)
    train_loader, val_loader, train_obj = datasets.setup_loaders(args)
    print(len(train_loader))
    # criterion, criterion_val = loss.get_loss(args)
    ifr = args.iter_for_report
    # net = network.get_net(args, criterion)
    # optim, scheduler = optimizer.get_optimizer(args, net)
    
    # Set up the network
    net = Unet(in_ch=2, out_ch=1)
    optim = torch.optim.Adam(list(net.enc.parameters()) + list(net.dec.parameters()), lr=1e-4)
    crit = nn.MSELoss()
    tr_loss = list()
    '''
    if args.fp16:
        net, optim = amp.initialize(net, optim, opt_level="O1")

    net = network.warp_network_in_dataparallel(net, args.apex)
    if args.snapshot:
        optimizer.load_weights(net, optim, args.snapshot, args.restore_optimizer)
    '''

    torch.cuda.empty_cache()
    # Main Loop
    for epoch in range(args.start_epoch, args.max_epoch):
        # Update EPOCH CTR
        cfg.immutable(False)
        cfg.EPOCH = epoch
        cfg.immutable(True)
        curr_epoch = cfg.EPOCH
        curr_iter = curr_epoch * len(train_loader)

        # scheduler.step()
        net.train()
        net.to('cuda:0')

        train_main_loss = AverageMeter()
        curr_iter = curr_epoch * len(train_loader)

        for i, data in enumerate(train_loader):
            # inputs = (2,3,713,713)
            # gts    = (2,713,713)
            depthmask_original, gts_depth, in_aud1, in_aud6, _img_name = data

            depthmask_original, gts_depth, in_aud1, in_aud6 = depthmask_original.type(torch.FloatTensor).cuda(), gts_depth.type(torch.FloatTensor).cuda(), in_aud1.type(torch.FloatTensor).cuda(), in_aud6.type(torch.FloatTensor).cuda()

            optim.zero_grad()

            x = torch.cat((in_aud1, in_aud6), 1)
            y = gts_depth

            x = torch.nn.functional.interpolate(x, size=(128, 256))
            print(x.shape)
            depthmask_original = torch.nn.functional.interpolate(depthmask_original, size=(128, 256))
            y = torch.nn.functional.interpolate(y, size=(128, 256))

            o = net(x)
            loss = crit(o, y)
            loss.backward()

            tr_loss.append(loss.item())

            if (curr_epoch % ifr) == 0 and i == (len(train_loader) - 1):
                score = np.mean(tr_loss)
                print(curr_epoch, score)
                writer.add_scalar('training/loss', score, curr_epoch)
                depthmask_original_img = create_disp(depthmask_original[0], "gt_" + _img_name[0], args.log_dir, False)
                o_img = create_disp(o[0], "pred_" + _img_name[0], args.log_dir, False)
                save_img(writer, np.expand_dims(depthmask_original_img, 0), np.expand_dims(o_img, 0), curr_epoch)
                tr_loss = list()

def create_disp(disp_resized, output_name, output_directory, save=True):
    # Saving colormapped depth image
    disp_resized = unscale_disp(disp_resized, 0.1, 100)
    disp_resized_np = disp_resized.squeeze().cpu().detach().numpy()
    vmax = np.percentile(disp_resized_np, 95)
    normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)

    if save:
        im = pil.fromarray(colormapped_im)
        
        name_dest_im = os.path.join(output_directory, "{}_disp.jpeg".format(output_name))
        im.save(name_dest_im)

    return colormapped_im

def unscale_disp(scaled_disp, min_depth, max_depth):
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    unscaled_disp = (scaled_disp - min_disp) / (max_disp - min_disp)
    return unscaled_disp


def save_img(writer, true, pred, curr_iter):
    imgs = np.zeros((pred.shape[0], 128, 512, 3), np.uint8)
    for i, (t, p) in enumerate(zip(true, pred)):
        imgs[i] = np.concatenate((t, p), axis=1)
        cv2.rectangle(
            imgs[i], (0, 0), (512, 128), color=(255, 255, 255), thickness=2
        )
    imgs = imgs.transpose(0, 3, 1, 2)
    writer.add_images("rec", imgs, curr_iter)

'''
def train(train_loader, net, optim, curr_epoch, writer, crit, ifr, tr_loss):
    """
    Runs the training loop per epoch
    train_loader: Data loader for train
    net: the network
    optimizer: optimizer
    curr_epoch: current epoch
    writer: tensorboard writer
    return:
    """
    net.train()
    net.to('cuda:0')

    train_main_loss = AverageMeter()
    curr_iter = curr_epoch * len(train_loader)

    for i, data in enumerate(train_loader):
        # inputs = (2,3,713,713)
        # gts    = (2,713,713)
        gts_depth, in_aud1, in_aud6, _img_name = data

        gts_depth, in_aud1, in_aud6 = gts_depth.type(torch.FloatTensor).cuda(), in_aud1.type(torch.FloatTensor).cuda(), in_aud6.type(torch.FloatTensor).cuda()

        optim.zero_grad()

        print(gts_depth.shape)
        print(in_aud1.shape)
        print(in_aud6.shape)
        print(_img_name)
        
        x = torch.cat((in_aud1, in_aud6), 1)
        y = gts_depth

        x = torch.nn.functional.interpolate(x, size=(256, 512))
        y = y.squeeze(2)
        print(y.shape)
        y = torch.nn.functional.interpolate(y, size=(256, 512))

        o = net(x)
        
        print(torch.max(y))
        print(torch.min(y))
        print(torch.max(o))
        print(torch.min(o))
 
        print("o")
        print(o.shape)
        
        loss = crit(o, y)
        loss.backward()

        tr_liss.append(loss.item())

        if (curr_epoch % ifr) == 0:
            score = np.mean(tr_loss)
            writer.add_scalar('training/loss', score, curr_epochi)
            tr_loss = list()

        main_loss = net(in_imgs, in_aud1, in_aud6, gts_diff_2=gts_diff_2,gts_diff_5=gts_diff_5,gts_depth=gts_depth)
        
        if args.apex:
            log_main_loss = main_loss.clone().detach_()
            torch.distributed.all_reduce(log_main_loss, torch.distributed.ReduceOp.SUM)
            log_main_loss = log_main_loss / args.world_size
        else:
            main_loss = main_loss.mean()
            log_main_loss = main_loss.clone().detach_()

        train_main_loss.update(log_main_loss.item(), batch_pixel_size)
        if args.fp16:  # and 0:
            with amp.scale_loss(main_loss, optim) as scaled_loss:
                scaled_loss.backward()
        else:
            main_loss.backward()
        clip=1
        torch.nn.utils.clip_grad_norm_(net.parameters(),clip)
        optim.step()

        curr_iter += 1

        if args.local_rank == 0:
            msg = '[epoch {}], [iter {} / {}], [train main loss {:0.6f}], [lr {:0.6f}]'.format(
                curr_epoch, i + 1, len(train_loader), train_main_loss.avg,
                optim.param_groups[-1]['lr'])

            logging.info(msg)

            # Log tensorboard metrics for each iteration of the training phase
            writer.add_scalar('training/loss', (train_main_loss.val),
                              curr_iter)
            writer.add_scalar('training/lr', optim.param_groups[-1]['lr'],
                              curr_iter)

       
       if i > 5 and args.test_mode:
            returnss =
        '''

if __name__ == '__main__':
    main()
