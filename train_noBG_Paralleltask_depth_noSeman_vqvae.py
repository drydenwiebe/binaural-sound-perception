from __future__ import absolute_import
from __future__ import division
"""
Author: Arun Balajee Vasudevan
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
"""
training code
"""
import argparse
import logging
import os
import torch
import torch.nn.functional as F
import torchvision
from apex import amp

from config import cfg, assert_and_infer_cfg
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

sys.path.insert(0, './../../monodepth2-master/')
from layers import disp_to_depth

sys.path.insert(0, './../pytorch-msssim/')
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

def compute_errors(gt, pred):
    """
    Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log

# Argument Parser
parser = argparse.ArgumentParser(description='Semantic Segmentation')
parser.add_argument('--lr', type=float, default=0.00001)
parser.add_argument('--arch', type=str, default='network.deepv3_noBG_Paralleltask_depth_noSeman.DeepWV3Plus',
                    help='Network architecture. We have DeepSRNX50V3PlusD (backbone: ResNeXt50) \
                    and deepWV3Plus (backbone: WideResNet38).')

parser.add_argument('--dataset', type=str, default='OmniAudio_noBG_Paralleltask_depth_noSeman_modified',
                    help='cityscapes, mapillary, camvid, kitti, OmniAudio')

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

# Test Mode run two epochs with a few iterations of training and val
if args.test_mode:
    args.max_epoch = 2

def spectrogram_for_SOP(mag_img):
    B = mag_img.shape[0]
    T = mag_img.shape[3]
    grid_warp = torch.from_numpy(
                warpgrid(B, 256, T, warp=True)).cuda()
    mag_img = F.grid_sample(mag_img, grid_warp)
    log_mag_img = torch.log(mag_img).detach()
    return log_mag_img

def save_disp_jpg(disp_resized, output_directory, curr_epoch, img_name):
    # Saving colormapped depth image
    disp_resized_np = disp_resized.squeeze().cpu().numpy()
    vmax = np.percentile(disp_resized_np, 95)
    normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3]) # * 255).astype(np.uint8)
    name_dest_im = os.path.join(output_directory,"best_depths/", str(curr_epoch)+"_{}_disp.jpg".format(img_name))
    
    return torch.from_numpy(colormapped_im), name_dest_im

def main():
    """
    Main Function
    """

    # Set up the Arguments, Tensorboard Writer, Dataloader, Loss Fn, Optimizer
    assert_and_infer_cfg(args)
    writer = prep_experiment(args, parser)
    train_loader, val_loader, train_obj = datasets.setup_loaders(args)
    criterion, criterion_val = loss.get_loss(args)
    net = network.get_net(args, criterion)
    optim, scheduler = optimizer.get_optimizer(args, net)

    print("Number of batches")
    print(len(train_loader))

    net = network.warp_network_in_dataparallel(net, args.apex)
    if args.snapshot:
        optimizer.load_weights(net, optim,
                               args.snapshot, args.restore_optimizer)

    torch.cuda.empty_cache()
    # Main Loop
    for epoch in range(args.start_epoch, args.max_epoch):
        # Update EPOCH CTR
        cfg.immutable(False)
        cfg.EPOCH = epoch
        cfg.immutable(True)

        train(train_loader, net, optim, epoch, writer)
        scheduler.step()
        validate(val_loader, net, criterion_val, optim, epoch, writer)

def train(train_loader, net, optim, curr_epoch, writer):
    """
    Runs the training loop per epoch
    train_loader: Data loader for train
    net: thet network
    optimizer: optimizer
    curr_epoch: current epoch
    writer: tensorboard writer
    return:
    """
    net.train()

    train_main_loss = AverageMeter()
    curr_iter = curr_epoch * len(train_loader)

    mean_train_loss = []

    for i, data in enumerate(train_loader):
        gts_diff_2, gts_diff_5, gts_depth, in_aud1, in_aud6, _img_name = data 

        # batch_pixel_size = in_imgs.size(0) * in_imgs.size(2) * in_imgs.size(3)
        batch_pixel_size = gts_depth.size(0) * gts_depth.size(2) * gts_depth.size(3)

        gts_diff_2, gts_diff_5 = gts_diff_2.cuda(), gts_diff_5.cuda()
        gts_depth,  in_aud1, in_aud6 = gts_depth.type(torch.FloatTensor).cuda(), in_aud1.type(torch.FloatTensor).cuda(), in_aud6.type(torch.FloatTensor).cuda()

        optim.zero_grad()
        in_imgs = torch.zeros(gts_depth.shape)
        depth_loss, main_loss = net(in_imgs, in_aud1, in_aud6, gts_diff_2=gts_diff_2, gts_diff_5=gts_diff_5, gts_depth=gts_depth)

        main_loss = main_loss.mean()
        log_main_loss = main_loss.detach().clone()

        train_main_loss.update(log_main_loss.item(), batch_pixel_size)
            
        main_loss.backward()
        
        clip = 1
        torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
        optim.step()

        curr_iter += 1
        
        mean_train_loss.append(depth_loss.mean().detach().clone().item())

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
            return
    
    writer.add_scalar('training/loss_epoch', (np.mean(mean_train_loss)), curr_epoch)


def validate(val_loader, net, criterion, optim, curr_epoch, writer):
    """
    Runs the validation loop after each training epoch
    val_loader: Data loader for validation
    net: thet network
    criterion: loss fn
    optimizer: optimizer
    curr_epoch: current epoch
    writer: tensorboard writer
    return: val_avg for step function if required
    """

    net.eval()
    val_loss = AverageMeter()
    val_loss_depth = AverageMeter()
    iou_acc = 0
    dump_images = []
    errors_abs_rel = []
    errors_sq_rel = []
    errors_rmse = []
    errors_rmse_log = []
    errors = []
    MSEcriterion = torch.nn.MSELoss()
    
    ssim_module = ssim_module = SSIM(data_range=1.0, size_average=True, channel=1)

    mean_val_ssim_loss = []
    mean_val_mse_loss = []

    for val_idx, data in enumerate(val_loader):
        gts_depth, in_aud1, in_aud6, img_names = data

        batch_pixel_size = gts_depth.size(0) * gts_depth.size(2) * gts_depth.size(3)
        
        gts_depth, in_aud1, in_aud6 = gts_depth.type(torch.FloatTensor).cuda(), in_aud1.type(torch.FloatTensor).cuda(), in_aud6.type(torch.FloatTensor).cuda()

        in_imgs = torch.zeros(gts_depth.shape)

        with torch.no_grad():
            outdepth = net(in_imgs, in_aud1, in_aud6, gts_depth=gts_depth)
        
        depth_loss = MSEcriterion(outdepth, gts_depth)
        
        depth_ssim = 1 - ssim_module(outdepth, gts_depth)

        mean_val_mse_loss.append(depth_loss.item())

        mean_val_ssim_loss.append(depth_ssim.item())

        val_loss_depth.update(depth_loss.item(), batch_pixel_size)

        predictions_depth = outdepth.data
        
        # Logging
        if val_idx % 20 == 0:
            if args.local_rank == 0:
                logging.info("validating: %d / %d): Loss %f", val_idx + 1, len(val_loader), val_loss_depth.avg)
        if val_idx > 10 and args.test_mode:
            break

        for i in range(predictions_depth.shape[0]):
            scaled_disp1, _ = disp_to_depth(predictions_depth[i,:,:,:], 0.1, 100)
            scaled_disp2, _ = disp_to_depth(gts_depth[i,:,:,:], 0.1, 100)
            computed_errors = compute_errors(scaled_disp2.cpu().numpy(), scaled_disp1.cpu().numpy()) 
            errors.append(computed_errors)
            errors_abs_rel.append(computed_errors[0])
            errors_sq_rel.append(computed_errors[1])
            errors_rmse.append(computed_errors[2])
            errors_rmse_log.append(computed_errors[3])
 

        # Image Dumps
        if val_idx % 100 == 0:
            img_name = img_names[0]
            disp = predictions_depth[0:1,:,:,:]
            disp_resized = torch.nn.functional.interpolate(disp, (in_imgs.size(2), in_imgs.size(3)), mode="bilinear", align_corners=False)

            output_directory = "/home/drydenw/projects/rrg-kyi/drydenw/binaural-sound-perception/logs/ckpt/default/Omni-network.deepv3_noBG_Paralleltask_depth_noSeman.DeepWV3Plus/"
            os.makedirs(output_directory+"best_depths/", exist_ok=True)
            
            depth, depth_path = save_disp_jpg(disp_resized, output_directory, curr_epoch, img_name)

            gt_img_name = img_names[0] + "_gt"
            gt_disp = gts_depth[0:1,:,:,:]
            gt_disp_resized = torch.nn.functional.interpolate(gt_disp, (in_imgs.size(2), in_imgs.size(3)), mode="bilinear", align_corners=False)
            
            gt_depth, _ = save_disp_jpg(gt_disp_resized, output_directory, curr_epoch, gt_img_name)
            
            depth = depth.permute((2, 0, 1))
            gt_depth = gt_depth.permute((2, 0, 1))
            image_to_save = torch.cat((depth, gt_depth), -1)
            torchvision.utils.save_image(torch.cat((depth, gt_depth), -1), depth_path)

        del outdepth, data

    writer.add_scalar('validation/mse_loss_epoch', (np.mean(mean_val_mse_loss)), curr_epoch)
    writer.add_scalar('validation/ssim_loss_epoch', (np.mean(mean_val_ssim_loss)), curr_epoch)

    writer.add_scalar('validation/abs_rel', (np.mean(np.array(errors_abs_rel))), curr_epoch)
    writer.add_scalar('validation/sq_rel', (np.mean(np.array(errors_sq_rel))), curr_epoch)
    writer.add_scalar('validation/rmse', (np.mean(np.array(errors_rmse))), curr_epoch)
    writer.add_scalar('validation/rmse_log', (np.mean(np.array(errors_rmse_log))), curr_epoch)


    to_save_dir = os.path.join(output_directory,"models_depth")
    os.makedirs(to_save_dir, exist_ok=True)
    save_snapshot = 'SOP_epoch_{}.pth'.format(curr_epoch)
    save_snapshot = os.path.join(to_save_dir, save_snapshot)
    
    torch.cuda.synchronize()
    
    torch.save({
        'state_dict': net.state_dict(),
        'optimizer': optim.state_dict(),
        'epoch': curr_epoch
    }, save_snapshot)
    
    print("Saving the model")
    print(save_snapshot)
    
    return val_loss.avg


if __name__ == '__main__':
    main()
