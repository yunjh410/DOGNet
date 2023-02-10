import os, time, glob, random
import argparse
import cv2
import torch
import numpy as np

from utils import *
from model import *
from dataset import JPEG_Dataset_val, JPEG_Dataset_train_gray, JPEG_Dataset_train_color
from torch.utils.data import DataLoader, ConcatDataset
from torch import nn
from torch.nn import functional as F

random.seed(2022)
torch.manual_seed(2022)
np.random.seed(2022)

def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    
    # Dataset
    parser.add_argument('--train_dataset', nargs='+', type=str, default=['DIV2K_train', 'Flickr2K'], help='list of train dataset')
    parser.add_argument('--val_dataset', type=str, default='live1', help='validation dataset')
    parser.add_argument('--mode', type=str, default='gray', choices=['gray', 'color'], help='type of the dataset. color or gray')
    
    # GPU
    parser.add_argument('--gpu',type=int, nargs='+', default=0, help='GPU index')
    parser.add_argument('--mgpu', type=str, default=False, help='use multiple gpu to train')
    
    # Training
    parser.add_argument('--max_epoch',type=int ,default='30000', help='max train epoch')
    parser.add_argument('--batch_size',type=int ,default='2', help='training batch size')
    parser.add_argument('--num_workers',type=int ,default='4', help='number of workers')
    parser.add_argument('--exp_name', type=str, default='temp', help='the name of experiment, where path file saved')
    parser.add_argument('--patch_size', type=int, default='128', help='training patch size')
    parser.add_argument('--double_aug', type=bool, default=False, help='double compression augmentation during train')

    # Validation
    parser.add_argument('--val_qf', type=int, default='10', help='')
    parser.add_argument('--val_freq', type=int, default='50', help='validate model every [val_freq] epoch')
    
    args = parser.parse_args()
    
    # make path to model save
    os.makedirs(f'./saved_models/{args.exp_name}', exist_ok=True)
    
    # GPU settings
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if args.mgpu:
        device = (f"cuda:{args.gpu[0]}" if torch.cuda.is_available() else "cpu")
    else:
        device = (f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    log = logger(f'./saved_models/{args.exp_name}/train_log.txt', 'train', 'a')
    args_log = '------------ Options -------------\n'
    for k, v in vars(args).items():
        args_log += f'{str(k)}: {str(v)}\n'
    args_log += '---------------------------------------\n'
    log.info(args_log)

    # Dataset
    val_dataset = JPEG_Dataset_val(args)
    train_dataset = JPEG_Dataset_train_gray(args) if args.mode == 'gray' else JPEG_Dataset_train_color(args)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=int(args.num_workers))
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=int(args.num_workers))
    
    log.info(f'training : {len(train_dataset)}')
    
    ## Model
    model = DOG_VAE(channels=64, mode=args.mode)
    Dis = Discriminator(channels=64, mode=args.mode) 

    if args.mgpu:
        model = nn.DataParallel(model, device_ids = args.gpu, output_device=args.gpu[0])
        Dis = nn.DataParallel(Dis, device_ids = args.gpu, output_device=args.gpu[0])
    
    model = model.cuda()
    Dis = Dis.cuda()

    # get checkpoint adrs
    saved_model_adrs = sorted(glob.glob(f'./saved_models/{args.exp_name}/*.pth'))
    saved_model = None
    if len(saved_model_adrs):
        saved_model = saved_model_adrs[-1]
    else:
        log.info(f'no saved weights in ./saved_models/{args.exp_name}/')
    
    optimizer = torch.optim.Adam(params=model.parameters(), lr=5e-5)
    optimizer_d = torch.optim.Adam(params=Dis.parameters(), lr=5e-5, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=2e-6)
    scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_d, T_max=1000, eta_min=2e-6)

    start = time.time()
    start_epoch = 0
    if saved_model:
        log.info(f'loading pretrained model from {saved_model}')
        checkpoint = torch.load(saved_model, map_location=device)
        start_epoch = checkpoint['epoch']
        
        if args.mgpu:
            model.module.load_state_dict(checkpoint['model_state_dict'])
            Dis.module.load_state_dict(checkpoint['discriminator_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
            Dis.load_state_dict(checkpoint['discriminator_state_dict'])

        load_optimizer_state_dict(optimizer, checkpoint['optimizer_state_dict'])
        load_optimizer_state_dict(optimizer_d, checkpoint['optimizer_d_state_dict'])
        
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scheduler_d.load_state_dict(checkpoint['scheduler_d_state_dict'])
        
        log.info(f'continue to train, start_epoch : {start_epoch}')

    for epoch in range(start_epoch + 1, args.max_epoch):
        loss, loss_denoise, loss_KL, loss_g, loss_ae_recon, loss_EST, loss_d, loss_fft = train(args=args,
                                                                                    model=model,
                                                                                    Discriminator=Dis,  
                                                                                    train_loader=train_loader,
                                                                                    optimizer=optimizer,
                                                                                    optimizer_d=optimizer_d,
                                                                                    device=device)
        elapsed_time = date_time(time.time() - start)

        scheduler.step()
        scheduler_d.step()
        lr = optimizer.param_groups[0]['lr']

        # validation
        if epoch % args.val_freq == 0:
            validation(args, val_loader, model, log, device)
            save_checkpoint(f'./saved_models/{args.exp_name}/{str(epoch).zfill(5)}.pth', model, Dis, optimizer, optimizer_d, epoch, scheduler, scheduler_d, args.mgpu)
            log.info(f'Elapsed time: {elapsed_time}')
            log.info(f'[{epoch}/{args.max_epoch}] Train Loss: {loss}\n')
            log.info(f'learning rate is {lr}')
            log.info(f'Denoise loss: {loss_denoise:0.3f}, KL loss: {loss_KL:0.3f}, Gen loss: {loss_g:0.3f}, AE recon loss: {loss_ae_recon:0.3f}, Noise level loss: {loss_EST:0.3f}, Discrim loss: {loss_d:0.3f}, FFT loss : {loss_fft:0.3f}')


def train(args, train_loader, model, Discriminator, optimizer, optimizer_d, device):
    
    model.train()
    Discriminator.train()
    
    losses = AverageMeter()
    losses_d = AverageMeter()
    losses_g = AverageMeter()
    losses_denoise = AverageMeter()
    losses_KL = AverageMeter()
    losses_EST = AverageMeter()
    losses_ae_recon = AverageMeter()
    losses_fft = AverageMeter()

    l1_loss = nn.L1Loss()
    l1char_loss = L1_Charbonnier_loss()
    for batch_idx, (images, labels, qf_map) in enumerate(train_loader):
        images, labels, qf_map = images.to(device), labels.to(device), qf_map.to(device)
        
        # forward prop
        mu, log_sigma_sq, y_hat, x, qf_map_est = model(images, is_training=True)

        # update Discriminator first
        f_logits_d = Discriminator(y_hat)
        r_logits_d = Discriminator(images)
        
        # adversarial loss for discriminator
        loss_d_fake = F.binary_cross_entropy_with_logits(input=f_logits_d, target=torch.zeros_like(f_logits_d)).mean()
        loss_d_real = F.binary_cross_entropy_with_logits(input=r_logits_d, target=torch.ones_like(r_logits_d)).mean()
        loss_d = loss_d_fake + loss_d_real
        losses_d.update(loss_d.item(), images.size(0))
        
        optimizer_d.zero_grad()
        loss_d.backward(retain_graph = True)
        optimizer_d.step()
        
        # update model
        f_logits_g = Discriminator(y_hat)
                
        # calculate losses
        # First term
        loss_denoise = l1char_loss(x, labels)
        losses_denoise.update(loss_denoise.item(), images.size(0))
        
        # Second term, KL Divergence Loss
        loss_KL = 0.5 * (torch.exp(log_sigma_sq) + torch.square(mu) - 1 - log_sigma_sq).mean()
        losses_KL.update(loss_KL.item(), images.size(0))
        
        # Third Term
        # l1 loss between decoder output and model input
        loss_AE_recon = l1char_loss(y_hat, images)
        losses_ae_recon.update(loss_AE_recon.item(), images.size(0))
        
        # l1 loss between qf and estimated qf
        loss_EST = l1char_loss(torch.mean(torch.mean(qf_map, axis=-1), axis=-1), qf_map_est) 
        losses_EST.update(loss_EST.item(), images.size(0))
        
        # adversarial loss for generator
        loss_g = F.binary_cross_entropy_with_logits(input=f_logits_g, target=torch.ones_like(f_logits_g)).mean()
        losses_g.update(loss_g.item(), images.size(0))
        
        loss = loss_denoise + 0.01 * loss_KL + 0.5 * loss_AE_recon + 0.001 * loss_g + loss_EST
        losses.update(loss.item(), images.size(0))
        
        # update model
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2)
        optimizer.step()
         
    return losses.avg, losses_denoise.avg, losses_KL.avg, losses_g.avg, losses_ae_recon.avg, losses_EST.avg, losses_d.avg, losses_fft.avg


def validation(args, val_loader, model, log, device):
    
    # make visualization folder
    os.makedirs(f'./viz/{args.val_qf*10}', exist_ok=True)
    
    model.eval()
    psnr_out = list()
    ssim_out = list()
    
    with torch.no_grad():
        for batch_idx, (image, label, sigmas_noise) in enumerate(val_loader):
            label = label.to(device)
            _, _, img_h, img_w = image.shape
            patches = split_image_to_patches(image)
            patches = patches.to(device)
            # forward prop
            _, _, _, x, _= model(patches, is_training=False)
            x = x.detach().cpu()
            x = recon_patches_to_image(x, image_size = (img_h, img_w))
            
            label = label.detach().cpu().numpy() * 255
            
            x = x.numpy() * 255
            x = np.clip(x, 0, 255)
            cv2.imwrite(f'./viz/{args.val_qf*10}/{batch_idx}.png', x[0][0])
            ssim_out.append(ssim2(x[0][0], label[0][0]))
            psnr_out.append(psnr(x, label))
            
    log.info(f'QF : {args.val_qf} PSNR : {np.mean(psnr_out)}, SSIM : {np.mean(ssim_out)}')


if __name__ == '__main__':
    main()
