import torch
from torch import nn
from torch.nn import Parameter
import math
import cv2
import numpy as np
import logging
from collections import OrderedDict

def conv2d(in_, out_, stride=1):
    return torch.nn.Conv2d(in_channels=in_, out_channels=out_, kernel_size=(3, 3), stride=(stride, stride), padding=1, bias=True, padding_mode='zeros')

def conv1x1(in_, out_, stride=1, bias=True):
    return torch.nn.Conv2d(in_channels=in_, out_channels=out_, kernel_size=(1, 1), stride=(stride, stride), padding=0, bias=bias, padding_mode='zeros')

def sequential(*args):
    """Advanced nn.Sequential.
    Args:
        nn.Sequential, nn.Module
    Returns:
        nn.Sequential
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)

def save_checkpoint(path, model, discriminator, optimizer, optimizer_d, epoch, scheduler, scheduler_d, mgpu=False):
    if mgpu:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.module.state_dict(),
            'discriminator_state_dict' : discriminator.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'optimizer_d_state_dict': optimizer_d.state_dict(),
            'scheduler_state_dict' : scheduler.state_dict(),
            'scheduler_d_state_dict' : scheduler_d.state_dict()
        }, path)
    else:
        torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'discriminator_state_dict' : discriminator.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'optimizer_d_state_dict': optimizer_d.state_dict(),
        'scheduler_state_dict' : scheduler.state_dict(),
        'scheduler_d_state_dict' : scheduler_d.state_dict()
        }, path)
    
def load_optimizer_state_dict(optimizer, state_dict):
    optimizer.load_state_dict(state_dict)
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()
            
# Spectral Normalization from https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

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

def logger(log_adr, logger_name, mode='w'):
    """
    Logger
    """
    # create logger
    _logger = logging.getLogger(logger_name)
    # set level
    _logger.setLevel(logging.INFO)
    # set format
    formatter = logging.Formatter('%(message)s')
    # stdout
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    _logger.addHandler(stream_handler)
    # file
    file_handler = logging.FileHandler(log_adr, mode=mode)
    file_handler.setFormatter(formatter)
    _logger.addHandler(file_handler)
    return _logger

def date_time(secs):
    day = secs // (24 * 3600)
    secs = secs % (24 * 3600)
    hour = secs // 3600
    secs %= 3600
    minutes = secs // 60
    secs %= 60
    seconds = int(secs)
    return f'{day} d {hour} h {minutes} m {seconds} s'

def _convert_input_type_range(img):
    """Convert the type and range of the input image.

    It converts the input image to np.float32 type and range of [0, 1].
    It is mainly used for pre-processing the input image in colorspace
    convertion functions such as rgb2ycbcr and ycbcr2rgb.

    Args:
        img (ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].

    Returns:
        (ndarray): The converted image with type of np.float32 and range of
            [0, 1].
    """
    img_type = img.dtype
    img = img.astype(np.float32)
    if img_type == np.float32:
        pass
    elif img_type == np.uint8:
        img /= 255.
    else:
        raise TypeError('The img type should be np.float32 or np.uint8, ' f'but got {img_type}')
    return img

def _convert_output_type_range(img, dst_type):
    """Convert the type and range of the image according to dst_type.

    It converts the image to desired type and range. If `dst_type` is np.uint8,
    images will be converted to np.uint8 type with range [0, 255]. If
    `dst_type` is np.float32, it converts the image to np.float32 type with
    range [0, 1].
    It is mainly used for post-processing images in colorspace convertion
    functions such as rgb2ycbcr and ycbcr2rgb.

    Args:
        img (ndarray): The image to be converted with np.float32 type and
            range [0, 255].
        dst_type (np.uint8 | np.float32): If dst_type is np.uint8, it
            converts the image to np.uint8 type with range [0, 255]. If
            dst_type is np.float32, it converts the image to np.float32 type
            with range [0, 1].

    Returns:
        (ndarray): The converted image with desired type and range.
    """
    if dst_type not in (np.uint8, np.float32):
        raise TypeError('The dst_type should be np.float32 or np.uint8, ' f'but got {dst_type}')
    if dst_type == np.uint8:
        img = img.round()
    else:
        img /= 255.
    return img.astype(dst_type)

def ycbcr2rgb(img):
    '''same as matlab ycbcr2rgb
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    # if in_img_type != np.uint8:
    #     img *= 255.
    # convert
    rlt = np.matmul(img, [[0.00456621, 0.00456621, 0.00456621], [0, -0.00153632, 0.00791071],
                          [0.00625893, -0.00318811, 0]]) * 255.0 + [-222.921, 135.576, -276.836]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)

def bgr2ycbcr(img, y_only=False):
    """Convert a BGR image to YCbCr image.

    The bgr version of rgb2ycbcr.
    It implements the ITU-R BT.601 conversion for standard-definition
    television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

    It differs from a similar function in cv2.cvtColor: `BGR <-> YCrCb`.
    In OpenCV, it implements a JPEG conversion. See more details in
    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.

    Args:
        img (ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].
        y_only (bool): Whether to only return Y channel. Default: False.

    Returns:
        ndarray: The converted YCbCr image. The output image has the same type
            and range as input image.
    """
    img_type = img.dtype
    img = _convert_input_type_range(img)
    if y_only:
        out_img = np.dot(img, [24.966, 128.553, 65.481]) + 16.0
    else:
        out_img = np.matmul(
            img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786], [65.481, -37.797, 112.0]]) + [16, 128, 128]
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img

def augmentation(x,mode):
    if mode ==0:
        y=x

    elif mode ==1:
        y=np.flipud(x)

    elif mode == 2:
        y = np.rot90(x,1)

    elif mode == 3:
        y = np.rot90(x, 1)
        y = np.flipud(y)

    elif mode == 4:
        y = np.rot90(x, 2)

    elif mode == 5:
        y = np.rot90(x, 2)
        y = np.flipud(y)

    elif mode == 6:
        y = np.rot90(x, 3)

    elif mode == 7:
        y = np.rot90(x, 3)
        y = np.flipud(y)

    return y

class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt( diff * diff + self.eps )
        loss = torch.mean(error) 
        return loss

def split_image_to_patches(image, patch_size=(128, 128), stride=64):
    '''
    input image is torch tensor [1, C, H, W]
    return: patches
    '''
    _, c, img_h, img_w = np.shape(image)
    patch_h, patch_w = patch_size
    stride_h = int(np.ceil(img_h/patch_h*2) - 1); stride_w = int(np.ceil(img_w/patch_h*2) - 1)
    patch_num = stride_h * stride_w
    
    p_image = torch.zeros((patch_num, c, patch_h, patch_w))
    margin_h = patch_h - img_h % patch_h; margin_w = patch_w - img_w % patch_w
    for i in range(stride_h):
        for j in range(stride_w):
            if i!=stride_h-1 and j!=stride_w-1:
                p_image[j*stride_h + i, ...] = image[:, :, i*stride:i*stride + patch_h, j*stride:j*stride + patch_w]
            elif i!=stride_h-1 and j==stride_w-1:
                p_image[j*stride_h + i, ...] = image[:, :, i*stride:i*stride + patch_h, -patch_w:]
            elif i==stride_h-1 and j!=stride_w-1:
                p_image[j*stride_h + i, ...] = image[:, :, -patch_h:, j*stride:j*stride + patch_w]
            else:
                p_image[j*stride_h + i, ...] = image[:, :, -patch_h:, -patch_w:]
    
    return p_image

def recon_patches_to_image(patches, image_size, patch_size=(128,128), stride=64, mode='gray'):
    
    img_h, img_w = image_size
    c = patches.shape[1]
    output_clean_image = torch.zeros((1, c, *image_size))
    weight_clean_image = torch.zeros((1, c, *image_size))

    patch_h, patch_w = patch_size
    stride_h = int(np.ceil(img_h/patch_h*2) - 1); stride_w = int(np.ceil(img_w/patch_h*2) - 1)

    margin_h = patch_h - img_h % patch_h; margin_w = patch_w - img_w % patch_w

    for i in range(stride_h):
        for j in range(stride_w):
            if i!=stride_h-1 and j!=stride_w-1:
                output_clean_image[:, :, i*stride:i*stride + patch_h, j*stride:j*stride + patch_w] += patches[j*stride_h + i, ...]
                weight_clean_image[:, :, i*stride:i*stride + patch_h, j*stride:j*stride + patch_w] += np.ones(patch_size)
            elif i!=stride_h-1 and j==stride_w-1:
                output_clean_image[:, :, i*stride:i*stride + patch_h, -patch_w:] += patches[j*stride_h + i, ...]
                weight_clean_image[:, :, i*stride:i*stride + patch_h, -patch_w:] += np.ones(patch_size)
            elif i==stride_h-1 and j!=stride_w-1:
                output_clean_image[:, :, -patch_h:, j*stride:j*stride + patch_w] += patches[j*stride_h + i, ...]
                weight_clean_image[:, :, -patch_h:, j*stride:j*stride + patch_w] += np.ones(patch_size)
            else:
                output_clean_image[:, :, -patch_h:, -patch_w:] += patches[j*stride_h + i, ...]
                weight_clean_image[:, :, -patch_h:, -patch_w:] += np.ones(patch_size)
        
    output_clean_image = torch.div(output_clean_image, weight_clean_image)
    return output_clean_image

def ssim2(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def psnr(img1, img2):
    img1=np.float32(img1)
    img2=np.float32(img2)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    if np.max(img1) <= 1.0:
        PIXEL_MAX= 1.0
    else:
        PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))