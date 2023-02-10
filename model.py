import torch
from torch import nn
from utils import *
from deform_conv import ModulatedDeformConv as DeformConv2

class Encoder(nn.Module):
    
    def __init__(self, channels=64, mode='gray'):
        super(Encoder, self).__init__()
        img_ch = 1 if mode == 'gray' else 3

        self.conv1 = conv2d(img_ch, channels)
        self.conv2 = conv2d(channels, channels*2)
        self.conv3 = conv2d(channels*2, channels*2)
        self.conv4 = conv2d(channels*2, channels*4)
        self.conv5 = conv2d(channels*4, channels*4)

        self.conv6 = conv2d(channels*4, channels*4)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.linear1 = nn.Linear(256, 256)
        self.linear2 = nn.Linear(256, 256)
        self.adaptivepool = nn.AdaptiveAvgPool2d((1,1))
        
    def forward(self, input):
        # Block 1
        x = self.conv1(input)
        x = self.maxpool(x)
        x = self.relu(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.maxpool(x)
        x = self.relu(x)
        
        # Block 3
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.maxpool(x)
        x = self.relu(x)
        
        x = self.conv6(x)

        mu = self.linear1(nn.Flatten()(self.adaptivepool(x)))
        log_sig = self.linear2(nn.Flatten()(self.adaptivepool(x)))

        return mu, log_sig

class Decoder(nn.Module):
    
    def __init__(self, channels=64, mode='gray'):
        super(Decoder, self).__init__()
        img_ch = 1 if mode == 'gray' else 3
        
        self.conv1 = conv2d(1, channels)
        self.conv2 = conv2d(channels, channels)
        self.conv3 = conv2d(channels, channels)
        self.conv4 = conv2d(channels, channels)
        self.conv5 = conv2d(channels, channels)
        self.conv6 = conv2d(channels, img_ch)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.relu = nn.ReLU()
    
    def forward(self, input):
        # Block 1
        x = self.conv1(input)
        x = self.upsample(x)
        x = self.relu(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.upsample(x)
        x = self.relu(x)
        
        # Block 3
        x = self.conv4(x)
        x = self.relu(x)
        x = self.upsample(x)
        x = self.conv5(x)
        x = self.relu(x)

        # output
        output = self.conv6(x)
        return output
         
class ResBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64):
        super(ResBlock, self).__init__()
        self.conv1 = conv2d(in_channels, in_channels)
        self.conv2 = conv2d(in_channels, in_channels)
        self.conv_last = conv2d(in_channels, out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, input):
        x = self.conv1(input)
        x = self.relu(x)
        out = self.conv2(x) + input
        return self.conv_last(out)

class DeformResBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, offset_channels=32):
        super(DeformResBlock, self).__init__()
        self.conv1 = DeformConv2(in_channels, in_channels, offset_in_channels=offset_channels, extra_offset_mask=True)
        self.conv2 = DeformConv2(in_channels, in_channels, offset_in_channels=offset_channels, extra_offset_mask=True)
        self.leakyrelu = nn.LeakyReLU()
        self.conv_last = conv2d(in_channels, out_channels)
        
    def forward(self, input, offset):
        x = self.conv1([input, offset])
        x = self.leakyrelu(x)
        out = self.conv2([x, offset]) + input

        return self.conv_last(out)
    
class OffsetBlock(nn.Module):
    def __init__(self, in_channels=64, offset_channels=32):
        super(OffsetBlock, self).__init__()
        self.conv1 = conv2d(in_channels, offset_channels)
        self.conv2 = conv2d(offset_channels, 1)
        self.conv3 = conv2d(offset_channels*3, offset_channels)
        self.leakyrelu = nn.LeakyReLU()
        self.upsample = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, input, last_offset=None, lp=None):
        reshaped_input = self.conv1(input)
        offset = self.leakyrelu(reshaped_input)
        if last_offset is not None:
            last_offset = nn.Upsample(size=(offset.shape[2], offset.shape[3]), mode='bilinear', align_corners=True)(last_offset)
            offset = self.conv3(torch.cat([offset, last_offset], 1))
    
        prior = lp.unsqueeze(-1).unsqueeze(-1)
        out = offset * prior
        return out
    
class Denoiser(nn.Module):
    def __init__(self, channels=64, mode='gray'):
        super(Denoiser, self).__init__()
        img_ch = 1 if mode == 'gray' else 3
            
        self.conv0 = conv2d(img_ch + 1, channels)
        
        self.res1 = ResBlock(in_channels=channels, out_channels=channels*2)
        self.res2 = ResBlock(in_channels=channels*2, out_channels=channels*4)
        self.res3 = ResBlock(in_channels=channels*4, out_channels=channels*8)
        self.res4 = ResBlock(in_channels=channels*8, out_channels=channels*8)
        
        self.offset1 = OffsetBlock(in_channels=channels*8, offset_channels=128)
        self.offset2 = OffsetBlock(in_channels=channels*4, offset_channels=64)
        self.offset3 = OffsetBlock(in_channels=channels*2, offset_channels=32)

        self.dres1 = DeformResBlock(in_channels=channels*8, out_channels=channels*4, offset_channels=128)
        self.dres2 = DeformResBlock(in_channels=channels*4, out_channels=channels*2, offset_channels=64)
        self.dres3 = DeformResBlock(in_channels=channels*2, out_channels=channels, offset_channels=32)
        
        self.dec_conv0 = conv2d(channels*16, channels*8)
        self.dec_conv1 = conv2d(channels*8, channels*4)
        self.dec_conv2 = conv2d(channels*4, channels*2)
        self.dec_conv3 = conv2d(channels, channels)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.leakyrelu = nn.LeakyReLU(inplace=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)

        self.conv_last = conv2d(channels, img_ch)

        
        self.to_lp_1 = sequential(nn.Linear(256, 256),
                                  nn.LeakyReLU(inplace=True),
                                  nn.Linear(256, 128),
                                  nn.LeakyReLU(inplace=True))

        self.to_lp_2 = sequential(nn.Linear(256, 256),
                                  nn.LeakyReLU(inplace=True),
                                  nn.Linear(256, 64),
                                  nn.LeakyReLU(inplace=True))

        self.to_lp_3 = sequential(nn.Linear(256, 256),
                                  nn.LeakyReLU(inplace=True),
                                  nn.Linear(256, 32),
                                  nn.LeakyReLU(inplace=True))


        
    def forward(self, input, latent_c):
        
        resized_c_init = latent_c.reshape(-1, 1, 16, 16)
        resized_c = nn.Upsample(size=(input.shape[2], input.shape[3]), mode='bilinear', align_corners=True)(resized_c_init)
        x = torch.cat([input, resized_c], dim=1)
        x = self.conv0(x)

        lp_1 = self.to_lp_1(latent_c)
        lp_2 = self.to_lp_2(latent_c)
        lp_3 = self.to_lp_3(latent_c)

        x1 = self.res1(x) # x1 :  B, channels*2, H, W
        x2 = self.leakyrelu(self.avgpool(x1)) # x2 :  B, channels*2, H/2, W/2
        
        x3 = self.res2(x2) #  x3 :  B, channels*4, H/2, W/2
        x4 = self.leakyrelu(self.avgpool(x3)) #  x4 :  B, channels*4, H/4, W/4
        
        x5 = self.res3(x4) #  x5 :  B, channels*8, H/4, W/4
        x6 = self.leakyrelu(self.avgpool(x5)) #  x6 :  B, channels*8, H/8, W/8
        
        dec_in = self.res4(x6) # dec_in : B, channels*8, H/8, W/8
        dec_in = nn.Upsample(size=(x5.shape[2], x5.shape[3]), mode='bilinear', align_corners=True)(dec_in)

        dec_in = self.dec_conv0(torch.cat([x5, dec_in], 1)) # dec_in : B, channels*16, H/4, W/4
        offset1 = self.offset1(dec_in, lp=lp_1) # offset1 : B, offset_channels, H/4, W/4
        up1 = self.dres1(dec_in, offset1) # up1 : B, channels*4, H/4, W/4
        up1 = nn.Upsample(size=(x3.shape[2], x3.shape[3]), mode='bilinear', align_corners=True)(up1)
        
        up2 = self.dec_conv1(torch.cat([x3, up1], 1)) ## up2 : B, channels*4, H/2, W/2
        offset2 = self.offset2(up2, offset1, lp=lp_2) # offset2 : B, offset_channels, H/2, W/2
        up2 = self.dres2(up2, offset2) ## up2 : B, channels*2, H/2, W/2
        up2 = nn.Upsample(size=(x1.shape[2], x1.shape[3]), mode='bilinear', align_corners=True)(up2)
        
        up3 = self.dec_conv2(torch.cat([x1, up2], 1)) # up3 : B, channels*2, H, W
        offset3 = self.offset3(up3, offset2, lp=lp_3) # offset3 : B, offset_channels, H, W
        up3 = self.dres3(up3, offset3) # up3 : B, channels, H, W
        
        return self.conv_last(up3)

class Discriminator(nn.Module):
    def __init__(self, channels=64, mode='gray'):
        super(Discriminator, self).__init__()
        
        img_ch = 1 if mode == 'gray' else 3
        
        self.model = nn.Sequential(
            
            SpectralNorm(conv2d(img_ch, channels)),
            torch.nn.LeakyReLU(0.2),
            SpectralNorm(conv2d(channels, channels, stride=2)),
            torch.nn.LeakyReLU(0.2),
            
            SpectralNorm(conv2d(channels, channels*2)),
            torch.nn.LeakyReLU(0.2),
            SpectralNorm(conv2d(channels*2, channels*2, stride=2)),
            torch.nn.LeakyReLU(0.2),
            
            SpectralNorm(conv2d(channels*2, channels*4)),
            torch.nn.LeakyReLU(0.2),
            SpectralNorm(conv2d(channels*4, channels*4, stride=2)),
            torch.nn.LeakyReLU(0.2),
            
            SpectralNorm(conv2d(channels*4, channels*8)),
            torch.nn.LeakyReLU(0.2),
            SpectralNorm(conv2d(channels*8, channels*8, stride=2)),
            torch.nn.LeakyReLU(0.2),
            
            SpectralNorm(conv2d(channels*8, channels*8)),
            torch.nn.LeakyReLU(0.2),
            SpectralNorm(conv2d(channels*8, channels*8, stride=2)),
            torch.nn.LeakyReLU(0.2),
            
            SpectralNorm(conv2d(channels*8, img_ch))
        )
        
    def forward(self, input):
        return self.model(input)     

class Estimator(nn.Module):
    def __init__(self):
        super(Estimator, self).__init__()
        self.model = sequential(nn.Linear(256,256),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(256,256),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(256,1),
                                nn.LeakyReLU(inplace=True))

    def forward(self, input):
        x = self.model(input)
        return x

class DOG_VAE(nn.Module):
    def __init__(self, channels=64, mode='gray'):
        super(DOG_VAE, self).__init__()
        self._Encoder = Encoder(channels, mode)
        self._Decoder = Decoder(channels, mode)
        self._Denoiser = Denoiser(channels=128, mode=mode) if mode=='gray' else Denoiser(channels=64, mode=mode)
        self._Estimator = Estimator()
        
        self.normal = torch.distributions.normal.Normal(0., 1.)
        
    def forward(self, input, is_training=True):
                
        # re-parametrization trick
        mu, log_sigma_sq = self._Encoder(input)
        eps = self.normal.sample(log_sigma_sq.shape)
        eps = eps.cuda()
        latent_c = eps*torch.exp(log_sigma_sq / 2) + mu

        # sigma Estimators
        qf = self._Estimator(latent_c)
        
        # Decoder
        if is_training:
            dec_in = latent_c.reshape(-1, 1, 16, 16)
            y_hat = self._Decoder(dec_in)
        else: y_hat = None

        # Denoiser
        res = self._Denoiser(input, latent_c)

        denoised = input + res
        
        return mu, log_sigma_sq, y_hat, denoised, qf