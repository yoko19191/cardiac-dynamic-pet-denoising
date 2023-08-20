#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn





"""
reference : 
- https://ieeexplore.ieee.org/document/9584851
"""


class CNN(nn.Module):
    """
        simple 2 layer CNN
    """
    def __init__(self ,n_chan ,chan_embed=48):
        super(CNN, self).__init__()

        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1 = nn.Conv2d(n_chan ,chan_embed ,3 ,padding=1)
        self.conv2 = nn.Conv2d(chan_embed, chan_embed, 3, padding = 1)
        self.conv3 = nn.Conv2d(chan_embed, n_chan, 1)
    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x)
        return x



class DnCNN(nn.Module):
    """
        Implementation of DnCNN
    """
    def __init__(self, channels, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
    def forward(self, x):
        out = self.dncnn(x)
        return out



class AttentionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        q = self.conv1(x)
        k = self.conv2(x)
        attn = self.softmax(torch.matmul(q, k.transpose(-1, -2)))
        out = torch.matmul(attn, k)
        return out + x


class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # if not in_channel == out_channel, using 1x1 conv2d to fix
        if in_channels != out_channels:
            self.match_channels = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.match_channels = None

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # if not in_channel == out_channel, using 1x1 conv2d to fix
        if self.match_channels:
            residual = self.match_channels(x)

        out += residual
        out = self.relu(out)
        return out


class UNet2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.encoder1 = UNetBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = UNetBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.encoder3 = UNetBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.encoder4 = UNetBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.encoder5 = UNetBlock(512, 1024)
        self.pool5 = nn.MaxPool2d(2)

        self.center = nn.Sequential(
            UNetBlock(1024, 2048),
            AttentionModule(2048, 2048)
        )

        self.up5 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.decoder5 = UNetBlock(2048, 1024)
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = UNetBlock(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = UNetBlock(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = UNetBlock(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = UNetBlock(128, 64)
        self.conv_final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        enc5 = self.encoder5(self.pool4(enc4))
        center = self.center(self.pool5(enc5))
        dec5 = self.decoder5(torch.cat([enc5, self.up5(center)], 1))
        dec4 = self.decoder4(torch.cat([enc4, self.up4(dec5)], 1))
        dec3 = self.decoder3(torch.cat([enc3, self.up3(dec4)], 1))
        dec2 = self.decoder2(torch.cat([enc2, self.up2(dec3)], 1))
        dec1 = self.decoder1(torch.cat([enc1, self.up1(dec2)], 1))
        final = self.conv_final(dec1)
        return final






##################

# from fastai2.learner import Learner
# from fastai2.vision.learner import unet_learner, unet_config
# from fastai2.vision.models import xresnet

# from metrics_and_losses import mae_loss



# def unet_25d(dls_25d, backbone=None, pretrained=False, n_in=3, n_out=1, wd=0.001, last_cross=True, blur=True,
#              self_attention=True, metrics=None, output_dir='models', loss_func=None):
#     """ 2.5D U-Net model """

#     if not backbone:
#         backbone = xresnet.xresnet34
#     if not loss_func:
#         loss_func = mae_loss

#     return unet_learner(dls=dls_25d, arch=backbone,loss_func=loss_func, pretrained=pretrained, n_in=n_in,
#                         n_out=n_out, wd=wd, normalize=False, config=unet_config(last_cross=last_cross,
#                         blur=blur, self_attention=self_attention, y_range=None), metrics=metrics,
#                         model_dir=output_dir)


# def unet_2d(dls_2d, backbone=None, pretrained=False, n_in=1, n_out=1, wd=0.001, last_cross=True, blur=True,
#              self_attention=True, metrics=None, output_dir='models', loss_func=None):
#     """ 2D U-Net model """

#     if not backbone:
#         backbone = xresnet.xresnet34
#     if not loss_func:
#         loss_func = mae_loss

#     return unet_learner(dls=dls_2d, arch=backbone,loss_func=loss_func, pretrained=pretrained, n_in=n_in,
#                         n_out=n_out, wd=wd, normalize=False, config=unet_config(last_cross=last_cross,
#                         blur=blur, self_attention=self_attention, y_range=None), metrics=metrics,
#                         model_dir=output_dir)


# class HybridModel(nn.Module):
#     """ Hybrid 2D/3D U-Net """
#     def __init__(self, model_2d, model_3d):
#         super(HybridModel, self).__init__()
#         self.model_2d = model_2d
#         self.model_3d = model_3d

#     def forward(self, X):
#         # Model 2D
#         X2d_in = X.permute(0, 2, 1, 3, 4)
#         X2d_out = []
#         for i in range(X2d_in.shape[0]):
#             X2d_out.append(self.model_2d(X2d_in[i]).permute(1, 0, 2, 3).unsqueeze(0))
#         X2d_out = torch.cat(X2d_out)

#         # Model 3D
#         X3d_in = torch.cat((X, X2d_out), dim=1)
#         X3d_out = self.model_3d(X3d_in)

#         return X3d_out

#     def predict(self, X, *args, **kwargs):
#         return self.forward(X)


# class HybridModelx3(nn.Module):
#     """ Hybrid 2.5D/3D U-Net """
#     def __init__(self, model_25d, model_3d):
#         super(HybridModelx3, self).__init__()
#         self.model_25d = model_25d
#         self.model_3d = model_3d

#     def forward(self, X):
#         # Model 2.5D
#         X25d_batches = []
#         for b in range(X.shape[0]):
#             X25d_vol = []
#             for plane in range(X.shape[-3]):
#                 # start edge
#                 if plane == 0:
#                     X25d_in = torch.cat([X[b, 0, :1, :, :], X[b, 0, :2, :, :]], axis=-3)
#                 # end edge
#                 elif plane == X.shape[-3] - 1:
#                     X25d_in = torch.cat([X[b, 0, -1:, :, :], X[b, 0, -2:, :, :]], axis=-3)
#                 else:
#                     X25d_in = X[b, 0, plane - 1:plane + 2, :, :]

#                 X25d_vol.append(self.model_25d(X25d_in.unsqueeze(0)))
#             X25d_batches.append(torch.cat(X25d_vol).unsqueeze(0))
#         X25d_out = torch.cat(X25d_batches).permute(0, 2, 1, 3, 4)

#         # Model 3D
#         X3d_in = torch.cat((X, X25d_out), dim=1)
#         X3d_out = self.model_3d(X3d_in)

#         return X3d_out

#     def predict(self, X, *args, **kwargs):
#         return self.forward(X)


