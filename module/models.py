#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models



########### Res34-UNet ###########



"""
reference : 
- https://ieeexplore.ieee.org/document/9584851
"""


class CNN(nn.Module):
    """
        A CNN with a variable number of layers
    """
    def __init__(self, n_chan, chan_embed=48, num_hidden_layers=2):
        super(CNN, self).__init__()
        
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        # Input layer
        self.input_layer = nn.Conv2d(n_chan, chan_embed, 3, padding=1)
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_hidden_layers):
            self.hidden_layers.append(nn.Conv2d(chan_embed, chan_embed, 3, padding=1))
        
        # Output layer
        self.output_layer = nn.Conv2d(chan_embed, n_chan, 1)

    def forward(self, x):
        x = self.act(self.input_layer(x))
        
        # Pass through hidden layers
        for layer in self.hidden_layers:
            x = self.act(layer(x))
        
        x = self.output_layer(x)
        return x



class DnCNN(nn.Module):
    """
    Implementation of 2D DnCNN
    """
    def __init__(self, in_channels=1, out_channels=1, num_layers=17, features=64):
        super(DnCNN, self).__init__()

        layers = []
        
        layers.append(nn.Conv2d(in_channels, features, kernel_size=3, padding=1, stride=1))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(features, features, kernel_size=3, padding=1, stride=1))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(features, out_channels, kernel_size=3, padding=1, stride=1))

        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        return out



class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),  # Up-sampling to [B, 64, 96, 96]
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1) # Output layer
        )

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.middle(x1)
        x3 = self.decoder(x2)
        return x3


class AttentionModule(nn.Module):
    """
        Attention module to provide self-attention within a channel dimension.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
    """
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
    """
    Basic UNet block comprising two convolutional layers with batch normalization and ReLU activation.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """
    def __init__(self, in_channels, out_channels):
        super(UNetBlock, self).__init__()
        self.in_channels = in_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        #self.relu = nn.ReLU(inplace=True)
        self.leaky_relu = nn.LeakyReLU(inplace=True)

        # if not in_channel == out_channel, using 1x1 conv2d to fix
        if in_channels != out_channels:
            self.match_channels = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.match_channels = None

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        #out = self.relu(out)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # if not in_channel == out_channel, using 1x1 conv2d to fix
        if self.match_channels:
            residual = self.match_channels(x)

        out += residual
        #out = self.relu(out)
        out = self.leaky_relu(out)
        return out


class UNet2_5D(nn.Module):
    """
         2.5D UNet architecture with self-attention at the bottleneck.
        Takes three consecutive slices (top, middle, bottom) as input and predicts the middle slice.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels (usually number of classes).
    """
    def __init__(self, in_channels, out_channels):
        super(UNet2_5D, self).__init__()
        # Multiply in_channels by 3 as there are 3 slices: top, middle, bottom
        #self.encoder1 = UNetBlock(in_channels * 3, 64)
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

    def forward(self, x_top, x_middle, x_bottom):
        """
        Forward pass for the 2.5D UNet.

        Args:
            x_top (torch.Tensor): Input tensor for the top slice (batch_size, in_channels, height, width).
            x_middle (torch.Tensor): Input tensor for the middle slice (batch_size, in_channels, height, width).
            x_bottom (torch.Tensor): Input tensor for the bottom slice (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor for the middle slice (batch_size, out_channels, height, width).
        """
         # Concatenate the channels from the top, middle, and bottom slices
        x = torch.cat([x_top, x_middle, x_bottom], dim=1)
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


###### 2D U-Net #########
class UNet2D(nn.Module):
    """
    2D UNet architecture.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels (usually number of classes).
    """
    def __init__(self, in_channels, out_channels):
        super(UNet2D, self).__init__()

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

    def forward(self, x_middle):
        """
        Forward pass for the 2D UNet.

        Args:
            x_middle (torch.Tensor): Input tensor (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor (batch_size, out_channels, height, width).
        """
        enc1 = self.encoder1(x_middle)
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



###### ResNet-34 2.5D U-Net #########


class ResNet34_UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNet34_UNet, self).__init__()
        
        # Load pre-trained ResNet34 model + higher level features
        resnet34 = models.resnet34(pretrained=True)
        
        # Encoder layers
        self.enc1 = nn.Sequential(*list(resnet34.children())[:3])
        self.enc2 = nn.Sequential(*list(resnet34.children())[4])
        self.enc3 = nn.Sequential(*list(resnet34.children())[5])
        self.enc4 = nn.Sequential(*list(resnet34.children())[6])
        self.enc5 = nn.Sequential(*list(resnet34.children())[7])
        
        self.center = nn.Sequential(
            UNetBlock(512, 1024),
            AttentionModule(1024, 1024)
        )

        # Decoder layers
        self.up5 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec5 = UNetBlock(1024, 512)
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = UNetBlock(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = UNetBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = UNetBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = UNetBlock(128, 64)
        
        self.conv_final = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, x_top, x_middle, x_bottom):
        # Encoder path
        x = torch.cat([x_top, x_middle, x_bottom], dim=1)
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)
        
        center = self.center(enc5)
        
        # Decoder path
        dec5 = self.dec5(torch.cat([F.interpolate(enc5, size=self.up5(center).size()[2:], mode='bilinear', align_corners=True), self.up5(center)], 1))
        dec4 = self.dec4(torch.cat([F.interpolate(enc4, size=self.up4(dec5).size()[2:], mode='bilinear', align_corners=True), self.up4(dec5)], 1))
        dec3 = self.dec3(torch.cat([F.interpolate(enc3, size=self.up3(dec4).size()[2:], mode='bilinear', align_corners=True), self.up3(dec4)], 1))
        dec2 = self.dec2(torch.cat([F.interpolate(enc2, size=self.up2(dec3).size()[2:], mode='bilinear', align_corners=True), self.up2(dec3)], 1))
        dec1 = self.dec1(torch.cat([F.interpolate(enc1, size=self.up1(dec2).size()[2:], mode='bilinear', align_corners=True), self.up1(dec2)], 1))

        final = self.conv_final(dec1)
        
        # Ensure the output has the same size as the input (192x192)
        return F.interpolate(final, size=(192, 192), mode='bilinear', align_corners=True)



# class UNet2_5D(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(UNet2_5D, self).__init__()

#         self.encoder1 = UNetBlock(in_channels, 64)
#         self.pool1 = nn.MaxPool2d(2)
#         self.encoder2 = UNetBlock(64, 128)
#         self.pool2 = nn.MaxPool2d(2)
#         self.encoder3 = UNetBlock(128, 256)
#         self.pool3 = nn.MaxPool2d(2)
#         self.encoder4 = UNetBlock(256, 512)
#         self.pool4 = nn.MaxPool2d(2)
#         self.encoder5 = UNetBlock(512, 1024)
#         self.pool5 = nn.MaxPool2d(2)

#         self.center = UNetBlock(1024, 2048)  # Removed the AttentionModule

#         self.up5 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
#         self.decoder5 = UNetBlock(2048, 1024)
#         self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
#         self.decoder4 = UNetBlock(1024, 512)
#         self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
#         self.decoder3 = UNetBlock(512, 256)
#         self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
#         self.decoder2 = UNetBlock(256, 128)
#         self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
#         self.decoder1 = UNetBlock(128, 64)
#         self.conv_final = nn.Conv2d(64, out_channels, kernel_size=1)

#     def forward(self, x_top, x_middle, x_bottom):
#         x = torch.cat([x_top, x_middle, x_bottom], dim=1)

#         enc1 = self.encoder1(x)
#         enc2 = self.encoder2(self.pool1(enc1))
#         enc3 = self.encoder3(self.pool2(enc2))
#         enc4 = self.encoder4(self.pool3(enc3))
#         enc5 = self.encoder5(self.pool4(enc4))

#         center = self.center(self.pool5(enc5))

#         dec5 = self.decoder5(torch.cat([enc5, self.up5(center)], 1))
#         dec4 = self.decoder4(torch.cat([enc4, self.up4(dec5)], 1))
#         dec3 = self.decoder3(torch.cat([enc3, self.up3(dec4)], 1))
#         dec2 = self.decoder2(torch.cat([enc2, self.up2(dec3)], 1))
#         dec1 = self.decoder1(torch.cat([enc1, self.up1(dec2)], 1))
#         final = self.conv_final(dec1)

#         return final


