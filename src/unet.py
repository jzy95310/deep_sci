"""
---
title: U-Net
summary: >
    PyTorch implementation and tutorial of U-Net model.
---

# U-Net

This is an implementation of the U-Net model from the paper,
[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597).

U-Net consists of a contracting path and an expansive path.
The contracting path is a series of convolutional layers and pooling layers,
where the resolution of the feature map gets progressively reduced.
Expansive path is a series of up-sampling layers and convolutional layers
where the resolution of the feature map gets progressively increased.

At every step in the expansive path the corresponding feature map from the contracting path
concatenated with the current feature map.

![U-Net diagram from paper](unet.png)

Here is the [training code](experiment.html) for an experiment that trains a U-Net
on [Carvana dataset](carvana.html).
"""
import torch
import torchvision.transforms.functional
from torch import nn


class DoubleConvolution(nn.Module):
    """
    ### Two $3 \times 3$ Convolution Layers

    Each step in the contraction path and expansive path have two $3 \times 3$
    convolutional layers followed by ReLU activations.
    """

    def __init__(self, in_channels: int, out_channels: int, padding: int = 1, batch_norm: bool = False, 
                 p_dropout: float = 0.0):
        """
        :param in_channels: is the number of input channels
        :param out_channels: is the number of output channels
        :param padding: is the padding size for the convolutional layers
        :param batch_norm: if True, batch normalization is applied
        :param p_dropout: is the dropout probability
        """
        super().__init__()

        # First $3 \times 3$ convolutional layer
        self.first = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding)
        self.act1 = nn.ReLU()
        # Second $3 \times 3$ convolutional layer
        self.second = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=padding)
        self.act2 = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()
        self.dropout = nn.Dropout2d(p=p_dropout)

    def forward(self, x: torch.Tensor):
        # Apply the two convolution layers and activations
        x = self.first(x)
        x = self.bn(x)
        x = self.act1(x)
        x = self.dropout(x)
        x = self.second(x)
        x = self.bn(x)
        x = self.act2(x)
        x = self.dropout(x)
        return x


class DownSample(nn.Module):
    """
    ### Down-sample

    Each step in the contracting path down-samples the feature map with
    a $2 \times 2$ max pooling layer.
    """

    def __init__(self):
        super().__init__()
        # Max pooling layer
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor):
        return self.pool(x)


class UpSample(nn.Module):
    """
    ### Up-sample

    Each step in the expansive path up-samples the feature map with
    a $2 \times 2$ up-convolution.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        # Up-convolution
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        return self.up(x)


class CropAndConcat(nn.Module):
    """
    ### Crop and Concatenate the feature map

    At every step in the expansive path the corresponding feature map from the contracting path
    concatenated with the current feature map.
    """
    def forward(self, x: torch.Tensor, contracting_x: torch.Tensor):
        """
        :param x: current feature map in the expansive path
        :param contracting_x: corresponding feature map from the contracting path
        """

        # Crop the feature map from the contracting path to the size of the current feature map
        contracting_x = torchvision.transforms.functional.center_crop(contracting_x, [x.shape[2], x.shape[3]])
        # Concatenate the feature maps
        x = torch.cat([x, contracting_x], dim=1)
        #
        return x


class UNet(nn.Module):
    """
    ## U-Net
    """
    def __init__(self, in_channels: int, out_channels: int, depth: int = 2, padding: int = 1, batch_norm: bool = False, 
                 p_dropout: float = 0.0):
        """
        :param in_channels: number of channels in the input image
        :param out_channels: number of channels in the result feature map
        :param depth: depth of the U-Net
        :param padding: padding size for the convolutional layers
        :param batch_norm: if True, batch normalization is applied
        :param p_dropout: dropout probability for the convolutional layers
        """
        super().__init__()

        # Double convolution layers for the contracting path based on the given depth.
        # The number of features gets doubled at each step starting from $64$.
        self.down_conv = nn.ModuleList([DoubleConvolution(in_channels, 64, padding=padding)])
        if depth > 1:
            self.down_conv.extend([DoubleConvolution(64 * 2 ** i, 64 * 2 ** (i + 1), padding=padding, batch_norm=batch_norm, 
                                                     p_dropout=p_dropout) for i in range(depth - 1)])
        # Down sampling layers for the contracting path
        self.down_sample = nn.ModuleList([DownSample() for _ in range(depth)])

        # The two convolution layers at the lowest resolution (the bottom of the U).
        self.middle_conv = DoubleConvolution(64 * 2 ** (depth - 1), 64 * 2 ** depth, padding=padding, batch_norm=batch_norm, 
                                             p_dropout=p_dropout)

        # Up sampling layers for the expansive path.
        # The number of features is halved with up-sampling.
        self.up_sample = nn.ModuleList([UpSample(64 * 2 ** i, 64 * 2 ** (i - 1)) for i in range(depth, 0, -1)])
        # Double convolution layers for the expansive path.
        # Their input is the concatenation of the current feature map and the feature map from the
        # contracting path. Therefore, the number of input features is double the number of features
        # from up-sampling.
        self.up_conv = nn.ModuleList([DoubleConvolution(64 * 2 ** i, 64 * 2 ** (i - 1), padding=padding, batch_norm=batch_norm, 
                                                        p_dropout=p_dropout) for i in range(depth, 0, -1)])
        # Crop and concatenate layers for the expansive path.
        self.concat = nn.ModuleList([CropAndConcat() for _ in range(depth)])
        # Final $1 \times 1$ convolution layer to produce the output
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        """
        :param x: input image
        """
        # To collect the outputs of contracting path for later concatenation with the expansive path.
        pass_through = []
        # Contracting path
        for i in range(len(self.down_conv)):
            # Two $3 \times 3$ convolutional layers
            x = self.down_conv[i](x)
            # Collect the output
            pass_through.append(x)
            # Down-sample
            x = self.down_sample[i](x)

        # Two $3 \times 3$ convolutional layers at the bottom of the U-Net
        x = self.middle_conv(x)

        # Expansive path
        for i in range(len(self.up_conv)):
            # Up-sample
            x = self.up_sample[i](x)
            # Concatenate the output of the contracting path
            x = self.concat[i](x, pass_through.pop())
            # Two $3 \times 3$ convolutional layers
            x = self.up_conv[i](x)

        # Final $1 \times 1$ convolution layer
        x = self.final_conv(x)

        # Calculate the mean of the center 4 pixels in the output feature map
        # to get the final output.
        x = torchvision.transforms.functional.center_crop(x, [2, 2])
        x = torch.mean(x, dim=(-2, -1))
        return x
    
    def initialize_weights(self, method: str = 'kaiming'):
        """
        Initialize the weights of the model.
        """
        if method == 'kaiming':
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        elif method == 'xavier':
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        else:
            raise ValueError(f'Unknown initialization method: {method}')