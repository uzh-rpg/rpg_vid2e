# Taken from https://github.com/avinashpaliwal/Super-SloMo/blob/bbf0375958d66dab48143166a5b80cd26a406458/model.py

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class down(nn.Module):
    """
    A class for creating neural network blocks containing layers:

    Average Pooling --> Convlution + Leaky ReLU --> Convolution + Leaky ReLU

    This is used in the UNet Class to create a UNet like NN architecture.

    ...

    Methods
    -------
    forward(x)
        Returns output tensor after passing input `x` to the neural network
        block.
    """

    def __init__(self, inChannels, outChannels, filterSize):
        """
        Parameters
        ----------
            inChannels : int
                number of input channels for the first convolutional layer.
            outChannels : int
                number of output channels for the first convolutional layer.
                This is also used as input and output channels for the
                second convolutional layer.
            filterSize : int
                filter size for the convolution filter. input N would create
                a N x N filter.
        """


        super(down, self).__init__()
        # Initialize convolutional layers.
        self.conv1 = nn.Conv2d(inChannels,  outChannels, filterSize, stride=1, padding=int((filterSize - 1) / 2))
        self.conv2 = nn.Conv2d(outChannels, outChannels, filterSize, stride=1, padding=int((filterSize - 1) / 2))

    def forward(self, x):
        """
        Returns output tensor after passing input `x` to the neural network
        block.

        Parameters
        ----------
            x : tensor
                input to the NN block.

        Returns
        -------
            tensor
                output of the NN block.
        """


        # Average pooling with kernel size 2 (2 x 2).
        x = F.avg_pool2d(x, 2)
        # Convolution + Leaky ReLU
        x = F.leaky_relu(self.conv1(x), negative_slope = 0.1)
        # Convolution + Leaky ReLU
        x = F.leaky_relu(self.conv2(x), negative_slope = 0.1)
        return x


class up(nn.Module):
    """
    A class for creating neural network blocks containing layers:

    Bilinear interpolation --> Convlution + Leaky ReLU --> Convolution + Leaky ReLU

    This is used in the UNet Class to create a UNet like NN architecture.

    ...

    Methods
    -------
    forward(x, skpCn)
        Returns output tensor after passing input `x` to the neural network
        block.
    """

    def __init__(self, inChannels, outChannels):
        """
        Parameters
        ----------
            inChannels : int
                number of input channels for the first convolutional layer.
            outChannels : int
                number of output channels for the first convolutional layer.
                This is also used for setting input and output channels for
                the second convolutional layer.
        """

        super(up, self).__init__()
        # Initialize convolutional layers.
        self.conv1 = nn.Conv2d(inChannels,  outChannels, 3, stride=1, padding=1)
        # (2 * outChannels) is used for accommodating skip connection.
        self.conv2 = nn.Conv2d(2 * outChannels, outChannels, 3, stride=1, padding=1)

    def forward(self, x, skpCn):
        """
        Returns output tensor after passing input `x` to the neural network
        block.

        Parameters
        ----------
            x : tensor
                input to the NN block.
            skpCn : tensor
                skip connection input to the NN block.

        Returns
        -------
            tensor
                output of the NN block.
        """

        # Bilinear interpolation with scaling 2.
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        # Convolution + Leaky ReLU
        x = F.leaky_relu(self.conv1(x), negative_slope = 0.1)
        # Convolution + Leaky ReLU on (`x`, `skpCn`)
        x = F.leaky_relu(self.conv2(torch.cat((x, skpCn), 1)), negative_slope = 0.1)
        return x



class UNet(nn.Module):
    """
    A class for creating UNet like architecture as specified by the
    Super SloMo paper.

    ...

    Methods
    -------
    forward(x)
        Returns output tensor after passing input `x` to the neural network
        block.
    """


    def __init__(self, inChannels, outChannels):
        """
        Parameters
        ----------
            inChannels : int
                number of input channels for the UNet.
            outChannels : int
                number of output channels for the UNet.
        """

        super(UNet, self).__init__()
        # Initialize neural network blocks.
        self.conv1 = nn.Conv2d(inChannels, 32, 7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(32, 32, 7, stride=1, padding=3)
        self.down1 = down(32, 64, 5)
        self.down2 = down(64, 128, 3)
        self.down3 = down(128, 256, 3)
        self.down4 = down(256, 512, 3)
        self.down5 = down(512, 512, 3)
        self.up1   = up(512, 512)
        self.up2   = up(512, 256)
        self.up3   = up(256, 128)
        self.up4   = up(128, 64)
        self.up5   = up(64, 32)
        self.conv3 = nn.Conv2d(32, outChannels, 3, stride=1, padding=1)

    def forward(self, x):
        """
        Returns output tensor after passing input `x` to the neural network.

        Parameters
        ----------
            x : tensor
                input to the UNet.

        Returns
        -------
            tensor
                output of the UNet.
        """


        x  = F.leaky_relu(self.conv1(x), negative_slope = 0.1)
        s1 = F.leaky_relu(self.conv2(x), negative_slope = 0.1)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        s5 = self.down4(s4)
        x  = self.down5(s5)
        x  = self.up1(x, s5)
        x  = self.up2(x, s4)
        x  = self.up3(x, s3)
        x  = self.up4(x, s2)
        x  = self.up5(x, s1)
        x  = F.leaky_relu(self.conv3(x), negative_slope = 0.1)
        return x


class backWarp(nn.Module):
    """
    A class for creating a backwarping object.

    This is used for backwarping to an image:

    Given optical flow from frame I0 to I1 --> F_0_1 and frame I1, 
    it generates I0 <-- backwarp(F_0_1, I1).

    ...

    Methods
    -------
    forward(x)
        Returns output tensor after passing input `img` and `flow` to the backwarping
        block.
    """

    def __init__(self, W, H, device):
        """
        Parameters
        ----------
            W : int
                width of the image.
            H : int
                height of the image.
            device : device
                computation device (cpu/cuda). 
        """


        super(backWarp, self).__init__()
        # create a grid
        gridX, gridY = np.meshgrid(np.arange(W), np.arange(H))
        self.W = W
        self.H = H
        self.gridX = torch.tensor(gridX, requires_grad=False, device=device)
        self.gridY = torch.tensor(gridY, requires_grad=False, device=device)

    def forward(self, img, flow):
        """
        Returns output tensor after passing input `img` and `flow` to the backwarping
        block.
        I0  = backwarp(I1, F_0_1)

        Parameters
        ----------
            img : tensor
                frame I1.
            flow : tensor
                optical flow from I0 and I1: F_0_1.

        Returns
        -------
            tensor
                frame I0.
        """


        # Extract horizontal and vertical flows.
        u = flow[:, 0, :, :]
        v = flow[:, 1, :, :]
        x = self.gridX.unsqueeze(0).expand_as(u).float() + u
        y = self.gridY.unsqueeze(0).expand_as(v).float() + v
        # range -1 to 1
        x = 2*(x/self.W - 0.5)
        y = 2*(y/self.H - 0.5)
        # stacking X and Y
        grid = torch.stack((x,y), dim=3)
        # Sample pixels using bilinear interpolation.
        imgOut = torch.nn.functional.grid_sample(img, grid, align_corners=True)
        return imgOut
