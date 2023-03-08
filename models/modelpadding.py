import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

'''
    Add circular convolution (using paddings) for fixing the line.
'''

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.double_conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Add zero padding to top and bottom, circular padding to left and right
        # Output will be 192x292, therefore crop it to 192x288
        
        zero_pad = F.pad(input=x, pad=(0, 0, 1, 1), mode="constant", value=0)
        circular_pad = F.pad(input=zero_pad, pad=(3, 3, 0, 0), mode="circular")
        out1 = self.double_conv1(circular_pad)
        out1 = out1[:,:,:, 2:-2]
        
        zero_pad = F.pad(input=out1, pad=(0, 0, 1, 1), mode="constant", value=0)
        circular_pad = F.pad(input=zero_pad, pad=(3, 3, 0, 0), mode="circular")
        out2 = self.double_conv2(circular_pad)
        out2 = out2[:,:,:, 2:-2]
        return out2


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def _freeze(self):
        freeze_list = [
            #self.down1, self.down2, self.down3, self.down4, self.up1, self.up2, self.up3, self.up4,
            #self.down1, self.down2, self.down3, self.down4, self.up1, self.up2, self.up3, self.up4,
            # self.inc, self.outc, self.down1, self.down2, self.up3, self.up4
        ]

        for elem in freeze_list:
            for param in elem.parameters():
                param.requires_grad = False


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4) #256
        x = self.up2(x, x3) #128
        x = self.up3(x, x2) #64
        x = self.up4(x, x1) #out
        logits = self.outc(x)
        return logits
