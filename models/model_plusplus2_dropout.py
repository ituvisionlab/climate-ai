import torch
import torch.nn as nn
import torch.nn.functional as F

class VGGBlock(nn.Module):
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


class NestedUNet2(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.deep_supervision = deep_supervision

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            
        #self.dropout = nn.Dropout(0.7)
        
        self.training = True

    def forward(self, input):
        p = 0.4
        x0_0 = self.conv0_0(input) #down
        x1_0 = self.conv1_0(self.pool(x0_0)) #down
        x0_1 = F.dropout(self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1)), p=p) # inter

        x2_0 = self.conv2_0(self.pool(x1_0)) #down
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))#inter
        x0_2 = F.dropout(self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1)), p=p) #inter

        x3_0 = self.conv3_0(self.pool(x2_0)) #down 
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))#up
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1)) #up
        x0_3 = F.dropout(self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))) #up

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            return [output1, output2, output3]

        else:
            output = self.final(x0_3)
            return output
        