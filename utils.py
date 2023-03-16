from models.modelpadding import UNet
from models.model_plusplus2 import NestedUNet2
from models.model_plusplus2_dropout import NestedUNet2 as DropoutUNetpp
from serial_experiments.models.resnext_model import custom_resnext, custom_resnext_pretrained, custom_resnext2, custom_resnext3, custom_resnext4
from serial_experiments.models.seg_models import UNetPlusPlus, PSPNet, DeepLabV3, UNetSE, UNetJ
from models.model_bayesian import BayesianUNetPP
from torch.utils.data import Dataset
from models.model_tcn import TCN
import numpy as np
import torch

def generate_PE(size_x, size_y):
    '''
        Sinusoidal PE, circular
        return 192x288 PE
        k = d//2
        depth = 192 (d)
        position = 288 (t)
    '''
    
    depth = size_y # 192
    t = size_x # 288
    pos_encoding_matrix = np.zeros((size_y, size_x))
    wk = np.asarray([1 / (10000**(2*k/depth)) for k in range(depth//2)])
    odds = np.arange(1, 192, 2)
    evens = np.arange(0, 192, 2)
    
    for pos in range(t):
        if pos <= 144:
            sink = np.sin(wk*pos)
            cosk = np.cos(wk*pos)
            pos_encoding_matrix[odds, pos] = cosk*pos
            pos_encoding_matrix[evens, pos] = sink*pos
        if pos > 144:
            sink = np.sin(wk*(288-pos))
            cosk = np.cos(wk*(288-pos))
            pos_encoding_matrix[odds, pos] = cosk*(288-pos)
            pos_encoding_matrix[evens, pos] = sink*(288-pos)
    
    return pos_encoding_matrix


def model_select(model_name, input_size, output_size, device_name):

    if model_name == "CMIP6-resnext-2":
        model = custom_resnext2(input_size + 1)
    elif model_name == "CMIP6-resnext-3":
        model = custom_resnext3(input_size + 1)
    elif model_name == "CMIP6-resnext-4":
        model = custom_resnext4(input_size + 1)
    elif model_name == "CMIP6-pretrained-resnext":
        model = custom_resnext_pretrained(input_size + 1)
    elif model_name == "CMIP6-UNetPlusPlus":
        model = UNetPlusPlus(input_size + 1, "resnext50_32x4d", None)
    elif model_name == "PSPNet":
        model = PSPNet(input_size + 1)
    elif model_name == "DeepLabV3":
        model = DeepLabV3(input_size + 1)
    elif model_name == "UNetSE":
        model = UNetSE(input_size + 1)
    elif model_name == "UNetPlusPlusSE":
        model = UNetPlusPlus(input_size + 1, "se_resnext50_32x4d", None)
    elif model_name == "CMIP6-UNet-AttentionSE":
        model = UNetJ(input_size + 1, "se_resnext50_32x4d", "scse")
    elif model_name == "CMIP6-UNet-Attention-withoutDA":
        model = UNetJ(input_size, "se_resnext50_32x4d", "scse")
    elif model_name == "CMIP6-UNet-AttentionSE-withoutDA":
        model = UNetJ(input_size+1, "se_resnext50_32x4d", "scse")
    elif model_name == "CMIP6-UNetPlusPlus-AttentionSE":
        model = UNetPlusPlus(input_size + 1, "se_resnext50_32x4d", "scse")
    elif model_name == "CMIP6-withoutDA":
        model = UNet(n_channels=input_size, n_classes=output_size)
    elif model_name == "CMIP6-differentDA":
        model = UNet(n_channels=input_size + 1, n_classes=output_size)
    elif model_name == "CMIP6-AttentionUNet":
        model = AttentionUNet(input_channels=input_size + 1, num_classes=output_size)
    elif model_name == "CMIP6-NestedUNet":
        model = NestedUNet(input_channels=input_size + 1, num_classes=output_size)
    elif model_name == "CMIP6-NestedUNet2":
        model = NestedUNet2(input_channels=input_size + 1, num_classes=output_size)
    elif model_name == "CMIP6-TCN":
        channel_sizes = [32, 24, 16, 8, 1]
        model = TCN(num_channels=channel_sizes, input_size = input_size+1, output_size=output_size, kernel_size=3, dropout=0.0)
    elif model_name == "CMIP6-BayesianUNetPP":
        model = BayesianUNetPP(input_channels=input_size + 1, num_classes=output_size, device=device_name)
    elif "dropout" in model_name.lower():
        model = DropoutUNetpp(input_channels=input_size + 1, num_classes=output_size, device=device_name)
    elif model_name == "CMIP6-Ensemble":
        model = NestedUNet2(input_channels=input_size + 1, num_classes=2, device=device_name)
    else:
        model = UNet(n_channels=input_size + 1, n_classes=output_size)
    return model

def nll_criterion(logsigma2, mean, ys):
    sigma2 = torch.exp(logsigma2)
    return torch.mean(0.5*torch.log(sigma2) + 0.5*torch.div(torch.square(ys-mean), sigma2)) + 5
