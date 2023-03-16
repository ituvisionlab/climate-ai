import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

def KL_DIV(mu_q, sig_q, mu_p, sig_p):
    kl = 0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).sum()
    return kl

class ModuleWrapper(nn.Module):
    """Wrapper for nn.Module with support for arbitrary flags and a universal forward pass"""

    def __init__(self):
        super(ModuleWrapper, self).__init__()

    def set_flag(self, flag_name, value):
        setattr(self, flag_name, value)
        for m in self.children():
            if hasattr(m, 'set_flag'):
                m.set_flag(flag_name, value)

    def forward(self, x):
        for module in self.children():
            x = module(x)

        kl = 0.0
        for module in self.modules():
            if hasattr(module, 'kl_loss'):
                kl = kl + module.kl_loss()

        return x, kl

class BayesianConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True):
        super(BayesianConv2d, self).__init__()

        self.kernel_size = (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.use_bias = bias
        self.groups = 1

        self.priors = {
                'prior_mu': 0,
                'prior_sigma': 0.1,
                'posterior_mu_initial': (0, 0.1),
                'posterior_sigma2_initial': (-3, 0.1),
            }

        self.W_bias = nn.Parameter(torch.Tensor(out_channels))
        self.W_mu = nn.Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        self.W_prior_mu = nn.Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size), requires_grad=False)
        self.W_prior_mu.data.zero_()

        self.W_logsigma2 = nn.Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        self.W_prior_logsigma2 = nn.Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size), requires_grad=False)
        self.W_prior_logsigma2.data.zero_()

        if self.use_bias:
            self.bias_mu = nn.Parameter(torch.empty((out_channels)))
            self.bias_logsigma2 = nn.Parameter(torch.empty((out_channels)))

    def reset_parameters(self):
        stdv = 1.0 / np.sqrt(self.W_mu.size(1))
        self.W_mu.data.normal_(0, stdv)
        self.W_logsigma2.data.zero_().normal_(-9, 0.001)  # var init via Louizos
        self.W_bias.data.zero_()

    def forward(self, input, sample=True):
        if self.training or sample:
            W_eps = torch.empty(self.W_mu.size()).normal_(0, 1).to('cuda:1')
            self.W_sigma2 = torch.log1p(torch.exp(self.W_logsigma2))
            weight = self.W_mu + W_eps * self.W_sigma2.sqrt()

            if self.use_bias:
                bias_eps = torch.empty(self.bias_mu.size()).normal_(0, 1).to('cuda:1')
                self.bias_sigma2 = torch.log1p(torch.exp(self.bias_logsigma2))
                bias = self.bias_mu + bias_eps * self.bias_sigma2.sqrt()
            else:
                bias = None
        else:
            weight = self.W_mu
            bias = self.bias_mu if self.use_bias else None

        return F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)

    def KL(self):
        return torch.distributions.kl.kl_divergence(Normal(self.W_mu, self.W_logsigma2.exp().sqrt()), Normal(self.W_prior_mu, self.W_prior_logsigma2.exp().sqrt())).sum()

class BBBConv2d(ModuleWrapper):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=1, dilation=1, bias=True, priors=None, device='cuda:1'):

        super(BBBConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = 1
        self.use_bias = bias
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        if priors is None:
            priors = {
                'prior_mu': 0,
                'prior_sigma': 0.1,
                'posterior_mu_initial': (0, 0.1),
                'posterior_rho_initial': (-3, 0.1),
            }
        self.prior_mu = priors['prior_mu']
        self.prior_sigma = priors['prior_sigma']
        self.posterior_mu_initial = priors['posterior_mu_initial']
        self.posterior_rho_initial = priors['posterior_rho_initial']

        self.W_mu = nn.Parameter(torch.empty((out_channels, in_channels, *self.kernel_size), device=self.device))
        self.W_rho = nn.Parameter(torch.empty((out_channels, in_channels, *self.kernel_size), device=self.device))

        if self.use_bias:
            self.bias_mu = nn.Parameter(torch.empty((out_channels), device=self.device))
            self.bias_rho = nn.Parameter(torch.empty((out_channels), device=self.device))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.W_mu.data.normal_(*self.posterior_mu_initial)
        self.W_rho.data.normal_(*self.posterior_rho_initial)

        if self.use_bias:
            self.bias_mu.data.normal_(*self.posterior_mu_initial)
            self.bias_rho.data.normal_(*self.posterior_rho_initial)

    def forward(self, input, sample=True):
        if self.training or sample:
            W_eps = torch.empty(self.W_mu.size()).normal_(0, 1).to(self.device)
            self.W_sigma = torch.log1p(torch.exp(self.W_rho))
            weight = self.W_mu + W_eps * self.W_sigma

            if self.use_bias:
                bias_eps = torch.empty(self.bias_mu.size()).normal_(0, 1).to(self.device)
                self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
                bias = self.bias_mu + bias_eps * self.bias_sigma
            else:
                bias = None
        else:
            weight = self.W_mu
            bias = self.bias_mu if self.use_bias else None

        return F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)

    def KL(self):
        kl = KL_DIV(self.prior_mu, self.prior_sigma, self.W_mu, self.W_sigma)
        if self.use_bias:
            kl += KL_DIV(self.prior_mu, self.prior_sigma, self.bias_mu, self.bias_sigma)
        return kl

class DoubleConv(nn.Module):

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



class BayesianLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BayesianLinear, self).__init__()

        self.a = nn.Parameter(torch.Tensor(55000, 55000))
        
        # self.bias = nn.Parameter(torch.Tensor(out_channels))
        # self.mu = nn.Parameter(torch.Tensor(out_channels, in_channels))
        # self.prior_mu = nn.Parameter(torch.Tensor(out_channels, in_channels), requires_grad=False)
        # self.prior_mu.data.zero_()

        # self.logsigma2 = nn.Parameter(torch.Tensor(out_channels, in_channels))
        # self.prior_logsigma2 = nn.Parameter(torch.Tensor(out_channels, in_channels), requires_grad=False)
        # self.prior_logsigma2.data.zero_()

        # self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1.0 / np.sqrt(self.mu.size(1))
        self.mu.data.normal_(0, stdv)
        self.logsigma2.data.zero_().normal_(-9, 0.001)  # var init via Louizos
        self.bias.data.zero_()

    def forward(self, input):
        mu_out = torch.nn.functional.linear(input, self.mu, self.bias)
        logsigma2 = self.logsigma2
        sigma2 = logsigma2.exp()
        var_out = torch.nn.functional.linear(input.pow(2), sigma2) + 1e-8
        return mu_out + var_out.sqrt() * torch.randn_like(mu_out)

    def KL(self):
        return torch.distributions.kl.kl_divergence(Normal(self.mu, self.logsigma2.exp().sqrt()), Normal(self.prior_mu, self.prior_logsigma2.exp().sqrt()))



class BayesianUNetPP(nn.Module):
    def __init__(self, num_classes, input_channels=3, device='cuda:1'):
        super().__init__()

        nb_filter = [64, 128, 256, 512]
        self.device = device

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = DoubleConv(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = DoubleConv(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = DoubleConv(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = DoubleConv(nb_filter[2], nb_filter[3], nb_filter[3])

        self.conv0_1 = DoubleConv(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = DoubleConv(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = DoubleConv(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_2 = DoubleConv(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = DoubleConv(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_3 = DoubleConv(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = BBBConv2d(nb_filter[0], 1, kernel_size=3, device=self.device)
        #self.fcn = BayesianLinear(1*192*288, 192*288)

    def forward(self, input):
        x0_0 = self.conv0_0(input) #down
        x1_0 = self.conv1_0(self.pool(x0_0)) #down
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1)) # inter

        x2_0 = self.conv2_0(self.pool(x1_0)) #down
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1)) #inter
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1)) #inter

        x3_0 = self.conv3_0(self.pool(x2_0)) #down 
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1)) #up
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1)) #up
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1)) #up

        output = self.final(x0_3)
        #output = self.fcn(output)
        
        return output

    # def freeze_convs(self):
    #     for param in self.parameters():
    #          param.requires_grad_ = False
    #     self.fcn.requires_grad_ = True

    def KL_loss(self):
        return self.final.KL().sum()
        