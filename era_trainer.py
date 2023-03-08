'''
    Used for finetune purposes
'''

import torch
import numpy as np
import xarray as xr

import argparse
import os

import time
import matplotlib.pyplot as plt

from modelpadding import UNet
from model_plusplus import NestedUNet
from model_plusplus2 import NestedUNet2
from models.resnext_model import custom_resnext, custom_resnext_pretrained, custom_resnext2
from torch.utils.data import Dataset
from model_bayesian import BayesianUNetPP


import warnings

warnings.filterwarnings("ignore")

def save_model(state_dict, train_losses, val_losses, epoch):
    torch.save(
        {
            "state_dict": state_dict,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "epoch": epoch,
        },
        os.path.join(save_path, "best.pth"),
    )
    
def save_model2(state_dict, train_losses, val_losses, epoch, name):
    torch.save(
        {
            "state_dict": state_dict,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "epoch": epoch,
        },
        os.path.join(save_path, name+".pth"),
    )

def save_model(state_dict, train_losses, val_losses, epoch):
    torch.save(
        {
            "state_dict": state_dict,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "epoch": epoch,
        },
        os.path.join(save_path, "best.pth"),
    )


def save_plot(train_losses, val_losses, epoch):

    _, ax = plt.subplots()

    ax.plot(train_losses, label="train")
    ax.plot(val_losses, label="val")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()

    plt.savefig(os.path.join(loss_path, "loss_%d.png" % epoch), dpi=300)
    plt.cla()

def save_plot(train_losses, val_losses, loss_type):

    _, ax = plt.subplots()

    ax.plot(train_losses, label="train")
    ax.plot(val_losses, label="val")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()

    plt.savefig(os.path.join(loss_path, "loss_" + loss_type + ".png"), dpi=300)
    plt.cla()

    
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

PE = generate_PE(288,192)

plot_freq = 1
prereport_freq = 100

# From the user take the years and months as integers with argparse
parser = argparse.ArgumentParser()
parser.add_argument("--years", type=int, default=1)
parser.add_argument("--months", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--prediction_month", type=int, default=1)
parser.add_argument("--positional_encoding", type=bool, default=False)
parser.add_argument("--template_path", type=str)
parser.add_argument("--base_model", type=str)

# python trainer.py --years 2 --months 2 --batch_size 32 --epochs 200

years = parser.parse_args().years
months = parser.parse_args().months
batch_size = parser.parse_args().batch_size
epochs = parser.parse_args().epochs
device_name = parser.parse_args().device
prediction_month = parser.parse_args().prediction_month
positional_encoding = parser.parse_args().positional_encoding
template_path = parser.parse_args().template_path
base_model = parser.parse_args().base_model

#model_weight_path = "results/saves/year-%d-month-%d-+%d/best.pth" % (years, months, prediction_month)
model_weight_path = base_model+"/weights/CIMP6_year-%d-month-%d-+%d/best.pth" % (years, months, prediction_month)

main_path = template_path

loss_path = template_path + "/plots/CIMP6_year-%d-month-%d-predmont-%d" % (years, months, prediction_month)
save_path = template_path + "/weights/CIMP6_year-%d-month-%d-predmont-%d" % (years, months, prediction_month)

# Create directories if they don't exist
if not os.path.exists(loss_path):
    os.makedirs(loss_path)
if not os.path.exists(save_path):
    os.makedirs(save_path)

else:
    print("DIRECTORY ALREADY EXISTS, continuing")
    time.sleep(1)

######## DATA #################################################################
main_path_cesm = "/mnt/data/CESM1/f.e11.FAMIPC5CN/input/"

# Create Handle for ERA dataset
vars_handle = xr.open_dataset("/mnt/data/ERA5_REGRID/ERA5_REGRID_mar_T2_1979_2021.nc")

lon = np.array(vars_handle["lon"])
lat = np.array(vars_handle["lat"])

# Create a grid of longitude and latitude
lon_grid, lat_grid = np.meshgrid(lon, lat)
llgrid = np.concatenate(
    (np.expand_dims(lon_grid, axis=0), np.expand_dims(lat_grid, axis=0)),
    axis=0,
)

# Create a grid of longitude and latitude
lon_grid, lat_grid = np.meshgrid(lon, lat)
llgrid = np.concatenate(
    (np.expand_dims(lon_grid, axis=0), np.expand_dims(lat_grid, axis=0)),
    axis=0,
)

# Create area grid
diff_theta = np.pi * 0.9425 / 180
diff_phi = np.pi * 1.25 / 180
r = 6371
dy = r * diff_theta
dx = np.abs(np.cos(np.pi * llgrid[1, ...] / 180) * r * diff_phi)
dx[dx < 1.0e-4] = 0
da = dx * dy
da = da / np.max(da)
da = 2 * (da - 0.5)

altitude_map = np.load('meritdem_aligned_image.npy') / 10
if base_model == "CMIP6-altitude-circular-conv" or base_model == "CMIP6-NestedUNet" or base_model=="CMIP6-NestedUNet2":
    plt.imshow(altitude_map)
    da = altitude_map
    sea = altitude_map[157]
    da[160:] = sea

# Get grid height and width
grid_height = da.shape[0]
grid_width = da.shape[1]

# Standardization parameters
attribute_norm_vals = {
    "t2m": (
        np.load("../statistics/t2m_mean.npy"),
        np.load("../statistics/t2m_std.npy"),
    ),
    "TS": (
        np.load("../statistics/TS_mean.npy"),
        np.load("../statistics/TS_std.npy"),
    ),
}
attribute_name = "t2m"

###############################################################################

###### PREPARE DATA ###########################################################

# Select attributes

# Create template
month_idxs = []
for i in range(years):
    for j in range(months):
        month_idxs.append(-12 * (i + 1) - (j + 1))
        month_idxs.append(-12 * (i + 1) + (j + 1))
    month_idxs.append(-12 * (i + 1))
for j in range(months):
    month_idxs.append(-(j + 1))

month_idxs = sorted(month_idxs)
input_template = np.array(month_idxs)
input_template -= (prediction_month - 1)
input_template = np.expand_dims(input_template, axis=0)

# Input and output size
input_size = input_template.shape[1]
output_size = 1  # Can change

# Evaluation month idxs
start_idx = 0
val_idx = 400
end_idx = 516  # not inclusive

# Create input and output tensors
print("Creating input and output tensors...")
input_idx_tensor = torch.zeros((end_idx - start_idx, input_size), dtype=torch.long)
month_idx_tensor = torch.arange(start_idx, end_idx).long()
input_idx_tensor[month_idx_tensor - start_idx, :] = torch.tensor(
    input_template
) + month_idx_tensor.unsqueeze(1)

print("Creating handle...")
# dummy = np.arange(1512).reshape(1, 1512, 1, 1).repeat(10, axis=0).repeat(grid_height, axis=2).repeat(grid_width, axis=3)
#handle = vars_handle[attribute_name]
handle = vars_handle[attribute_name]
# dummy = xr.DataArray(dummy)

# Load data to memory
print("Loading data to memory...")
vars_data = np.array(handle)


class Reader(Dataset):
    def __init__(self, split="train"):

        valid_months = np.where((input_idx_tensor < 0).sum(1) == 0)[0]
        if split == "train":
            valid_months = valid_months[valid_months < val_idx]
        elif split == "val":
            valid_months = valid_months[valid_months >= val_idx]

        self.data_list = [month_idx for month_idx in valid_months]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):

        month_idx = self.data_list[idx]

        # Get data
        input_data = vars_data[input_idx_tensor[month_idx, :], :, :]
        output_data = vars_data[month_idx, :, :]

        # Standerdize data
        input_data = (
            input_data - attribute_norm_vals[attribute_name][0]
        ) / attribute_norm_vals[attribute_name][1]
        output_data = (
            output_data - attribute_norm_vals[attribute_name][0]
        ) / attribute_norm_vals[attribute_name][1]

        # Add area array to input data
        input_data = np.concatenate((input_data, np.expand_dims(da, axis=0)), axis=0)
        # Add positional encoding array to input data
        if positional_encoding:
            input_data = np.concatenate((input_data, np.expand_dims(PE, axis=0)), axis=0)

        # Create tensors
        input_data = torch.tensor(input_data, dtype=torch.float32)
        output_data = torch.tensor(output_data, dtype=torch.float32)

        return input_data, output_data


train_dataset = Reader(split="train")
val_dataset = Reader(split="val")

# Create loaders
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
)
validation_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
)

###############################################################################

###### Training functions #####################################################

def bayesian_train_epoch(model, optimizer, loader, criterion):
    model.train()
    KL_loss = 0
    MSE_loss = 0
    
    for i, (input_data, output_data) in enumerate(loader):

        input_data = input_data.to(device_name)
        output_data = output_data.to(device_name)

        output_pred = model(input_data)
        optimizer.zero_grad()

        output_pred = output_pred.reshape(-1, 192, 288)
                
        #model.fcn.requires_grad_(False)
        conv_loss = criterion(output_pred, output_data)
        #conv_loss.backward(retain_graph=True)

        #model.freeze_convs()
        bayes_loss = model.KL_loss()
        loss = conv_loss + bayes_loss
        loss.backward()

        optimizer.step()

        MSE_loss += conv_loss.item()
        KL_loss += bayes_loss.item()

    return MSE_loss / len(loader), KL_loss / len(loader)

def bayesian_val_epoch(model, loader, criterion):

    model.eval()
    KL_loss = 0
    MSE_loss = 0

    for i, (input_data, output_data) in enumerate(loader):

        input_data = input_data.to(device_name)
        output_data = output_data.to(device_name)
        
        output_pred = model(input_data)

        output_pred = output_pred.reshape(-1, 192, 288)

        conv_loss = criterion(output_pred, output_data)
        bayes_loss = model.KL_loss()

        MSE_loss += conv_loss.item()
        KL_loss += bayes_loss.item()

    return MSE_loss / len(loader), KL_loss / len(loader)


def train_epoch(model, optimizer, loader, criterion):
    model.train()
    total_loss = 0
    for i, (input_data, output_data) in enumerate(loader):

        input_data = input_data.to(device_name)
        output_data = output_data.to(device_name)

        optimizer.zero_grad()
        output_pred = model(input_data).squeeze(1)
        if "resnext" in base_model:
            if prediction_month == 1:
                output_pred = output_pred.reshape(-1, 192, 288)
            else:
                output_pred = output_pred.reshape(-1, prediction_month, 192, 288)
        loss = criterion(output_pred, output_data)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def val_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0
    for i, (input_data, output_data) in enumerate(loader):

        input_data = input_data.to(device_name)
        output_data = output_data.to(device_name)

        output_pred = model(input_data).squeeze(1)
        if "resnext" in base_model:
            if prediction_month == 1:
                output_pred = output_pred.reshape(-1, 192, 288)
            else:
                output_pred = output_pred.reshape(-1, prediction_month, 192, 288)

        loss = criterion(output_pred, output_data)
        total_loss += loss.item()

        #if i % prereport_freq == 0:
        #    print(
        #        f"{i}/{len(loader)}: Loss = {total_loss / (i + 1):.4f}"
        #    )

    return total_loss / len(loader)


###############################################################################

print("Model initializing: ", end="")
if positional_encoding:
    model = UNet(n_channels=input_size + 2, n_classes=output_size).to(device_name)
else:
    model = UNet(n_channels=input_size + 1, n_classes=output_size).to(device_name)
    
if base_model == "CMIP6-resnext":
    model = custom_resnext(input_size+1).to(device_name)

if base_model == "CMIP6-resnext-2":
    model = custom_resnext_pretrained(input_size+1).to(device_name)
    
if(base_model == "CMIP6-NestedUNet"):
    model = NestedUNet(input_channels=input_size + 1, num_classes=output_size).to(device_name)
    
if(base_model == "CMIP6-NestedUNet2"):
    model = NestedUNet2(input_channels=input_size + 1, num_classes=output_size).to(device_name)

if(base_model == "CMIP6-BayesianUNetPP"):
    model = BayesianUNetPP(input_channels=input_size + 1, num_classes=output_size).to(device_name)

model.load_state_dict(torch.load(model_weight_path)["state_dict"])

if not "resnext" in base_model and not "Nested" in base_model and not "Bayesian" in base_model:
    model._freeze()

params = [ param for param in model.parameters() if param.requires_grad ]
print("Done")
# Create optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(params, lr=1.0e-5, weight_decay=1.0e-5)

# Training loop
r_schedual = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

best_model = None
best_loss = 99999

train_losses = []
val_losses = []

print("Training starting...")

if "bayesian" in base_model.lower():

    MSE_train_losses = []
    KL_train_losses = []
    MSE_val_losses = []
    KL_val_losses = []
    val_losses = []

    for epoch in range(epochs):
            
        start_time = time.time()

        MSE_train_loss, KL_train_loss = bayesian_train_epoch(model, optimizer, train_loader, criterion)

        with torch.no_grad():
            MSE_val_loss, KL_val_loss = bayesian_val_epoch(model, validation_loader, criterion)
        r_schedual.step()

        MSE_val_losses.append(MSE_val_loss)
        KL_val_losses.append(KL_val_loss)
        KL_train_losses.append(KL_train_loss)
        MSE_train_losses.append(MSE_train_loss)

        val_loss = MSE_val_loss+KL_val_loss
        val_losses.append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            model = model.cpu()
            best_model = model.state_dict()
            model = model.to(device_name)
            save_model(best_model, MSE_train_loss, MSE_val_loss, epoch)

        print(np.asarray(KL_train_losses).shape)
        save_plot(KL_train_losses, KL_val_losses, "KL")
        save_plot(MSE_train_losses, MSE_val_losses, "MSE")

        report = "Epoch: %d, Train MSE Loss: %.6f, Train KL Loss: %.6f, Val MSE Loss: %.6f, Val KL Loss: %.6f, Time: %.2f" % (
                epoch, MSE_train_loss, KL_train_loss, MSE_val_loss, KL_val_loss, time.time() - start_time
            )
        print(report)
else:

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
            
        start_time = time.time()

        train_loss = train_epoch(model, optimizer, train_loader, criterion)
        with torch.no_grad():
            val_loss = val_epoch(model, validation_loader, criterion)
        r_schedual.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            model = model.cpu()
            best_model = model.state_dict()
            model = model.to(device_name)
            save_model(best_model, train_losses, val_losses, epoch)

        if epoch % plot_freq == 0:
            save_plot(train_losses, val_losses, epoch)

        report = "Epoch: %d, Train Loss: %.6f, Val Loss: %.6f, Time: %.2f" % (
                epoch, train_loss, val_loss, time.time() - start_time
            )
        print(report)
