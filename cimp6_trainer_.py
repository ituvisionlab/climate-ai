#%%
import torch
import torch.nn as nn
import numpy as np
import xarray as xr

import argparse
import os

import time
import matplotlib.pyplot as plt

from models.model_bayesian import BayesianUNetPP
from torch.utils.data import Dataset
from train_utils import *
from evaluators.utils import *

import warnings

from PIL import Image
import tqdm

warnings.filterwarnings("ignore")

# zamana bağlı ayır, 

import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

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


def save_plot(train_losses, val_losses, loss_type):

    _, ax = plt.subplots()

    ax.plot(train_losses, label="train")
    ax.plot(val_losses, label="val")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()

    plt.savefig(os.path.join(loss_path, "loss_" + loss_type + ".png"), dpi=300)
    plt.cla()


PE = generate_PE(288,192)
plot_freq = 1
prereport_freq = 100

# From the user take the years and months as integers with argparse
parser = argparse.ArgumentParser()
parser.add_argument("--years", type=int, default=3)
parser.add_argument("--months", type=int, default=2)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--device", type=str, default="cuda:1")
parser.add_argument("--prediction_month", type=int, default=1)
parser.add_argument("--positional_encoding", type=bool, default=False)
parser.add_argument("--template_path", type=str, default="../experiments/CMIP6-TCN3D")
parser.add_argument("--model", type=str, default="CMIP6-TCN3D")
parser.add_argument("--kl_coef", type=int, default=1)

# python trainer.py --years 2 --months 2 --batch_size 32 --epochs 200

years = parser.parse_args().years
months = parser.parse_args().months
batch_size = parser.parse_args().batch_size
epochs = parser.parse_args().epochs
device_name = parser.parse_args().device
prediction_month = parser.parse_args().prediction_month
positional_encoding = parser.parse_args().positional_encoding
template_path = parser.parse_args().template_path
model_name = parser.parse_args().model
coef = parser.parse_args().kl_coef

main_path = template_path

loss_path = os.path.join(main_path, "plots/CIMP6_year-%d-month-%d-pred-%d" % (years, months, prediction_month))
save_path = os.path.join(main_path, "weights/CIMP6_year-%d-month-%d-+%d" % (years, months, prediction_month))

# Create directories if they don't exist
if not os.path.exists(loss_path):
    os.makedirs(loss_path)
if not os.path.exists(save_path):
    os.makedirs(save_path)

else:
    print("DIRECTORY ALREADY EXISTS, continuing")
    time.sleep(1)

######## DATA #################################################################
main_path_cesm = "/mnt/data/CMIP6_REGRID/Amon/tas/tas.nc"

# Create handle for CESM dataset
vars_handle = xr.open_dataset(os.path.join(main_path_cesm))

# Load longitude and latitude
altitude_map = np.load('../info/meritdem_aligned_image.npy') / 10
lon = np.array(vars_handle["lon"])
lat = np.array(vars_handle["lat"])

# Create a grid of longitude and latitude
lon_grid, lat_grid = np.meshgrid(lon, lat)
llgrid = np.concatenate(
    (np.expand_dims(lon_grid, axis=0), np.expand_dims(lat_grid, axis=0)),
    axis=0,
)

da = altitude_map
sea = altitude_map[157]
da[160:] = sea

if model_name == "CMIP6-differentDA":
    # Another area grid option
    column = np.arange(0, 0.2, 0.2/192)
    da = np.transpose(np.tile(column, (288, 1)))
# else:
#     # Create area grid
#     da = generate_area_grid(lat, lon)

# Get grid height and width
grid_height = da.shape[0]
grid_width = da.shape[1]

# Standardization parameters
attribute_norm_vals = {
    "tas": (
        np.load("../statistics/tas_mean.npy"),
        np.load("../statistics/tas_std.npy"),
    ),
}

###############################################################################

###### PREPARE DATA ###########################################################

# Select attributes
attribute_name = "tas"
ens_ids = np.arange(0, 9)

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
print(input_template)
input_template = np.expand_dims(input_template, axis=0)

# Input and output size
input_size = input_template.shape[1]
output_size = 1  # Can change

# Evaluation month idxs
start_idx = 0
validation_start_idx = 1700
end_idx = 1800  # not inclusive

# Create input and output tensors
print("Creating input and output tensors...")
input_idx_tensor = torch.zeros((end_idx - start_idx, input_size), dtype=torch.long)
month_idx_tensor = torch.arange(start_idx, end_idx).long()
input_idx_tensor[month_idx_tensor - start_idx, :] = torch.tensor(
    input_template
) + month_idx_tensor.unsqueeze(1)

print("Creating handle...")
handle = vars_handle[attribute_name]

# Load data to memory
print("Loading data to memory...")
vars_data = np.array(handle[ens_ids, :, :, :])

class Reader(Dataset):
    def __init__(self,split="train"):

        valid_months = np.where((input_idx_tensor < 0).sum(1) == 0)[0]
        if split == "train":
            valid_months = valid_months[valid_months < validation_start_idx]
        elif split == "val":
            valid_months = valid_months[valid_months >= validation_start_idx]

        self.data_list = [
            (ens_idx, month_idx)
            for ens_idx in range(vars_data.shape[0])
            for month_idx in valid_months
        ]
        
        # HISTOGRAM
        all_values = []
        for month_idx in valid_months:
            flat = (vars_data[0, month_idx, :, :].flatten() - attribute_norm_vals[attribute_name][0].flatten()) / attribute_norm_vals[attribute_name][1].flatten()
            all_values.append(flat)
                    
        
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):

        ens_idx, month_idx = self.data_list[idx]

        # Get data
        input_data = vars_data[ens_idx, input_idx_tensor[month_idx, :], :, :]
        output_data = vars_data[ens_idx, month_idx, :, :]

        # Standerize data
        input_data = (
            input_data - attribute_norm_vals[attribute_name][0]
        ) / attribute_norm_vals[attribute_name][1]
        output_data = (
            output_data - attribute_norm_vals[attribute_name][0]
        ) / attribute_norm_vals[attribute_name][1]

        # Add area array to input data
        if not model_name == "CMIP6-withoutDA" and not model_name == "CMIP6-UNet-Attention-withoutDA":
            input_data = np.concatenate((input_data, np.expand_dims(da, axis=0)), axis=0)
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

        bayes_loss = (1.0/coef)*model.KL_loss()
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
        bayes_loss = (1.0/coef)*model.KL_loss()

        MSE_loss += conv_loss.item()
        KL_loss += bayes_loss.item()

    return MSE_loss / len(loader), KL_loss / len(loader)


def train_epoch(model, optimizer, loader, criterion):
    model.train()
    total_loss = 0
    
    for i, (input_data, output_data) in enumerate(loader):

        input_data = input_data.to(device_name)
        output_data = output_data.to(device_name)

        print(input_data.shape)
        output_pred = model(input_data)
        optimizer.zero_grad()

        if model_name == "CMIP6-UNet-Attention-withoutDA" or "Attention" in model_name:
            output_pred = output_pred[0]

        output_pred = output_pred.squeeze(1)
        print(output_pred[0], output_data[0])
        print(torch.nn.functional.mse_loss(output_pred, output_data))

        if "resnext" in model_name:
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
        
        output_pred = model(input_data)
        if model_name == "CMIP6-UNet-Attention-withoutDA" or "Attention" in model_name:
            output_pred = output_pred[0]

        output_pred = output_pred.squeeze(1)

        if "resnext" in model_name:
            if prediction_month == 1:
                output_pred = output_pred.reshape(-1, 192, 288)
            else:
                output_pred = output_pred.reshape(-1, prediction_month, 192, 288)
        loss = criterion(output_pred, output_data)
        total_loss += loss.item()

    return total_loss / len(loader)


###############################################################################

print("Model initializing: ", end="")

if positional_encoding:
    input_size += 1

model = model_select(model_name, input_size, output_size, device_name).to(device_name)

# Create optimizer

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3, weight_decay=1.0e-3)

# Training loop
r_schedual = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

best_model = None
best_loss = 99999

print("Training starting...")

if "bayesian" in model_name.lower():

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
            save_plot(train_losses, val_losses, "MSE")

        report = "Epoch: %d, Train Loss: %.6f, Val Loss: %.6f, Time: %.2f" % (
                epoch, train_loss, val_loss, time.time() - start_time
            )
        print(report)
