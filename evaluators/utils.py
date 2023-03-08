from sys import platform
import torch
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

def generate_fake_area_grid(lat, lon):
    column = np.arange(0, 0.2, 0.2/192)
    da = np.transpose(np.tile(column, (288, 1)))
    return da

def generate_area_grid(lat, lon):
    
    '''
        Generate an area grid in order to let model notice the earth projection errors
        Output: 192x288 matrix having the area size values
    '''
    
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
    
    return da

def generate_month_indexes(year, month):
    
    '''
        Generate month index arrays for the model according to each experiment setting
        For example: [0 1 2 12] for predicting 13
    '''

    month_idxs = []
    for i in range(year):
        for j in range(month):
            month_idxs.append(-12 * (i + 1) - (j + 1))
            month_idxs.append(-12 * (i + 1) + (j + 1))
        month_idxs.append(-12 * (i + 1))
    for j in range(month):
        month_idxs.append(-(j + 1))
        
    return month_idxs


def create_input_and_output_idxs(start_idx, end_idx, input_size, input_template, prediction_month, eval_type, relevant_months):
    
    # Create input and output tensors
    input_idx_tensor = torch.zeros((end_idx - start_idx, input_size), dtype=torch.long)
    month_idx_tensor = torch.arange(start_idx, end_idx).long()
    input_idx_tensor[month_idx_tensor - start_idx, :] = torch.tensor(
        input_template
    ) + month_idx_tensor.unsqueeze(1)
    
    month_idx_tensor = month_idx_tensor[ (input_idx_tensor < 0).sum(1) == 0 ]
    input_idx_tensor = input_idx_tensor[ (input_idx_tensor < 0).sum(1) == 0 ]
    
    if eval_type == "season" or eval_type == "month":
        input_idx_tensor = input_idx_tensor[np.argwhere(np.isin(input_idx_tensor[:,-1], relevant_months-1)).ravel()]
        month_idx_tensor = input_idx_tensor[:,-1] + 1        
    output_idx_tensor = torch.zeros((0,))
    
    for i in range(prediction_month):
        output_idx_tensor = torch.concat((output_idx_tensor, input_idx_tensor[:, -1]+(i+1)))
        
    output_idx_tensor = output_idx_tensor.reshape((input_idx_tensor.shape[0], prediction_month)).int()
    
    return input_idx_tensor, output_idx_tensor, month_idx_tensor

def create_input_and_output_tensors(attribute_norm_vals, prediction_month, output_idx_tensor, input_idx_tensor, month_idx_tensor, handle, attribute_name, da, model_type): 
    
    grid_height = da.shape[0]
    grid_width = da.shape[1]

    input_tensor = np.array(handle[np.array(input_idx_tensor).reshape(-1), ...])
    input_tensor = input_tensor.reshape(
        input_idx_tensor.shape[0],
        input_idx_tensor.shape[1],
        grid_height,
        grid_width,
    )
 
    # Standardize input tensor
    input_tensor = (input_tensor - attribute_norm_vals[attribute_name][0]) / attribute_norm_vals[attribute_name][1]
    if prediction_month >= 2:
        output_tensor = []
        for out in output_idx_tensor:
            # if out[1] < 516:
            output_tensor.append(np.array(handle[out, ...])) # bu da numpydır, prediction not standardized
        output_tensor = np.asarray(output_tensor)
    else:
        output_tensor = np.array(handle[month_idx_tensor, ...]) # bu da numpydır, prediction not standardized
    ###############################################################################

    # Add area map
    if not "withoutDA" in model_type:
        input_tensor = np.concatenate(
            (
                input_tensor,
                np.expand_dims(da, axis=(0, 1)).repeat(input_idx_tensor.shape[0], axis=0),
            ),
            axis=1,
        ) # Bu bir numpydır
    
    return input_tensor, output_tensor

def get_model_weights(model_type, year, month, prediction_month):
    current_path = os.path.dirname(os.getcwd())
    if "finetune" in model_type:
        model_weight_path = current_path + "/" + model_type + "/weights/CIMP6_year-%d-month-%d-predmont-%d/best.pth" % (year, month, prediction_month)
    elif model_type == "CMIP6-UNetppDropout":
        model_weight_path = current_path + "/" + "CMIP6-finetune-NestedUNet2" + "/weights/CIMP6_year-%d-month-%d-predmont-%d/best.pth" % (year, month, prediction_month)
    else:
        model_weight_path = current_path + "/" + model_type + "/weights/CIMP6_year-%d-month-%d-+%d/best.pth" % (year, month, prediction_month)
    return model_weight_path


def plot_comparison_graphs(fig_num, num_row_figs, num_col_figs, plot_count, img, title, plot_type, plot_conf):
    fig = plt.figure(fig_num)
    plt.subplot(num_row_figs, num_col_figs, plot_count)
    plt.title(title, fontsize=10)
    plt.rc('font', size=10)
    
    tr_img = np.flip(img, axis=0)
    preds2 = tr_img[:, :144]
    preds1 = tr_img[:, 144:]
    tr_img = np.concatenate((preds1, preds2), axis=1)
        
    if plot_type == "prediction":
        plt.imshow(tr_img, vmin=-40, vmax=40, cmap=plot_conf["PREDcmap"])
        
    elif plot_type == "MAE" or plot_type == "RMSE":
        plt.imshow(tr_img, vmin=0, vmax=5, cmap=plot_conf["RMSEcmap"])

    #return fig

def calculate_RMSE_maps(preds, output_tensor, last_year_pred, last_month_pred):
    pred_err_map = np.mean((preds - output_tensor) ** 2, axis=0)
    if last_year_pred.shape[0] > 1:
        last_year_err_map = np.mean((last_year_pred - output_tensor) ** 2, axis=0)
    else:
        last_year_err_map = None
    last_month_err_map = np.mean((last_month_pred - output_tensor) ** 2, axis=0)
    return pred_err_map, last_year_err_map, last_month_err_map


def calculate_MAE_maps(preds, output_tensor, last_year_pred, last_month_pred):
    pred_err_map = np.mean(np.abs(preds - output_tensor), axis=0)
    if last_year_pred.shape[0] > 1:
        last_year_err_map = np.mean(np.abs(last_year_pred - output_tensor), axis=0)
    else:
        last_year_err_map = None

    last_month_err_map = np.mean(np.abs(last_month_pred - output_tensor), axis=0)
    return pred_err_map, last_year_err_map, last_month_err_map

def xarray_generator(array_to_convert, stamps, lat, lon):
    print("xarray generator")
    
    if len(array_to_convert.shape) == 2:
        array_to_convert = np.expand_dims(array_to_convert, axis=0)
        
    instances = array_to_convert.shape[0]
    difference = len(stamps) - instances
    stamps = stamps[difference:]
    generated_xr = xr.DataArray(array_to_convert, coords={'x':lon, 'y':lat, 'time':stamps}, dims=['time', 'y', 'x'], name="ML-data")
    return generated_xr


