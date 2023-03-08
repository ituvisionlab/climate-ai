'''
    Code that evaluates with the last 116 samples of ERA5 dataset.
    First 400 is used for finetune.
    
    Works with: 1 or multiple month predictions
    
    Outputs: Output is an image that its each pixel contains values average taken over all test set (ERA5)
    * Avg Prediction (img)
    * Avg RMSE Map (img)
    * Avg MAE Map (img)
    
    You can evaluate for:
    * all models (average of all months side by side for each experiment)
    * just one experiment result (average of all months)
    * average error and prediction of each month seperately (for single experiment or for all)
    * average error and prediction of each season seperately (for single experiment or for all)
    
'''

import torch
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl

import os

import argparse

from serial_experiments.model import UNet
from serial_experiments.model_plusplus import NestedUNet
from serial_experiments.model_plusplus2 import NestedUNet2
from serial_experiments.model_plusplus2_dropout import NestedUNet2 as UNetppDropout
from serial_experiments.modelpadding import UNet as Unetcircular
from serial_experiments.models.resnext_model import custom_resnext, custom_resnext2, custom_resnext_pretrained
from serial_experiments.models.seg_models import UNetPlusPlus, PSPNet, UNetSE, DeepLabV3, UNetJ
from serial_experiments.model_bayesian import BayesianUNetPP
from serial_experiments.model_calib import heteroscedastic_loss, calibModel, heteroscedastic_loss2

from utils import *

######## CONFIGURATIONS #################################################################

parser = argparse.ArgumentParser()
parser.add_argument("--prediction_month", type=int, default=1) # How many months after the inputs you want to predict
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--template_path", type=str) # Main folder of the model
parser.add_argument("--experiment_type", type=int, default=1) # Experiment type 1: +-1 and +-2, Experiment type 2: 0 years 6,12,18,24,36 months
parser.add_argument("--model", type=str) # Finetuned model or not
parser.add_argument("--metric_type", type=str, default="MAE") # Evaluation metrics available: RMSE, MAE
parser.add_argument("--batch_size", type=int, default=64) # Evaluation metrics available: RMSE, MAE
parser.add_argument("--single", type=bool, default=False) # Single experiment or all experiments
parser.add_argument("--eval_type", type=str, default="all") # Options: all, season, month
# If you want to evaluate for a month, which month you want to evaluate for
parser.add_argument("--month", type=int, default=-1)
# If you want to evaluate for a season, which season you want to evaluate for
parser.add_argument("--season", type=str, default="")
# If you want to evaluate for a single experiment, specify months and years
parser.add_argument("--exp_year", type=int, default=-1)
parser.add_argument("--exp_month", type=int, default=-1)
parser.add_argument("--save_xarray", type=bool, default=True)


# Experiment settings parameters

prediction_month = parser.parse_args().prediction_month
device_name = parser.parse_args().device
template_path = parser.parse_args().template_path
exp_type = parser.parse_args().experiment_type
model_type = parser.parse_args().model
metric = parser.parse_args().metric_type
exp_year = parser.parse_args().exp_year
exp_month = parser.parse_args().exp_month
batch_size = parser.parse_args().batch_size
eval_type = parser.parse_args().eval_type
season = parser.parse_args().season
month_to_predict = parser.parse_args().month
single = parser.parse_args().single
# Configurations for different experiments


current_path = os.path.dirname(os.getcwd())
template_path_base = current_path + "/" + template_path

template_path = current_path + "/" + template_path + "/evals"

months_dictionary = {"1":"Jan", "2":"Feb", "3":"Mar", "4":"Apr", "5":"May", "6":"June", "7":"July", "8":"Aug", "9":"Sep","10":"Oct","11":"Nov","12":"Dec"}

if single == True:
    template_path += "/" + "single" + "/ExpYear_" + str(exp_year) + "_ExpMonth_" + str(exp_month)

if eval_type == "all":
    template_path += "/" + eval_type
elif eval_type == "month":
    template_path += "/" + eval_type + "/Month_" + months_dictionary[str(month_to_predict)]
elif eval_type == "season":
    template_path += "/" + eval_type + "/Season_" + season
    
print(template_path)
if not os.path.exists(template_path):
    print(template_path)
    os.makedirs(template_path)
    print("Directory is created")

if single == True:
    print("Evaluating for Experiment year: " + str(exp_year) + " month: " + str(exp_month))
    all_years = [exp_year]
    all_months = [exp_month]
else:
    print("Evaluating for all experiments for experiment type " + str(exp_type))
    if exp_type == 1:
        all_years = [1, 2, 3, 4]
        all_months = [1, 2]
    else:
        all_years = [0]
        all_months = [6, 12, 18, 24, 30, 36]
    
months_to_predict = []
if season == "winter":
    print("Evaluating for Winter")
    months_to_predict = [12, 1, 2]
elif season == "spring":
    print("Evaluating for Spring")
    months_to_predict = [3, 4, 5]
elif season == "summer":
    print("Evaluating for Summer")
    months_to_predict = [6, 7, 8]
elif season == "fall":
    print("Evaluating for Fall")
    months_to_predict = [9, 10, 11]
else:
    print("No season chosen")

if month_to_predict != -1:
    print("Evaluating for month " + months_dictionary[str(month_to_predict)])
else:
    print("No month choosen")

######## LOAD DATA #################################################################

main_path_cesm = "/mnt/data/CESM1/f.e11.FAMIPC5CN/input/"

vars_handle = xr.open_dataset("/mnt/data/ERA5_REGRID/ERA5_REGRID_mar_T2_1979_2021.nc")
lon = np.array(vars_handle["lon"])
lat = np.array(vars_handle["lat"])

# Generate an area grid in order to let model notice the earth projection errors
altitude_map = np.load('/home/bugra/climate-ai/serial_experiments/meritdem_aligned_image.npy') / 10
if model_type == "CMIP6-differentDA":
    da = generate_fake_area_grid(lat, lon)
if (model_type == "CMIP6-altitude-circular-conv") or (model_type == "CMIP6-finetune-altitude-circular-conv") or (model_type == "CMIP6-UNet-AttentionSE") or (model_type=="CMIP6-UNetPlusPlus" or (model_type == "CMIP6-UNet-AttentionSE-withoutDA") or (model_type == "CMIP6-NestedUNet") or (model_type=="CMIP6-finetune-NestedUNet") or (model_type=="CMIP6-NestedUNet2") or (model_type=="CMIP6-calib1") or (model_type=="CMIP6-BayesianUNetPP")):
    da = altitude_map
    sea = altitude_map[157]
    da[160:] = sea
    np.save('elevation_map', da)
else:
    da = generate_area_grid(lat, lon)

# Standardization parameters
attribute_norm_vals = {
    "t2m": (
        np.load("../../statistics/t2m_mean.npy"),
        np.load("../../statistics/t2m_std.npy"),
    )
}
attribute_name = "t2m"

###### PREPARE DATA AND EVALUATE ###########################################################

plot_count = np.ones(prediction_month).astype('int')
plot_conf = {
    "PREDcmap": mpl.cm.get_cmap('RdYlBu'),
    "RMSEcmap": mpl.cm.get_cmap('Greens'),
    "RMSEnorm": mpl.colors.Normalize(vmin=0, vmax=5),
    "PREDnorm": mpl.colors.Normalize(vmin=-40, vmax=40)
}

fig1 = plt.figure(figsize=(30,15))
fig2 = plt.figure(figsize=(30,15))

maes = []
last_month_maes = []
for p in range(prediction_month):
    for year in all_years:
        for month in all_months:
            month_idxs = generate_month_indexes(year, month)
            month_idxs = sorted(month_idxs)
            
            input_template = np.array(month_idxs)
            input_template -= (prediction_month - 1)
            input_template = np.expand_dims(input_template, axis=0)
            
            output_template = np.arange(0, prediction_month, 1)
            output_template = np.expand_dims(output_template, axis=0)

            # Input and output size
            input_size = input_template.shape[1]
            output_size = prediction_month
            
            handle = vars_handle[attribute_name]
            
            relevant_months = np.zeros((0,0))
            if eval_type == "season":
                time_steps = vars_handle["time"][400:].to_numpy()
                years = time_steps.astype('datetime64[Y]').astype(int) + 1970
                months = time_steps.astype('datetime64[M]').astype(int) % 12 + 1
                days = time_steps - time_steps.astype('datetime64[M]') + 1
                for month_to_predict in months_to_predict:
                    relevant_months = np.append(relevant_months, np.where(months == month_to_predict)[0])
                relevant_months = relevant_months.astype(int)
                time_steps = time_steps[relevant_months, ...]
                years = years[relevant_months, ...]
                months = months[relevant_months, ...]
                days = days[relevant_months, ...]
                stamps = np.stack((years, months, days), axis=1)
                stamps = [ "%d-%d-%d" % tuple([stamps[i, j] for j in range(3)]) for i in range(stamps.shape[0])]
            elif eval_type == "month":
                time_steps = vars_handle["time"][400:].to_numpy()
                years = time_steps.astype('datetime64[Y]').astype(int) + 1970
                months = time_steps.astype('datetime64[M]').astype(int) % 12 + 1
                days = time_steps - time_steps.astype('datetime64[M]') + 1
                relevant_months = np.where(months == month_to_predict)[0]
                print(relevant_months)
                time_steps = time_steps[relevant_months, ...]
                years = years[relevant_months, ...]
                months = months[relevant_months, ...]
                days = days[relevant_months, ...]
                stamps = np.stack((years, months, days), axis=1)
                stamps = [ "%d-%d-%d" % tuple([stamps[i, j] for j in range(3)]) for i in range(stamps.shape[0])]
            else:
                time_steps = vars_handle["time"][400:].to_numpy()
                years = time_steps.astype('datetime64[Y]').astype(int) + 1970
                months = time_steps.astype('datetime64[M]').astype(int) % 12 + 1
                relevant_months = np.where(months == month_to_predict)[0]
                days = time_steps - time_steps.astype('datetime64[M]') + 1
                stamps = np.stack((years, months, days), axis=1)
                stamps = [ "%d-%d-%d" % tuple([stamps[i, j] for j in range(3)]) for i in range(stamps.shape[0])]
                
            start_idx = 0
            end_idx = 516 # not inclusive
            
            input_idx_tensor, output_idx_tensor, month_idx_tensor = create_input_and_output_idxs(start_idx, end_idx, input_size, input_template, prediction_month, eval_type, relevant_months)
            input_tensor, output_tensor = create_input_and_output_tensors(attribute_norm_vals, prediction_month, output_idx_tensor, input_idx_tensor, month_idx_tensor, handle, attribute_name, da, model_type)
            model_weight_path = get_model_weights(model_type, year, month, prediction_month)
            
            val_input = torch.tensor(input_tensor).float().to(device_name)

            if("circular-conv" in model_type):
                model = Unetcircular(n_channels=input_size + 1, n_classes=output_size).to(device_name)
            elif("multi" in model_type):
                model = Unetcircular(n_channels=input_size + 1, n_classes=output_size).to(device_name)
            elif(model_type == "CMIP6-resnext-2"):
                model = custom_resnext2(input_size + 1).to(device_name)
            elif(model_type == "CMIP6-resnext"):
                model = custom_resnext(input_size + 1).to(device_name)
            elif(model_type == "CMIP6-pretrained-resnext"):
                model = custom_resnext_pretrained(input_size + 1).to(device_name)
            elif(model_type == "CMIP6-PSPNet"):
                model = PSPNet(input_size + 1, "resnext50_32x4d").to(device_name)
            elif(model_type == "CMIP6-PSPNetSE"):
                model = PSPNet(input_size + 1, "se_resnext50_32x4d").to(device_name)
            elif(model_type == "CMIP6-UNetPlusPlus"):
                model = UNetPlusPlus(input_size + 1, "resnext50_32x4d", None).to(device_name)
            elif(model_type == "CMIP6-UNetPlusPlus-34"):
                model = UNetPlusPlus(input_size + 1, "resnet_34").to(device_name)
            elif(model_type == "CMIP6-UNetSE"):
                model = UNetSE(input_size + 1).to(device_name)
            elif(model_type == "CMIP6-DeepLabV3"):
                model = DeepLabV3(input_size + 1).to(device_name)
            elif(model_type == "CMIP6-UNet-AttentionSE"):
                model = UNetJ(input_size + 1, "se_resnext50_32x4d", "scse").to(device_name)
            elif(model_type == "CMIP6-UNet-Attention-withoutDA"):
                model = UNetJ(input_size, "se_resnext50_32x4d", "scse").to(device_name)
            elif(model_type == "CMIP6-UNet-AttentionSE-withoutDA"):
                model = UNetJ(input_size+1, "se_resnext50_32x4d", "scse").to(device_name)
            elif(model_type == "CMIP6-UNetPlusPlus-AttentionSE"):
                model = UNetPlusPlus(input_size + 1, "se_resnext50_32x4d", "scse").to(device_name)
            elif(model_type == "CMIP6-withoutDA"):
                model = Unetcircular(n_channels=input_size, n_classes=output_size).to(device_name)
            elif(model_type == "CMIP6-finetune"):
                model = UNet(n_channels=input_size + 1, n_classes=output_size).to(device_name)
            elif(model_type == "CMIP6-NestedUNet" or model_type == "CMIP6-finetune-NestedUNet"):
                model = NestedUNet(input_channels=input_size + 1, num_classes=output_size).to(device_name)
            elif(model_type == "CMIP6-NestedUNet2" or model_type == "CMIP6-finetune-NestedUNet2"):
                model = NestedUNet2(input_channels=input_size + 1, num_classes=output_size).to(device_name)
            elif(model_type=="CMIP6-calib1"):
                model = calibModel(in_channels=input_size+1, out_channels=output_size, dropout=0.0, model_name="CMIP6-NestedUNet2", years=year, months=month, path=True, finetuned=True, prediction_month=1).to(device_name)
            elif(model_type == "CMIP6-BayesianUNetPP"):
                model = BayesianUNetPP(input_channels=input_size + 1, num_classes=output_size).to(device_name)
            elif(model_type == "CMIP6-UNetppDropout"):
                model = UNetppDropout(input_channels=input_size + 1, num_classes=output_size).to(device_name)
                model.dropout.train()
            else:
                model = Unetcircular(n_channels=input_size + 1, n_classes=output_size).to(device_name)
            model.load_state_dict(torch.load(model_weight_path)["state_dict"])
            model = model.eval()
            for param in model.parameters():
                param.requires_grad = False
                
            all_preds = []

            bs = batch_size
            num_batches = val_input.shape[0] // bs + int((val_input.shape[0] % bs != 0))
            for i in range(num_batches):
                start = i * bs
                end = min((i + 1) * bs, val_input.shape[0])
                preds = model(val_input[start:end])
                if "resnext" in model_type:
                    if prediction_month == 1:
                        preds = preds.reshape(-1, 192, 288)
                    else:
                        preds = preds.reshape(-1, prediction_month, 192, 288)
                if "UNet-Attention" in model_type:
                    preds = preds[0].detach().cpu().numpy() * attribute_norm_vals[attribute_name][1] + attribute_norm_vals[attribute_name][0]
                else:
                    preds = preds.squeeze(1).detach().cpu().numpy() * attribute_norm_vals[attribute_name][1] + attribute_norm_vals[attribute_name][0]
                all_preds.append(preds)

            preds = np.concatenate(all_preds, axis=0)
            if prediction_month == 1:
                if len(preds.shape) < 4:
                    preds = np.expand_dims(preds, axis=1)
                if len(output_tensor.shape) < 4:
                    output_tensor = np.expand_dims(output_tensor, axis=1)
            
            if eval_type == "all":
                preds = preds[400:, p, ...]
                output_tensor = output_tensor[400:, p, ...]
            else:
                preds = preds[:, p, ...]
                output_tensor = output_tensor[:, p, ...]
            
            if preds.shape[0] >= 12:
                last_year_pred = output_tensor[np.arange(output_tensor.shape[0]) - 12]
            else: 
                last_year_pred = np.zeros((1,1))
            last_month_pred = output_tensor[np.arange(output_tensor.shape[0]) - 1]
            
            if single == False:
                if exp_type == 1:
                    num_row_figs = 2
                    num_col_figs = 4
                elif exp_type == 2:
                    num_row_figs = 2
                    num_col_figs = 3
            else:
                num_row_figs = 1
                num_col_figs = 1
                
                
            # PREDICTIONS XARRAY
            
            predictions_xr = xarray_generator(preds, stamps, lat, lon)
            # predicted_instances = preds.shape[0]
            # difference = len(stamps) - predicted_instances
            # stamps = stamps[difference:]
            # predictions_xr = xr.DataArray(preds, coords={'x':lon, 'y':lat, 'time':stamps}, dims=['time', 'y', 'x'])
            
            # OUTPUTS XARRAY
            outputs_xr = xarray_generator(output_tensor, stamps, lat, lon)
            
            if metric == "RMSE":
                pred_err_map, last_year_err_map, last_month_err_map = calculate_RMSE_maps(preds, output_tensor, last_year_pred, last_month_pred)
            elif metric == "MAE":
                pred_err_map, last_year_err_map, last_month_err_map = calculate_MAE_maps(preds, output_tensor, last_year_pred, last_month_pred)
                
            # RMSE XARRAY
            pred_err_map2 = np.abs(preds - output_tensor) ** 0.5
            rmse_xr = xarray_generator(pred_err_map2, stamps, lat, lon)
            
            last_month_maes.append(np.mean(last_year_err_map))

            if eval_type == "month":
                output_type = months_dictionary[str(month_to_predict)]
                output_type2 = output_type
            elif eval_type == "season":
                output_type = season + " months"
                output_type2 = season
            else:
                output_type = "All months avg"
                output_type2 = "all_months"
                
                
            # predictions_xr.to_netcdf(template_path_base+"/xarrays/"+eval_type+"/"+model_type+"_"+output_type2+"_year_"+str(year)+"_month_"+str(month)+"_predictions.nc", engine="h5netcdf", invalid_netcdf=True)
            # outputs_xr.to_netcdf(template_path_base+"/xarrays/"+eval_type+"/"+model_type+"_"+output_type2+"_year_"+str(year)+"_month_"+str(month)+"_ground_truths.nc", engine="h5netcdf", invalid_netcdf=True)
            # rmse_xr.to_netcdf(template_path_base+"/xarrays/"+eval_type+"/"+model_type+"_"+output_type2+"_year_"+str(year)+"_month_"+str(month)+"_RMSEs.nc", engine="h5netcdf", invalid_netcdf=True)

            title1 = "Prior Years: %d Prior Months: %d \n Considered: %s \n Prediction Month: %d Avg MAE: %.6f" % (year, month, output_type, prediction_month, np.mean(pred_err_map))
            title2 = "Prior Years: %d Prior Months: %d \n Considered: %s \n Prediction Month: %d Avg MAE: %.6f" % (year, month, output_type, prediction_month, np.mean(pred_err_map))

            maes.append(np.mean(pred_err_map))
            
            pred_err_map_no_south = pred_err_map[25:, :]
            #fig_num1 = 1 + p*prediction_month
            fig_num1 = 1
            data1 = np.mean(output_tensor, axis=(0)) - 273.5 # Take mean accross all samples and convert to Degree from Kelvin
            plot_comparison_graphs(fig_num1, num_row_figs, num_col_figs, plot_count[p], data1, title1, "prediction", plot_conf)

            #fig_num2 = 2 + p*prediction_month
            
            fig_num2 = 2
            data2 = pred_err_map
            plot_comparison_graphs(fig_num2, num_row_figs, num_col_figs, plot_count[p], data2, title2, "MAE", plot_conf)
            # plot_count[p] += 1
            
            avg_preds_xr = xarray_generator(data1, ['avg-preds-'+output_type2], lat, lon)
            avg_RMSE_xr = xarray_generator(data2, ['avg-RMSE-'+output_type2], lat, lon)
            
            #avg_preds_xr.to_netcdf(template_path_base+"/xarrays/"+eval_type+"/"+model_type+"_"+output_type2+"_year_"+str(year)+"_month_"+str(month)+"_avg_preds.nc", engine="h5netcdf", invalid_netcdf=True)
            #avg_RMSE_xr.to_netcdf(template_path_base+"/xarrays/"+eval_type+"/"+model_type+"_"+output_type2+"_year_"+str(year)+"_month_"+str(month)+"_avg_MAE.nc", engine="h5netcdf", invalid_netcdf=True)
            
            # arr = xr.open_dataset(template_path_base+"/xarrays/month/CMIP6-altitude-circular-conv_Apr_year_0_month_6_predictions.nc")
            # print(arr["time"])
            # data3 = pred_err_map_no_south ** 0.5
            # no_south_rmse = np.mean(pred_err_map_no_south) ** 0.5
            # title3 = "Prior Years: %d Prior Months: %d \n Considered: %s \n Prediction Month: %d Avg RMSE: %.6f" % (year, month, output_type, prediction_month, no_south_rmse)
            # plot_comparison_graphs(fig_num2, num_row_figs, num_col_figs, plot_count[p], data3, title3, "RMSE", plot_conf)
            plot_count[p] += 1
            
    # PREDICTIONS PLOT

    fig1 = plt.figure(fig_num1)
    axs = fig1.get_axes()
    if single == False:
        plt.suptitle("Prediction", size=40)
    else:
        plt.suptitle("Prediction", size=20)
        plt.tight_layout(pad=2)
    fig1.colorbar(mpl.cm.ScalarMappable(norm=plot_conf["PREDnorm"], cmap=plot_conf["PREDcmap"]), ax=axs, shrink=0.6, location='bottom').set_label(label="Temperature (Celcius)", size=10)
    plt.show()
    #plt.savefig(template_path + "/" + "PREDICTIONS_EXP_TYPE_" + str(exp_type) + "_MULTI_" + str(prediction_month) + "_PREDMONTH_" + str(p) + ".png", dpi=300)
    print("SAVED")
    
    # RMSE PLOT
    fig2 = plt.figure(fig_num2)
    axs = fig2.get_axes()
    if single == False:
        plt.suptitle("RMSE", size=40)
    else:
        plt.suptitle("RMSE", size=20)
        plt.tight_layout(pad=2)
    fig2.colorbar(mpl.cm.ScalarMappable(norm=plot_conf["RMSEnorm"], cmap=plot_conf["RMSEcmap"]), ax=axs, shrink=0.6, location='bottom').set_label(label="Average RMSE", size=10)
    plt.show()
    if metric == "RMSE":
        plt.savefig(template_path + "/" + "RMSE_EXP_TYPE_"  + str(exp_type) + "_MULTI_" + str(prediction_month) + "_PREDMONTH_" + str(p) + ".png", dpi=300)
        #plt.savefig(template_path_base + "/wout_south/" + "RMSE_EXP_TYPE_"  + str(exp_type) + "_MULTI_" + str(prediction_month) + "_PREDMONTH_" + str(p) + ".png", dpi=300)

        print("SAVED")
    elif metric == "MAE":
        plt.savefig(template_path + "/" + "MAE_EXP_TYPE_"  + str(exp_type) + "_MULTI_" + str(prediction_month)  + "_PREDMONTH_" + str(p) + ".png", dpi=300)
    plt.close()
    
    np.savetxt(template_path+"/maes"+str(exp_type)+'.txt', np.asarray(maes), delimiter=',')
    np.savetxt(template_path+"/persistance"+str(exp_type)+'.txt', np.asarray(last_month_maes))
    
    with open(template_path+'/maes'+str(exp_type)+'.txt', 'r') as f:
        lines = [line.replace('.', ',') for line in f.readlines()]
    with open(template_path+'/maes'+str(exp_type)+'.txt', 'w') as f:
        for line in lines:
            f.write(line)





