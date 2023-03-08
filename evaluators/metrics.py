import uncertainty_toolbox as uct
import numpy as np
import os
import matplotlib.pyplot as plt

def sharpness2d(stds):
    sharpness_per_pixel = np.zeros((stds.shape[1], stds.shape[2]))
    print(stds.shape)
    print(stds[:, 1, 1].shape)
    for k in range(stds.shape[1]):
        for m in range(stds.shape[2]):
            sharp_metric = uct.sharpness(stds[:, k, m])
            sharpness_per_pixel[k, m] = sharp_metric
    sharpness_per_pixel = np.flip(sharpness_per_pixel, axis=0)
    return sharpness_per_pixel
    
def nll_gaussian2d(log_sigma2s, means, ys):
    return np.flip(np.mean(np.exp(-log_sigma2s)*np.square(ys - means) + log_sigma2s, axis=0), axis=0)

def mae2d(mean_preds, targets):
    return np.flip(np.mean(np.abs(mean_preds - targets), axis=0), axis=0)

def rmse2d(mean_preds, targets):
    return np.flip(np.mean(np.square(mean_preds - targets), axis=0), axis=0)

def compute_metrics(mean_preds, std_preds, targets):
    nll = nll_gaussian2d(np.log(np.square(std_preds)), mean_preds, targets)
    sharpness_per_pixel = sharpness2d(std_preds)
    mae = mae2d(mean_preds, targets)
    rmse = rmse2d(mean_preds, targets)
    
    return nll, sharpness_per_pixel, mae, rmse

def plot(title, metric_name, model_name):
    current_path = os.path.dirname(os.getcwd())
    data = np.loadtxt(current_path+'/'+model_name+'/evals/single/ExpYear_3_ExpMonth_2/all/'+metric_name+'.txt')
    plt.imshow(data)
    plt.title(title + "\n Total: " + str(np.mean(data)))
    plt.colorbar(orientation="horizontal")
    plt.show()
    plt.savefig(current_path+'/'+model_name+'/evals/single/ExpYear_3_ExpMonth_2/all/'+metric_name+'.png')
    plt.clf()

def plot_metrics(model_name):
    plot("Negative Log Likelihood for Regression", "metric_nll", model_name)
    plot("Sharpness", "metric_sharpness", model_name)
    plot("MAE", "metric_acc_mae", model_name)
    plot("RMSE", "metric_acc_rmse", model_name)