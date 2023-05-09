import torch 
import numpy as np

from sklearn.isotonic import IsotonicRegression

import time

class CalibratedClimateModel():
    def __init__(self, model, N=30, pp=None, pp_params=None, device_name='cuda:1'):

        # model
        self.model = model
        self.posterior_predictive = pp
        self.pp_params = pp_params
        self.N = N
        self.device_name = device_name
        self.ir = None

    def construct_calibration_dataset(self, calib_data):

        calib_loader = torch.utils.data.DataLoader(
            calib_data, batch_size=1, shuffle=True, num_workers=4
        )

        preds = []
        targets = []
        print("Calculating predictions...")
        
        # for input_data, output_data in calib_loader:
            
        #     input_data = input_data.to(self.device_name)
        #     output_data = output_data.to(self.device_name)
        #     for i in range(self.N):
        #         pred = self.model(input_data)

        #     preds.append(np.squeeze(pred.detach().cpu().numpy(), axis=0))
        #     targets.append(output_data.detach().cpu().numpy())

        # np.save("/mnt/data/climate_arrays/preds.npy", preds)
        # np.save("/mnt/data/climate_arrays/targets.npy", targets)

        preds = np.load("/mnt/data/climate_arrays/preds.npy")
        targets = np.load("/mnt/data/climate_arrays/targets.npy")

        print("Started calculating quantiles!")
        quantiles_y = self.calculate_quantiles(preds, targets)
        np.save("/mnt/data/climate_arrays/predicted_cdf.npy", quantiles_y)
        print("Started calculating empirical cdfs")
        empirical_cdf = self.calculate_ecdf(quantiles_y)
        np.save("/mnt/data/climate_arrays/empirical_cdf.npy", empirical_cdf)

    def calculate_quantiles(self, samples, y):

        print(samples.shape, y.shape)
        if len(y.shape) < len(samples.shape):
            y = np.expand_dims(y, axis=(1,2))
        N = samples.shape[1]
        quantiles = np.sum(samples <= y, axis=1) / N
        print(quantiles.shape)
        return quantiles

    def calculate_ecdf(self, x):

        empirical_cdf = np.zeros_like(x, dtype=np.float16)
        H, W = x.shape[-2], x.shape[-1]
        x_repeated = np.expand_dims(np.expand_dims(x, axis=-1), axis=-1)
        for t, elem1 in enumerate(x_repeated):
            empirical_cdf[t] = np.sum(elem1 <= x[t], axis=(0, 1)) / (H*W)
        return np.expand_dims(empirical_cdf, axis=1)

    def fit_isotonic_regressor(self, predicted_cdf, empirical_cdf):

        self.ir = IsotonicRegression(out_of_bounds='clip')
        self.ir(empirical_cdf, predicted_cdf)
        calibrated_quantiles = ir.predict([0.025, 0.5, 0.975])
        return calibrated_quantiles

    def plot_cdfs(self):
        plt.scatter(predicted_quantiles, empirical_quantiles)
        plt.plot([0, 1], [0, 1], color='tab:grey', linestyle='--')
        plt.xlabel('Predicted Cumulative Distribution')
        plt.ylabel('Empirical Cumulative Distribution')
        plt.title('Calibration Dataset')

