import matplotlib.pyplot as plt
import numpy as np

model_name = "CMIP6-UNetppDropout"

nll = np.loadtxt('climate-ai/serial_experiments/'+model_name+'/evals/single/ExpYear_3_ExpMonth_2/all/metric_nll.txt')
plt.imshow(nll)
plt.title("Negative Log Likelihood for Regression")
plt.colorbar(orientation="horizontal")
plt.show()
plt.savefig('/home/bugra/climate-ai/serial_experiments/'+model_name+'/evals/single/ExpYear_3_ExpMonth_2/all/nll.png')
plt.clf()

sharpness = np.loadtxt('/home/bugra/climate-ai/serial_experiments/'+model_name+'/evals/single/ExpYear_3_ExpMonth_2/all/metric_sharpness.txt')

plt.imshow(sharpness)
plt.title("Sharpness")
plt.colorbar(orientation="horizontal")
plt.show()
plt.savefig('/home/bugra/climate-ai/serial_experiments/'+model_name+'/evals/single/ExpYear_3_ExpMonth_2/all/sharpness.png')
plt.clf()

mae = np.loadtxt('/home/bugra/climate-ai/serial_experiments/'+model_name+'/evals/single/ExpYear_3_ExpMonth_2/all/metric_acc_mae.txt')

plt.imshow(mae)
plt.title("MAE")
plt.colorbar(orientation="horizontal")
plt.show()
plt.savefig('/home/bugra/climate-ai/serial_experiments/'+model_name+'/evals/single/ExpYear_3_ExpMonth_2/all/mae.png')
plt.clf()

rmse = np.loadtxt('/home/bugra/climate-ai/serial_experiments/'+model_name+'/evals/single/ExpYear_3_ExpMonth_2/all/metric_acc_rmse.txt')

plt.imshow(rmse)
plt.title("RMSE")
plt.colorbar(orientation="horizontal")
plt.show()
plt.savefig('/home/bugra/climate-ai/serial_experiments/'+model_name+'/evals/single/ExpYear_3_ExpMonth_2/all/rmse.png')
plt.clf()