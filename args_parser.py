import argparse

def get_args():
# python cimp6_trainer_vars_reallyall.py --epochs 50 --template_path ../experiments/CMIP6-NestedUNet2all-0.0001 --model CMIP6-NestedUNet2all-0.0001 --batch_size 16 --device cuda:2 --years 3 --months 2 --weight_decay 0.0001 --lr 0.0001
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, default=3, help="Number of lag years")
    parser.add_argument("--months", type=int, default=2, help="Number of months that is going to be used for training. For historical: 6, 12, 18, 24, 30 36 For periodic: 1, 2")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--prediction_month", type=int, default=1, help="Number of months to predict at output")
    parser.add_argument("--positional_encoding", type=bool, default=False)
    parser.add_argument("--template_path", type=str, default="../experiments/CMIP6-Dropout04_Calib", help="Path to save the model")
    parser.add_argument("--model", type=str, default="CMIP6-Dropout04_Calib", help="Name of the model. Required for model_select function to chose the right model.")
    parser.add_argument("--kl_coef", type=int, default=16, help="Only required when training bayesian models")
    parser.add_argument("--climate_var", type=str , default="tas", choices=["pr", "psl", "tas", "uas", "vas", "zg500"])
    parser.add_argument("--lr", type=float , default=1e-5)
    parser.add_argument("--weight_decay", type=float , default=1e-5)
    parser.add_argument("--dropout", type=bool , default=False)
    parser.add_argument("--deeper", type=bool , default=False)
    parser.add_argument("--ensembles", type=int , default=5)

    parser.add_argument("--shell", "--transport", "--iopub", "-f", "--fff", "--ip", "--stdin", "--control", "--hb", "--Session.signature_scheme", "--Session.key", help="a dummy argument to fool ipython", default="1")

    args = parser.parse_args()

    if 0 < args.months < 3 and args.years > 4:
        raise ValueError("Experiment not found.")

    return args

