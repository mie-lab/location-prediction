import torch, os
import numpy as np
import argparse
import pandas as pd
from datetime import datetime
import json

from easydict import EasyDict as edict

from utils.utils import load_config, setup_seed, get_trainedNets, get_test_result, get_dataloaders, get_models

setup_seed(42)


def single_run(train_loader, val_loader, test_loader, config, device, log_dir):
    result_ls = []

    # get modelp
    model = get_models(config, device)

    # train
    model, perf = get_trainedNets(config, model, train_loader, val_loader, device, log_dir)
    result_ls.append(perf)

    # test
    perf, test_df = get_test_result(config, model, test_loader, device)
    test_df.to_csv(os.path.join(log_dir, "user_detail.csv"))

    result_ls.append(perf)

    return result_ls


def init_save_path(config):
    """define the path to save, and save the configuration file."""
    networkName = f"{config.dataset}_{config.networkName}"
    log_dir = os.path.join(
        config.save_root, f"{networkName}_{config.previous_day}_{str(int(datetime.now().timestamp()))}"
    )
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    with open(os.path.join(log_dir, "conf.json"), "w") as fp:
        json.dump(config, fp, indent=4, sort_keys=True)

    return log_dir


if __name__ == "__main__":
    # load configs
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, nargs="?", help=" Config file path.", default="config/geolife/transformer.yml")
    args = parser.parse_args()
    config = load_config(args.config)
    config = edict(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    result_ls = []
    train_loader, val_loader, test_loader = get_dataloaders(config)

    for i in range(2):
        # save the conf
        log_dir = init_save_path(config)
        res_single = single_run(train_loader, val_loader, test_loader, config, device, log_dir)

        print(res_single)
        result_ls.extend(res_single)

    result_df = pd.DataFrame(result_ls)
    print(result_df)

    train_type = "default"
    filename = os.path.join(
        config.save_root,
        f"{config.dataset}_{config.networkName}_{train_type}_{str(int(datetime.now().timestamp()))}.csv",
    )
    result_df.to_csv(filename, index=False)
