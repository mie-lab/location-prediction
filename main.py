import torch, os
import numpy as np
import argparse
import pandas as pd
from datetime import datetime
import json

import torch.distributed as dist
import torch.multiprocessing as mp

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


def init_save_path(config, time_now, i):
    """define the path to save, and save the configuration file."""
    networkName = f"{config.dataset}_{config.networkName}"
    log_dir = os.path.join(config.save_root, f"{networkName}_{config.previous_day}_{str(time_now)}_{str(i)}")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    with open(os.path.join(log_dir, "conf.json"), "w") as fp:
        json.dump(config, fp, indent=4, sort_keys=True)

    return log_dir


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def main(rank, world_size, config, time_now):
    # setup the process groups
    setup(rank, world_size)

    torch.cuda.set_device(rank)

    result_ls = []
    for i in range(2):

        train_loader, val_loader, test_loader = get_dataloaders(rank, world_size, config)
        # save the conf
        log_dir = init_save_path(config, time_now, i)

        res_single = single_run(train_loader, val_loader, test_loader, config, rank, log_dir)

        data = {"tensor": res_single}

        # we have to create enough room to store the collected objects
        outputs = [None for _ in range(world_size)]
        # the first argument is the collected lists, the second argument is the data unique in each process
        dist.all_gather_object(outputs, data)
        # we only want to operate on the collected objects at master node
        if rank == 0:
            print(outputs[0]["tensor"])
            result_ls.extend(outputs[0]["tensor"])

    result_df = pd.DataFrame(result_ls)
    if rank == 0:
        print(result_df)

    train_type = "default"
    filename = os.path.join(
        config.save_root,
        f"{config.dataset}_{config.networkName}_{train_type}_{str(int(datetime.now().timestamp()))}.csv",
    )
    result_df.to_csv(filename, index=False)

    cleanup()


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


if __name__ == "__main__":

    world_size = 1

    time_now = int(datetime.now().timestamp())
    # load configs
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config", type=str, nargs="?", help=" Config file path.", default="config/foursquare/transformer.yml"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    config = edict(config)

    mp.spawn(
        main,
        args=(world_size, config, time_now),
        nprocs=world_size,
    )
