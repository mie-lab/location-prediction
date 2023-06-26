import torch, os
import numpy as np
import pandas as pd
import pickle as pickle
import json
from datetime import datetime
from easydict import EasyDict as edict

from utils.dataloader import sp_loc_dataset, collate_fn
from utils.train import trainNet, validate, get_performance_dict
from models.MHSA import TransEncoder
from utils.utils import load_config, setup_seed

setup_seed(42)

# change
def get_dataloaders(config):

    kwds_train = {"shuffle": True, "num_workers": 0, "drop_last": True, "batch_size": config["batch_size"]}
    kwds_val = {"shuffle": False, "num_workers": 0, "batch_size": config["batch_size"]}
    kwds_test = {"shuffle": False, "num_workers": 0, "batch_size": config["batch_size"]}

    train_loader = []
    val_loader = []
    test_loader = []
    for user in range(1, config["total_user_num"]):
        dataset_train = sp_loc_dataset(
            config.source_root,
            user=user,
            data_type="train",
            previous_day=config.previous_day,
            model_type=config.networkName,
            dataset=config.dataset,
        )
        dataset_val = sp_loc_dataset(
            config.source_root,
            user=user,
            data_type="validation",
            previous_day=config.previous_day,
            model_type=config.networkName,
            dataset=config.dataset,
        )
        dataset_test = sp_loc_dataset(
            config.source_root,
            user=user,
            data_type="test",
            previous_day=config.previous_day,
            model_type=config.networkName,
            dataset=config.dataset,
        )

        current_train_loader = torch.utils.data.DataLoader(dataset_train, collate_fn=collate_fn, **kwds_train)
        current_val_loader = torch.utils.data.DataLoader(dataset_val, collate_fn=collate_fn, **kwds_val)
        current_test_loader = torch.utils.data.DataLoader(dataset_test, collate_fn=collate_fn, **kwds_test)

        train_loader.append(current_train_loader)
        val_loader.append(current_val_loader)
        test_loader.append(current_test_loader)

    print(len(train_loader), len(val_loader), len(test_loader))
    return train_loader, val_loader, test_loader


def get_models(config, device):
    total_params = 0

    # load the location numbers
    temp_data_path = os.path.join(
        config.source_root,
        "temp",
        "individual",
        f"{config.dataset}_{config.networkName}_{config.previous_day}",
        "loc.pk",
    )
    location_number_dict = pickle.load(open(temp_data_path, "rb"))

    model = []
    for user in range(1, config["total_user_num"]):
        location_number_user = location_number_dict[user]

        curr_model = TransEncoder(config=config, total_loc_num=location_number_user).to(device)

        model.append(curr_model)

        total_params += sum(p.numel() for p in curr_model.parameters() if p.requires_grad)

    print("Total number of trainable parameters: ", total_params)

    return model


def get_trainedNets(config, model, train_loader, val_loader, device):
    networkName = f"{config.dataset}_{config.networkName}_{config.previous_day}_individual"
    log_dir = os.path.join(config.save_root, f"{networkName}_{str(int(datetime.now().timestamp()))}")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    with open(os.path.join(log_dir, "conf.json"), "w") as fp:
        json.dump(config, fp, indent=4, sort_keys=True)

    user = 1
    best_model = []
    performance = []
    for curr_model, curr_train_loader, curr_val_loader in zip(model, train_loader, val_loader):
        curr_log_dir = log_dir + f"/{user}"
        if not os.path.exists(curr_log_dir):
            os.makedirs(curr_log_dir)

        curr_best_model, curr_performance = trainNet(
            config, curr_model, curr_train_loader, curr_val_loader, device, log_dir=curr_log_dir
        )

        best_model.append(curr_best_model)
        curr_performance["type"] = "vali"
        performance.append(curr_performance)
        user = user + 1

    perf_df = pd.DataFrame(performance)

    print(perf_df)
    print(
        "acc@1: {:.2f} f1: {:.2f} mrr: {:.2f}".format(
            perf_df["correct@1"].sum() / perf_df["total"].sum() * 100,
            perf_df["f1"].mean(),
            perf_df["rr"].sum() / perf_df["total"].sum() * 100,
        ),
    )

    perf_df.to_csv(log_dir + "/performance.csv")

    return best_model


def get_test_result(config, best_model, test_loader, device):
    res_dict = {
        "correct@1": 0,
        "correct@3": 0,
        "correct@5": 0,
        "correct@10": 0,
        "rr": 0,
        "f1": 0,
        "total": 0,
    }
    count = 0
    for curr_best_model, curr_test_loader in zip(best_model, test_loader):
        return_dict = validate(config, curr_best_model, curr_test_loader, device)
        res_dict["correct@1"] += return_dict["correct@1"]
        res_dict["correct@3"] += return_dict["correct@3"]
        res_dict["correct@5"] += return_dict["correct@5"]
        res_dict["correct@10"] += return_dict["correct@10"]
        res_dict["rr"] += return_dict["rr"]
        res_dict["f1"] += return_dict["f1"]
        res_dict["total"] += return_dict["total"]

        count += 1
    res_dict["f1"] = res_dict["f1"] / count
    performance = get_performance_dict(res_dict)

    return performance


if __name__ == "__main__":
    config = load_config("./config/geolife/ind_transformer.yml")

    config = edict(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    result_ls = []
    # get data
    train_loader, val_loader, test_loader = get_dataloaders(config)

    for _ in range(1):

        # get model
        model = get_models(config, device)

        # train
        model = get_trainedNets(config, model, train_loader, val_loader, device)

        # test
        perf = get_test_result(config, model, test_loader, device)

        print(perf)
        result_ls.append(perf)

    result_df = pd.DataFrame(result_ls)
    print(result_df)

    filename = os.path.join(
        config.save_root,
        f"{config.dataset}_{config.networkName}_0001_{str(int(datetime.now().timestamp()))}.csv",
    )
    result_df.to_csv(filename, index=False)
