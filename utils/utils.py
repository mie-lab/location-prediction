import yaml
import random, torch, os
import numpy as np
import pandas as pd

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


from utils.train import trainNet, test, get_performance_dict
from utils.dataloader import sp_loc_dataset, collate_fn

from models.MHSA import TransEncoder


def load_config(path):
    """
    Loads config file:
    Args:
        path (str): path to the config file
    Returns:
        config (dict): dictionary of the configuration parameters, merge sub_dicts
    """
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    config = dict()
    for _, value in cfg.items():
        for k, v in value.items():
            config[k] = v

    return config


def setup_seed(seed):
    """
    fix random seed for deterministic training
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_trainedNets(config, model, train_loader, val_loader, device, log_dir):
    best_model, performance = trainNet(config, model, train_loader, val_loader, device, log_dir=log_dir)
    performance["type"] = "vali"

    return best_model, performance


def get_test_result(config, best_model, test_loader, device):

    return_dict, result_arr_user = test(config, best_model, test_loader, device)

    performance = get_performance_dict(return_dict)
    performance["type"] = "test"
    # print(performance)

    result_user_df = pd.DataFrame(result_arr_user).T
    result_user_df.columns = [
        "correct@1",
        "correct@3",
        "correct@5",
        "correct@10",
        "rr",
        "ndcg",
        "total",
    ]
    result_user_df.index.name = "user"

    return performance, result_user_df


def get_models(config, device):
    total_params = 0

    if config.networkName == "deepmove":
        model = Deepmove(config=config).to(device)
    elif config.networkName == "rnn":
        model = RNNs(config=config, total_loc_num=config.total_loc_num).to(device)
    elif config.networkName == "mobtcast":
        model = Mobtcast(config=config).to(device)
    else:
        model = TransEncoder(config=config, total_loc_num=config.total_loc_num).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # wrap the model with DDP
    # device_ids tell DDP where is your model
    # output_device tells DDP where to output, in our case, it is rank
    # find_unused_parameters=True instructs DDP to find unused output of the forward() function of any module in the model
    model = DDP(model, device_ids=[device], output_device=device, find_unused_parameters=False)

    print("Total number of trainable parameters: ", total_params, flush=True)

    return model


def get_dataloaders(rank, world_size, config):
    # def get_dataloaders(config):

    dataset_train = sp_loc_dataset(
        config.source_root,
        data_type="train",
        model_type=config.networkName,
        previous_day=config.previous_day,
        dataset=config.dataset,
        day_selection=config.day_selection,
    )
    dataset_val = sp_loc_dataset(
        config.source_root,
        data_type="validation",
        model_type=config.networkName,
        previous_day=config.previous_day,
        dataset=config.dataset,
        day_selection=config.day_selection,
    )
    dataset_test = sp_loc_dataset(
        config.source_root,
        data_type="test",
        model_type=config.networkName,
        previous_day=config.previous_day,
        dataset=config.dataset,
        day_selection=config.day_selection,
    )

    train_sampler = DistributedSampler(dataset_train, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    val_sampler = DistributedSampler(dataset_val, num_replicas=world_size, rank=rank, shuffle=False)
    test_sampler = DistributedSampler(dataset_test, num_replicas=world_size, rank=rank, shuffle=False)

    kwds_train = {
        # "shuffle": True,
        "num_workers": config["num_workers"],
        "drop_last": True,
        "batch_size": config["batch_size"],
        "pin_memory": False,
        "sampler": train_sampler,
    }
    kwds_val = {
        # "shuffle": False,
        "num_workers": config["num_workers"],
        "batch_size": config["batch_size"],
        "pin_memory": False,
        "sampler": val_sampler,
    }
    kwds_test = {
        # "shuffle": False,
        "num_workers": config["num_workers"],
        "batch_size": config["batch_size"],
        "pin_memory": False,
        "sampler": test_sampler,
    }
    if config.networkName == "deepmove":
        fn = deepmove_collate_fn
    else:
        fn = collate_fn

    train_loader = torch.utils.data.DataLoader(dataset_train, collate_fn=fn, **kwds_train)
    val_loader = torch.utils.data.DataLoader(dataset_val, collate_fn=fn, **kwds_val)
    test_loader = torch.utils.data.DataLoader(dataset_test, collate_fn=fn, **kwds_test)

    print(len(train_loader), len(val_loader), len(test_loader))
    return train_loader, val_loader, test_loader
