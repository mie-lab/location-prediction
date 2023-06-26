import sys, os
import pandas as pd
import numpy as np

import torch
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import f1_score

import time
from timeit import default_timer as timer

from transformers import get_linear_schedule_with_warmup

from utils.earlystopping import EarlyStopping
from utils.dataloader import load_pk_file


def get_performance_dict(return_dict):
    perf = {
        "correct@1": return_dict["correct@1"],
        "correct@3": return_dict["correct@3"],
        "correct@5": return_dict["correct@5"],
        "correct@10": return_dict["correct@10"],
        "rr": return_dict["rr"],
        "ndcg": return_dict["ndcg"],
        "f1": return_dict["f1"],
        "total": return_dict["total"],
    }

    perf["acc@1"] = perf["correct@1"] / perf["total"] * 100
    perf["acc@5"] = perf["correct@5"] / perf["total"] * 100
    perf["acc@10"] = perf["correct@10"] / perf["total"] * 100
    perf["mrr"] = perf["rr"] / perf["total"] * 100
    perf["ndcg"] = perf["ndcg"] / perf["total"] * 100

    return perf


def send_to_device(inputs, device, config):
    x, y, x_dict = inputs
    if config.networkName == "deepmove":
        x = (x[0].to(device), x[1].to(device))

        for key in x_dict[0]:
            x_dict[0][key] = x_dict[0][key].to(device)
        for key in x_dict[1]:
            x_dict[1][key] = x_dict[1][key].to(device)
    else:
        x = x.to(device)
        for key in x_dict:
            x_dict[key] = x_dict[key].to(device)
    y = y.to(device)

    return x, y, x_dict


def calculate_correct_total_prediction(logits, true_y):

    # top_ = torch.eq(torch.argmax(logits, dim=-1), true_y).sum().cpu().numpy()
    top1 = []
    result_ls = []
    for k in [1, 3, 5, 10]:
        if logits.shape[-1] < k:
            k = logits.shape[-1]
        prediction = torch.topk(logits, k=k, dim=-1).indices
        # f1 score
        if k == 1:
            top1 = torch.squeeze(prediction).cpu()
            # f1 = f1_score(true_y.cpu(), prediction.cpu(), average="weighted")

        top_k = torch.eq(true_y[:, None], prediction).any(dim=1).sum().cpu().numpy()
        # top_k = np.sum([curr_y in pred for pred, curr_y in zip(prediction, true_y)])
        result_ls.append(top_k)
    # f1 score
    # result_ls.append(f1)
    # rr
    result_ls.append(get_mrr(logits, true_y))
    # ndcg
    result_ls.append(get_ndcg(logits, true_y))

    # total
    result_ls.append(true_y.shape[0])

    return np.array(result_ls, dtype=np.float32), true_y.cpu(), top1


def get_mrr(prediction, targets):
    """
    Calculates the MRR score for the given predictions and targets.

    Args:
        prediction (Bxk): torch.LongTensor. the softmax output of the model.
        targets (B): torch.LongTensor. actual target indices.

    Returns:
        the sum rr score
    """
    index = torch.argsort(prediction, dim=-1, descending=True)
    hits = (targets.unsqueeze(-1).expand_as(index) == index).nonzero()
    ranks = (hits[:, -1] + 1).float()
    rranks = torch.reciprocal(ranks)

    return torch.sum(rranks).cpu().numpy()


def get_ndcg(prediction, targets, k=10):
    """
    Calculates the NDCG score for the given predictions and targets.

    Args:
        prediction (Bxk): torch.LongTensor. the softmax output of the model.
        targets (B): torch.LongTensor. actual target indices.

    Returns:
        the sum rr score
    """
    index = torch.argsort(prediction, dim=-1, descending=True)
    hits = (targets.unsqueeze(-1).expand_as(index) == index).nonzero()
    ranks = (hits[:, -1] + 1).float().cpu().numpy()

    not_considered_idx = ranks > k
    ndcg = 1 / np.log2(ranks + 1)
    ndcg[not_considered_idx] = 0

    return np.sum(ndcg)


def get_optimizer(config, model):
    # define the optimizer & learning rate
    if config.optimizer == "SGD":
        optim = torch.optim.SGD(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
            momentum=config.momentum,
            nesterov=True,
        )
    elif config.optimizer == "Adam":
        optim = torch.optim.Adam(
            model.parameters(),
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay,
        )

    return optim


def trainNet(config, model, train_loader, val_loader, device, log_dir):

    performance = {}

    optim = get_optimizer(config, model)

    # define learning rate schedule
    scheduler = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=len(train_loader) * config.num_warmup_epochs,
        num_training_steps=len(train_loader) * config.num_training_epochs,
    )
    scheduler_ES = StepLR(optim, step_size=config.lr_step_size, gamma=config.lr_gamma)
    if config.verbose:
        print("Current learning rate: ", scheduler.get_last_lr()[0])

    # Time for printing
    training_start_time = time.time()
    globaliter = 0
    scheduler_count = 0

    loc_geom = None
    if config.networkName == "mobtcast":
        loc_geom = load_pk_file(
            os.path.join(config.source_root, "temp", f"{config.dataset}_loc_{config.previous_day}.pk")
        )
        loc_geom = torch.tensor(loc_geom).to(device)

    # initialize the early_stopping object
    early_stopping = EarlyStopping(log_dir, patience=config["patience"], verbose=config.verbose, delta=0.001)

    # Loop for n_epochs
    for epoch in range(config.max_epoch):
        # train for one epoch
        globaliter = train(
            config, model, train_loader, optim, device, epoch, scheduler, scheduler_count, globaliter, loc_geom
        )

        # At the end of the epoch, do a pass on the validation set
        return_dict = validate(config, model, val_loader, device, loc_geom)

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(return_dict, model)

        if early_stopping.early_stop:
            if config.verbose:
                print("=" * 50)
                print("Early stopping")
            if scheduler_count == 2:
                performance = get_performance_dict(early_stopping.best_return_dict)
                print(
                    "Training finished.\t Time: {:.2f}s.\t acc@1: {:.2f}%".format(
                        (time.time() - training_start_time),
                        performance["acc@1"],
                    )
                )

                break

            scheduler_count += 1
            model.load_state_dict(torch.load(log_dir + "/checkpoint.pt"))
            early_stopping.early_stop = False
            early_stopping.counter = 0
            scheduler_ES.step()

        if config.verbose:
            # print("Current learning rate: {:.5f}".format(scheduler.get_last_lr()[0]))
            # print("Current learning rate: {:.5f}".format(scheduler_ES.get_last_lr()[0]))
            print("Current learning rate: {:.6f}".format(optim.param_groups[0]["lr"]))
            print("=" * 50)

        if config.debug == True:
            break

    return model, performance


def train(config, model, train_loader, optim, device, epoch, scheduler, scheduler_count, globaliter, loc_geom=None):

    model.train()
    running_loss = 0.0
    # 1, 3, 5, 10, f1, rr, total
    result_arr = np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    n_batches = len(train_loader)

    CEL = torch.nn.CrossEntropyLoss(reduction="mean", ignore_index=0)

    if config.networkName == "mobtcast":
        MSE = torch.nn.MSELoss(reduction="mean")

    # define start time
    start_time = time.time()
    optim.zero_grad(set_to_none=True)
    for i, inputs in enumerate(train_loader):
        globaliter += 1

        x, y, x_dict = send_to_device(inputs, device, config)
        # inputs, Y = send_to_device(inputs, Y, device, config)

        if config.networkName == "mobtcast":
            logits, pred_geoms = model(x, x_dict, device)

            # CEL
            loss_size = CEL(logits, y.reshape(-1))

            # predict
            y_geom = loc_geom[y - 2, :]
            loss_size += MSE(pred_geoms, y_geom)
            # consistent
            infered = torch.topk(logits, k=1, dim=-1).indices.squeeze(-1)
            infered_geom = loc_geom[infered - 2, :]
            loss_size += MSE(infered_geom, y_geom)
        else:
            logits = model(x, x_dict, device)

            loss_size = CEL(logits.view(-1, logits.shape[-1]), y.reshape(-1))

        optim.zero_grad(set_to_none=True)
        loss_size.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optim.step()
        if scheduler_count == 0:
            scheduler.step()

        # Print statistics
        running_loss += loss_size.item()

        batch_result_arr, _, _ = calculate_correct_total_prediction(logits, y)
        result_arr += batch_result_arr

        if (config.verbose) and ((i + 1) % config["print_step"] == 0):
            print(
                "Epoch {}, {:.1f}%\t loss: {:.3f} acc@1: {:.2f} mrr: {:.2f}, ndcg: {:.2f}, took: {:.2f}s \r".format(
                    epoch + 1,
                    100 * (i + 1) / n_batches,
                    running_loss / config["print_step"],
                    100 * result_arr[0] / result_arr[-1],
                    100 * result_arr[4] / result_arr[-1],
                    100 * result_arr[5] / result_arr[-1],
                    time.time() - start_time,
                ),
                end="",
                flush=True,
            )

            # Reset running loss and time
            running_loss = 0.0
            result_arr = np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
            start_time = time.time()

        if (config["debug"] == True) and (i > 20):
            break
    if config.verbose:
        print()
    return globaliter


def validate(config, model, data_loader, device, loc_geom=None):

    total_val_loss = 0
    true_ls = []
    top1_ls = []

    result_arr = np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    CEL = torch.nn.CrossEntropyLoss(reduction="mean", ignore_index=0)

    if config.networkName == "mobtcast":
        MSE = torch.nn.MSELoss(reduction="mean")
    # change to validation mode
    model.eval()
    with torch.no_grad():
        for inputs in data_loader:

            x, y, x_dict = send_to_device(inputs, device, config)

            if config.networkName == "mobtcast":
                logits, pred_geoms = model(x, x_dict, device)

                loss = CEL(logits, y.reshape(-1))
                # predict, subtract 2 to account for padding and unknown
                y_geom = loc_geom[y - 2, :]
                loss += MSE(pred_geoms, y_geom)
                # consistent
                infered = torch.topk(logits, k=1, dim=-1).indices.squeeze(-1)
                infered_geom = loc_geom[infered - 2, :]
                loss += MSE(infered_geom, y_geom)
            else:
                logits = model(x, x_dict, device)

                loss = CEL(logits.view(-1, logits.shape[-1]), y.reshape(-1))

            total_val_loss += loss.item()

            batch_result_arr, batch_true, batch_top1 = calculate_correct_total_prediction(logits, y)
            result_arr += batch_result_arr
            true_ls.extend(batch_true.tolist())
            if not batch_top1.shape:
                top1_ls.extend([batch_top1.tolist()])
            else:
                top1_ls.extend(batch_top1.tolist())

    # loss
    val_loss = total_val_loss / len(data_loader)
    # f1
    f1 = f1_score(true_ls, top1_ls, average="weighted")
    # result_arr[4] = result_arr[4] / len(data_loader)

    if config.verbose:
        print(
            "Validation loss = {:.2f} acc@1 = {:.2f} f1 = {:.2f} mrr = {:.2f}, ndcg = {:.2f}".format(
                val_loss,
                100 * result_arr[0] / result_arr[-1],
                100 * f1,
                100 * result_arr[4] / result_arr[-1],
                100 * result_arr[5] / result_arr[-1],
            ),
        )

    return {
        "val_loss": val_loss,
        "correct@1": result_arr[0],
        "correct@3": result_arr[1],
        "correct@5": result_arr[2],
        "correct@10": result_arr[3],
        "f1": f1,
        "rr": result_arr[4],
        "ndcg": result_arr[5],
        "total": result_arr[6],
    }


def test(config, model, data_loader, device):
    true_ls = []
    top1_ls = []
    time_ls = []

    result_arr = np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    result_dict = {}
    count_user = {}
    for i in range(1, config.total_user_num):
        result_dict[i] = np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        count_user[i] = 0

    # change to validation mode
    model.eval()
    with torch.no_grad():
        for inputs in data_loader:
            x, y, x_dict = send_to_device(inputs, device, config)

            start = timer()
            if config.networkName == "mobtcast":
                logits, _ = model(x, x_dict, device)
            else:
                logits = model(x, x_dict, device)

            time_ls.append((timer() - start) / y.shape[-1])

            if config.networkName != "deepmove":
                user_arr = x_dict["user"].cpu().detach().numpy()
                unique = np.unique(user_arr)
                for user in unique:
                    index = np.nonzero(user_arr == user)[0]

                    batch_user, _, _ = calculate_correct_total_prediction(logits[index, :], y[index])
                    result_dict[user] += batch_user
                    count_user[user] += 1

            batch_result_arr, batch_true, batch_top1 = calculate_correct_total_prediction(logits, y)
            result_arr += batch_result_arr
            true_ls.extend(batch_true.numpy())
            top1_ls.extend(batch_top1.numpy())
    print(np.mean(np.array(time_ls) * 1e6), np.std(np.array(time_ls) * 1e6))
    # f1
    f1 = f1_score(true_ls, top1_ls, average="weighted")

    if config.verbose:
        print(
            "acc@1 = {:.2f} f1 = {:.2f} mrr = {:.2f} ndcg = {:.2f}".format(
                100 * result_arr[0] / result_arr[-1],
                100 * f1,
                100 * result_arr[4] / result_arr[-1],
                100 * result_arr[5] / result_arr[-1],
            ),
        )

    return (
        {
            "correct@1": result_arr[0],
            "correct@3": result_arr[1],
            "correct@5": result_arr[2],
            "correct@10": result_arr[3],
            "f1": f1,
            "rr": result_arr[4],
            "ndcg": result_arr[5],
            "total": result_arr[6],
        },
        result_dict,
    )
