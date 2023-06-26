import pickle
import pandas as pd
import numpy as np
import geopandas as gpd
from tqdm import tqdm
import os

from scipy import stats
import time
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import f1_score, recall_score

import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["figure.dpi"] = 300
matplotlib.rcParams["xtick.labelsize"] = 13
matplotlib.rcParams["ytick.labelsize"] = 13
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
np.random.seed(0)


def splitDataset(totalData):
    """Split dataset into train, vali and test."""
    totalData = totalData.groupby("user_id").apply(getSplitDaysUser)

    train_data = totalData.loc[totalData["Dataset"] == "train"].copy()
    vali_data = totalData.loc[totalData["Dataset"] == "vali"].copy()
    test_data = totalData.loc[totalData["Dataset"] == "test"].copy()

    # final cleaning
    train_data.drop(columns={"Dataset"}, inplace=True)
    vali_data.drop(columns={"Dataset"}, inplace=True)
    test_data.drop(columns={"Dataset"}, inplace=True)

    return train_data, vali_data, test_data


def getSplitDaysUser(df):
    """Split the dataset according to the tracked day of each user."""
    maxDay = df["start_day"].max()
    train_split = maxDay * 0.6
    vali_split = maxDay * 0.8

    df["Dataset"] = "test"
    df.loc[df["start_day"] < train_split, "Dataset"] = "train"
    df.loc[
        (df["start_day"] >= train_split) & (df["start_day"] < vali_split),
        "Dataset",
    ] = "vali"

    return df


def markov_transition_prob(df, n=1):
    COLUMNS = [f"loc_{i+1}" for i in range(n)]
    COLUMNS.append("toLoc")

    locSequence = pd.DataFrame(columns=COLUMNS)

    locSequence["toLoc"] = df.iloc[n:]["location_id"].values
    for i in range(n):
        locSequence[f"loc_{i+1}"] = df.iloc[i : -n + i]["location_id"].values
    return locSequence.groupby(by=COLUMNS).size().to_frame("size").reset_index()


def get_true_pred_pair(locSequence, df, n=1):
    testSeries = df["location_id"].values

    true_ls = []
    pred_ls = []

    for i in range(testSeries.shape[0] - n):
        locCurr = testSeries[i : i + n + 1]
        numbLoc = n

        # loop until finds a match
        while True:
            res_df = locSequence
            for j in range(n - numbLoc, n):
                res_df = res_df.loc[res_df[f"loc_{j+1}"] == locCurr[j]]
            res_df = res_df.sort_values(by="size", ascending=False)

            if res_df.shape[0]:  # if the dataframe contains entry, stop finding
                # choose the location which are visited most often for the matches
                pred = res_df["toLoc"].drop_duplicates().values
                break
            # decrese the number of location history considered
            numbLoc -= 1
            if numbLoc == 0:
                pred = np.zeros(10)
                break

        true_ls.append(locCurr[-1])
        pred_ls.append(pred)
    return true_ls, pred_ls


def get_performance_measure(true_ls, pred_ls):
    acc_ls = [1, 5, 10]

    res = []
    ndcg_ls = []

    correct_ls = [0, 0, 0]
    # total number
    res.append(len(true_ls))

    for true, pred in zip(true_ls, pred_ls):
        for i, top_acc in enumerate(acc_ls):

            if pred.shape[-1] < top_acc:
                top_acc = pred.shape[-1]

            if true in pred[:top_acc]:
                correct_ls[i] += 1

        # ndcg calculation
        target = pred[:10] if pred.shape[-1] < 10 else pred
        idx = np.where(true == target)[0]
        if len(idx) == 0:
            ndcg_ls.append(0)
        else:
            ndcg_ls.append(1 / np.log2(idx[0] + 1 + 1))

    res.extend(correct_ls)

    top1 = [pred[0] for pred in pred_ls]
    f1 = f1_score(true_ls, top1, average="weighted")
    recall = recall_score(true_ls, top1, average="weighted")

    res.append(f1)
    res.append(recall)
    res.append(np.mean(ndcg_ls))

    # rr
    rank_ls = []
    for true, pred in zip(true_ls, pred_ls):
        rank = np.where(pred == true)[0] + 1
        # (np.nonzero(pred == true)[0] + 1).astype(float)
        if len(rank):
            rank_ls.append(rank[0])
        else:
            rank_ls.append(0)
    rank = np.array(rank_ls, dtype=float)

    #
    rank = np.divide(1.0, rank, out=np.zeros_like(rank), where=rank != 0)
    # rank[rank == np.inf] = 0
    # append the result
    res.append(rank.sum())

    return pd.Series(res, index=["total", "correct@1", "correct@5", "correct@10", "f1", "recall", "ndcg", "rr"])


def get_markov_res(train, test, n=2):
    locSeq_df = markov_transition_prob(train, n=n)

    # true_ls, pred_ls = get_true_pred_pair(locSeq_df, test, n=n)

    # print(locSeq)
    return get_true_pred_pair(locSeq_df, test, n=n)


#


#  the number of previous locations considered (n-Markov)
n = 1

#
source_root = r"D:\Code\location_prediction\data"
# "gc" or "geolife" or "foursquare" or "gowalla"
dataset = "foursquare"
# read data
inputData = pd.read_csv(os.path.join(source_root, f"dataSet_{dataset}.csv"))
inputData.sort_values(by=["user_id", "start_day", "start_min"], inplace=True)

# split data
train_data, vali_data, test_data = splitDataset(inputData)

enc = OrdinalEncoder(dtype=np.int64, handle_unknown="use_encoded_value", unknown_value=-1).fit(
    train_data["location_id"].values.reshape(-1, 1)
)
# apply to all. add 2 to account for unseen locations (1) and to account for 0 padding
train_data["location_id"] = enc.transform(train_data["location_id"].values.reshape(-1, 1)) + 2
vali_data["location_id"] = enc.transform(vali_data["location_id"].values.reshape(-1, 1)) + 2
test_data["location_id"] = enc.transform(test_data["location_id"].values.reshape(-1, 1)) + 2

# filter records that we do not consider
valid_ids = pickle.load(open((os.path.join(source_root, f"valid_ids_{dataset}.pk")), "rb"))

# train vali and test then contains the same records as our dataloader
train_data = train_data.loc[train_data["id"].isin(valid_ids)]
# vali_data = vali_data.loc[vali_data["id"].isin(valid_ids)]
test_data = test_data.loc[test_data["id"].isin(valid_ids)]

training_start_time = time.time()
true_all_ls = []
pred_all_ls = []
total_parameter = 0
for user in tqdm(train_data["user_id"].unique()):

    # get the train and test sets for each user
    curr_train = train_data.loc[train_data["user_id"] == user]
    curr_test = test_data.loc[test_data["user_id"] == user]
    # get the results
    total_parameter += curr_train["location_id"].unique().shape[0] ** 2
    true_ls, pred_ls = get_markov_res(curr_train, curr_test, n=n)
    true_all_ls.extend(true_ls)
    pred_all_ls.extend(pred_ls)
print("Training finished.\t Time: {:.2f}s".format((time.time() - training_start_time)))
print("Total parameters: {:d}".format(total_parameter))
result = get_performance_measure(true_all_ls, pred_all_ls)

print(result["correct@1"].sum() / result["total"].sum() * 100, result["recall"] * 100)
print(result["correct@5"].sum() / result["total"].sum() * 100)
print(result["correct@10"].sum() / result["total"].sum() * 100)
print(result["rr"].sum() / result["total"].sum() * 100)
print(result["f1"] * 100)
print(result["ndcg"] * 100)
