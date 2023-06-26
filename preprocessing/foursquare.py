import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
import os
import json

from joblib import Parallel, delayed

from tqdm import tqdm
import pickle as pickle

from utils import split_dataset, get_valid_sequence


def applyParallelPD(dfGrouped, func, n_jobs, print_progress=True, **kwargs):
    df_ls = Parallel(n_jobs=n_jobs)(
        delayed(func)(group, **kwargs) for _, group in tqdm(dfGrouped, disable=not print_progress)
    )
    return pd.concat(df_ls)


def _get_time(df):
    min_day = pd.to_datetime(df["started_at"].min().date())
    df["started_at"] = df["started_at"].dt.tz_localize(tz=None)
    df["start_day"] = (df["started_at"] - min_day).dt.days
    df["start_min"] = df["started_at"].dt.hour * 60 + df["started_at"].dt.minute
    df["weekday"] = df["started_at"].dt.weekday
    return df


def enrich_time_info(sp):
    tqdm.pandas(desc="Time enriching")
    sp = applyParallelPD(sp.groupby("user_id", group_keys=False), _get_time, n_jobs=-1, print_progress=True)
    #     sp.groupby("user_id", group_keys=False).progress_apply(_get_time)
    sp.drop(columns={"started_at", "offset", "UTC", "name", "category"}, inplace=True)
    sp.sort_values(by=["user_id", "start_day", "start_min"], inplace=True)
    sp = sp.reset_index(drop=True)

    # sp["location_id"] = sp["location_id"].astype(int)
    sp["user_id"] = sp["user_id"].astype(int)

    # final cleaning, reassign ids
    sp.index.name = "id"
    sp.reset_index(inplace=True)
    return sp


def get_dataset(config):
    # note: location does not start at 0
    foursquare = pd.read_csv(
        os.path.join(config[f"raw_foursquare"], "dataset_TSMC2014_NYC.txt"),
        sep="\t",
        header=None,
        parse_dates=[-1],
        names=["user_id", "location_id", "category", "name", "latitude", "longitude", "offset", "UTC"],
    )
    foursquare["started_at"] = foursquare["UTC"] + pd.to_timedelta(foursquare["offset"], unit="m")

    foursquare_enriched = enrich_time_info(foursquare)

    # Filter infrequent user
    user_size = foursquare_enriched.groupby(["user_id"]).size()
    valid_users = user_size[user_size > 10].index
    foursquare_enriched = foursquare_enriched.loc[foursquare_enriched["user_id"].isin(valid_users)]

    # Filter infrequent POIs
    poi_size = foursquare_enriched.groupby(["location_id"]).size()
    valid_pois = poi_size[poi_size > 10].index
    foursquare_enriched = foursquare_enriched.loc[foursquare_enriched["location_id"].isin(valid_pois)]

    # split into train vali and test
    train_data, vali_data, test_data = split_dataset(foursquare_enriched)

    # encode unseen locations in validation and test into 0
    enc = OrdinalEncoder(
        dtype=np.int64,
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    ).fit(train_data["location_id"].values.reshape(-1, 1))
    # add 2 to account for unseen locations and to account for 0 padding
    train_data["location_id"] = enc.transform(train_data["location_id"].values.reshape(-1, 1)) + 2
    print(
        f"Max location id:{train_data.location_id.max()}, unique location id:{train_data.location_id.unique().shape[0]}"
    )

    # the days to consider when generating final_valid_id
    all_ids = foursquare_enriched[["id"]].copy()

    # for each previous_day, get the valid staypoint id
    valid_ids = get_valid_sequence(train_data, previous_day=7)
    valid_ids.extend(get_valid_sequence(vali_data, previous_day=7))
    valid_ids.extend(get_valid_sequence(test_data, previous_day=7))

    all_ids["7"] = 0
    all_ids.loc[all_ids["id"].isin(valid_ids), f"7"] = 1

    # get the final valid staypoint id
    all_ids.set_index("id", inplace=True)
    final_valid_id = all_ids.loc[all_ids.sum(axis=1) == all_ids.shape[1]].reset_index()["id"].values

    # filter the user again based on final_valid_id:
    # if an user has no record in final_valid_id, we discard the user
    valid_users_train = train_data.loc[train_data["id"].isin(final_valid_id), "user_id"].unique()
    valid_users_vali = vali_data.loc[vali_data["id"].isin(final_valid_id), "user_id"].unique()
    valid_users_test = test_data.loc[test_data["id"].isin(final_valid_id), "user_id"].unique()

    valid_users = set.intersection(set(valid_users_train), set(valid_users_vali), set(valid_users_test))

    foursquare_afterUser = foursquare_enriched.loc[foursquare_enriched["user_id"].isin(valid_users)].copy()

    train_data, vali_data, test_data = split_dataset(foursquare_afterUser)

    # encode unseen locations in validation and test into 0
    enc = OrdinalEncoder(dtype=np.int64, handle_unknown="use_encoded_value", unknown_value=-1).fit(
        train_data["location_id"].values.reshape(-1, 1)
    )
    # add 2 to account for unseen locations and to account for 0 padding
    train_data["location_id"] = enc.transform(train_data["location_id"].values.reshape(-1, 1)) + 2
    print(
        f"Max location id:{train_data.location_id.max()}, unique location id:{train_data.location_id.unique().shape[0]}"
    )

    #  after user filter, we reencode the users, to ensure the user_id is continues
    # we do not need to encode the user_id again in dataloader.py
    enc = OrdinalEncoder(dtype=np.int64)
    foursquare_afterUser["user_id"] = enc.fit_transform(foursquare_afterUser["user_id"].values.reshape(-1, 1)) + 1

    print(
        f"Max user id:{foursquare_afterUser.user_id.max()}, unique user id:{foursquare_afterUser.user_id.unique().shape[0]}"
    )

    # normalize the coordinates. implementation following the original paper
    foursquare_afterUser["longitude"] = (
        2
        * (foursquare_afterUser["longitude"] - foursquare_afterUser["longitude"].min())
        / (foursquare_afterUser["longitude"].max() - foursquare_afterUser["longitude"].min())
        - 1
    )
    foursquare_afterUser["latitude"] = (
        2
        * (foursquare_afterUser["latitude"] - foursquare_afterUser["latitude"].min())
        / (foursquare_afterUser["latitude"].max() - foursquare_afterUser["latitude"].min())
        - 1
    )
    foursquare_loc = (
        foursquare_afterUser.groupby(["location_id"])
        .head(1)
        .drop(columns={"id", "user_id", "start_day", "start_min", "weekday"})
    )
    foursquare_loc = foursquare_loc.rename(columns={"location_id": "id"})

    # save the valid_ids and dataset
    data_path = f"./data/valid_ids_foursquare.pk"
    with open(data_path, "wb") as handle:
        pickle.dump(final_valid_id, handle, protocol=pickle.HIGHEST_PROTOCOL)
    foursquare_afterUser.to_csv(f"./data/dataSet_foursquare.csv", index=False)
    foursquare_loc.to_csv(f"./data/locations_foursquare.csv", index=False)

    print("Final user size: ", foursquare_afterUser["user_id"].unique().shape[0])


if __name__ == "__main__":
    DBLOGIN_FILE = os.path.join(".", "paths.json")
    with open(DBLOGIN_FILE) as json_file:
        CONFIG = json.load(json_file)

    get_dataset(config=CONFIG)
