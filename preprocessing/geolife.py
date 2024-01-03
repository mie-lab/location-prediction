import json
import os
import pickle as pickle
from sklearn.preprocessing import OrdinalEncoder
from pathlib import Path

import pandas as pd
import numpy as np
import geopandas as gpd
from tqdm import tqdm
import argparse

# trackintel
from trackintel.io.dataset_reader import read_geolife
from trackintel.preprocessing.triplegs import generate_trips
import trackintel as ti

# from config import config
from utils import calculate_user_quality, enrich_time_info, split_dataset, get_valid_sequence


def get_dataset(config, epsilon=50, num_samples=2):
    """Construct the raw staypoint with location id dataset from geolife data."""
    # read file storage
    pfs, _ = read_geolife(config["raw_geolife"], print_progress=True)
    # generate staypoints
    pfs, sp = pfs.as_positionfixes.generate_staypoints(
        gap_threshold=24 * 60, include_last=True, print_progress=True, dist_threshold=200, time_threshold=30, n_jobs=-1
    )
    # create activity flag
    sp = sp.as_staypoints.create_activity_flag(method="time_threshold", time_threshold=25)

    ## select valid user, generate the file if user quality file is not generated
    quality_path = os.path.join(".", "data", "quality")
    quality_file = os.path.join(quality_path, "geolife_slide_filtered.csv")
    if Path(quality_file).is_file():
        valid_user = pd.read_csv(quality_file)["user_id"].values
    else:
        if not os.path.exists(quality_path):
            os.makedirs(quality_path)
        # generate triplegs
        pfs, tpls = pfs.as_positionfixes.generate_triplegs(sp)
        # the trackintel trip generation
        sp, tpls, trips = generate_trips(sp, tpls, add_geometry=False)

        quality_filter = {"day_filter": 50, "window_size": 10}
        valid_user = calculate_user_quality(sp.copy(), trips.copy(), quality_file, quality_filter)

    sp = sp.loc[sp["user_id"].isin(valid_user)]

    # filter activity staypoints
    sp = sp.loc[sp["is_activity"] == True]

    # generate locations
    sp, locs = sp.as_staypoints.generate_locations(
        epsilon=epsilon, num_samples=num_samples, distance_metric="haversine", agg_level="dataset", n_jobs=-1
    )
    # filter noise staypoints
    sp = sp.loc[~sp["location_id"].isna()].copy()
    print("After filter non-location staypoints: ", sp.shape[0])

    # save locations
    locs = locs[~locs.index.duplicated(keep="first")]
    filtered_locs = locs.loc[locs.index.isin(sp["location_id"].unique())]
    filtered_locs.as_locations.to_csv(os.path.join(".", "data", f"locations_geolife.csv"))
    print("Location size: ", sp["location_id"].unique().shape[0], filtered_locs.shape[0])

    sp = sp[["user_id", "started_at", "finished_at", "geom", "location_id"]]
    # merge staypoints
    sp_merged = sp.as_staypoints.merge_staypoints(
        triplegs=pd.DataFrame([]), max_time_gap="1min", agg={"location_id": "first"}
    )
    print("After staypoints merging: ", sp_merged.shape[0])
    # recalculate staypoint duration
    sp_merged["duration"] = (sp_merged["finished_at"] - sp_merged["started_at"]).dt.total_seconds() // 60

    # get the time info
    sp_time = enrich_time_info(sp_merged)

    print("User size: ", sp_time["user_id"].unique().shape[0])

    # save intermediate results for analysis
    sp_time.to_csv(f"./data/sp_time_temp_geolife.csv", index=False)

    #
    _filter_sp_history(sp_time)


def _filter_sp_history(sp):
    """To unify the comparision between different previous days"""
    # classify the datasets, user dependent 0.6, 0.2, 0.2
    train_data, vali_data, test_data = split_dataset(sp)

    # encode unseen locations in validation and test into 0
    enc = OrdinalEncoder(dtype=np.int64, handle_unknown="use_encoded_value", unknown_value=-1).fit(
        train_data["location_id"].values.reshape(-1, 1)
    )
    # add 2 to account for unseen locations and to account for 0 padding
    train_data["location_id"] = enc.transform(train_data["location_id"].values.reshape(-1, 1)) + 2
    vali_data["location_id"] = enc.transform(vali_data["location_id"].values.reshape(-1, 1)) + 2
    test_data["location_id"] = enc.transform(test_data["location_id"].values.reshape(-1, 1)) + 2

    # the days to consider when generating final_valid_id
    # previous_day_ls = list(np.arange(14) + 1)
    previous_day_ls = [7]
    all_ids = sp[["id"]].copy()

    # for each previous_day, get the valid staypoint id
    for previous_day in tqdm(previous_day_ls):
        valid_ids = get_valid_sequence(train_data, previous_day=previous_day)
        valid_ids.extend(get_valid_sequence(vali_data, previous_day=previous_day))
        valid_ids.extend(get_valid_sequence(test_data, previous_day=previous_day))

        all_ids[f"{previous_day}"] = 0
        all_ids.loc[all_ids["id"].isin(valid_ids), f"{previous_day}"] = 1

    # get the final valid staypoint id
    all_ids.set_index("id", inplace=True)
    final_valid_id = all_ids.loc[all_ids.sum(axis=1) == all_ids.shape[1]].reset_index()["id"].values

    # filter the user again based on final_valid_id:
    # if an user has no record in final_valid_id, we discard the user
    valid_users_train = train_data.loc[train_data["id"].isin(final_valid_id), "user_id"].unique()
    valid_users_vali = vali_data.loc[vali_data["id"].isin(final_valid_id), "user_id"].unique()
    valid_users_test = test_data.loc[test_data["id"].isin(final_valid_id), "user_id"].unique()

    valid_users = set.intersection(set(valid_users_train), set(valid_users_vali), set(valid_users_test))

    filtered_sp = sp.loc[sp["user_id"].isin(valid_users)].copy()

    train_data, vali_data, test_data = split_dataset(filtered_sp)

    # encode unseen locations in validation and test into 0
    enc = OrdinalEncoder(dtype=np.int64, handle_unknown="use_encoded_value", unknown_value=-1).fit(
        train_data["location_id"].values.reshape(-1, 1)
    )
    # add 2 to account for unseen locations and to account for 0 padding
    train_data["location_id"] = enc.transform(train_data["location_id"].values.reshape(-1, 1)) + 2
    print(
        f"Max location id:{train_data.location_id.max()}, unique location id:{train_data.location_id.unique().shape[0]}"
    )

    # after user filter, we reencode the users, to ensure the user_id is continues
    # we do not need to encode the user_id again in dataloader.py
    enc = OrdinalEncoder(dtype=np.int64)
    filtered_sp["user_id"] = enc.fit_transform(filtered_sp["user_id"].values.reshape(-1, 1)) + 1

    # save the valid_ids and dataset
    data_path = f"./data/valid_ids_geolife.pk"
    with open(data_path, "wb") as handle:
        pickle.dump(final_valid_id, handle, protocol=pickle.HIGHEST_PROTOCOL)
    filtered_sp.to_csv(f"./data/dataSet_geolife.csv", index=False)

    print("Final user size: ", filtered_sp["user_id"].unique().shape[0])


if __name__ == "__main__":
    DBLOGIN_FILE = os.path.join(".", "paths.json")
    with open(DBLOGIN_FILE) as json_file:
        CONFIG = json.load(json_file)

    parser = argparse.ArgumentParser()
    parser.add_argument("epsilon", type=int, nargs="?", help="epsilon for dbscan to detect locations", default=20)
    args = parser.parse_args()

    get_dataset(config=CONFIG, epsilon=args.epsilon)
