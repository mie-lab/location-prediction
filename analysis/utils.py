from sklearn.preprocessing import OrdinalEncoder
from tqdm import tqdm
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt

import multiprocessing
from joblib import Parallel, delayed

def filter_sp_history(sp, previous_day_ls):
    # classify the datasets, user dependent 0.6, 0.2, 0.2
    train_data, vali_data, test_data = _split_dataset(sp)

    # encode unseen locations in validation and test into 0
    enc = OrdinalEncoder(
        dtype=np.int64,
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    ).fit(train_data["location_id"].values.reshape(-1, 1))
    # add 2 to account for unseen locations and to account for 0 padding
    train_data["location_id"] = enc.transform(train_data["location_id"].values.reshape(-1, 1)) + 2
    vali_data["location_id"] = enc.transform(vali_data["location_id"].values.reshape(-1, 1)) + 2
    test_data["location_id"] = enc.transform(test_data["location_id"].values.reshape(-1, 1)) + 2

    # the days to consider when generating final_valid_id

    all_ids = sp[["id"]].copy()

    # for each previous_day, get the valid staypoint id
    for previous_day in tqdm(previous_day_ls):
        valid_ids = _get_valid_sequence(train_data, previous_day=previous_day)
        valid_ids.extend(_get_valid_sequence(vali_data, previous_day=previous_day))
        valid_ids.extend(_get_valid_sequence(test_data, previous_day=previous_day))

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
    filtered_sp = sp.loc[sp["user_id"].isin(valid_users)]

    print("Final user size: ", filtered_sp["user_id"].unique().shape[0])

    return filtered_sp["user_id"].unique()


def _split_dataset(totalData):
    """Split dataset into train, vali and test."""
    totalData = totalData.groupby("user_id",group_keys=False).apply(_get_split_days_user)

    train_data = totalData.loc[totalData["Dataset"] == "train"].copy()
    vali_data = totalData.loc[totalData["Dataset"] == "vali"].copy()
    test_data = totalData.loc[totalData["Dataset"] == "test"].copy()

    # final cleaning
    train_data.drop(columns={"Dataset"}, inplace=True)
    vali_data.drop(columns={"Dataset"}, inplace=True)
    test_data.drop(columns={"Dataset"}, inplace=True)

    return train_data, vali_data, test_data


def _get_split_days_user(df):
    """Split the dataset according to the tracked day of each user."""
    maxDay = df["start_day"].max()
    train_split = maxDay * 0.6
    validation_split = maxDay * 0.8

    df["Dataset"] = "test"
    df.loc[df["start_day"] < train_split, "Dataset"] = "train"
    df.loc[(df["start_day"] >= train_split) & (df["start_day"] < validation_split), "Dataset"] = "vali"

    return df


def _get_valid_sequence(input_df, previous_day=14):

    valid_id = []
    for user in input_df["user_id"].unique():
        df = input_df.loc[input_df["user_id"] == user].copy().reset_index(drop=True)

        min_days = df["start_day"].min()
        df["diff_day"] = df["start_day"] - min_days

        for index, row in df.iterrows():
            # exclude the first records
            if row["diff_day"] < previous_day:
                continue

            hist = df.iloc[:index]
            hist = hist.loc[(hist["start_day"] >= (row["start_day"] - previous_day))]
            if len(hist) < 3:
                continue

            valid_id.append(row["id"])

    return valid_id


def preprocess_to_trackintel(df):
    """Change dataframe to trackintel compatible format"""
    df.rename(
        columns={"userid": "user_id", "startt": "started_at", "endt": "finished_at", "dur_s": "duration"},
        inplace=True,
    )

    # read the time info
    df["started_at"] = pd.to_datetime(df["started_at"])
    df["finished_at"] = pd.to_datetime(df["finished_at"])
    df["started_at"] = df["started_at"].dt.tz_localize(tz="utc")
    df["finished_at"] = df["finished_at"].dt.tz_localize(tz="utc")

    df["duration"] = (df["finished_at"] - df["started_at"]).dt.total_seconds()
    # drop invalid
    df.drop(index=df[df["duration"] < 0].index, inplace=True)

    df.set_index("id", inplace=True)
    tqdm.pandas(desc="Load geometry")
    df["geom"] = df["geom"].progress_apply(wkt.loads)

    return gpd.GeoDataFrame(df, crs="EPSG:4326", geometry="geom")


def filter_duplicates(sp, tpls):

    # merge trips and staypoints
    sp["type"] = "sp"
    tpls["type"] = "tpl"
    df_all = pd.merge(sp, tpls, how="outer")

    df_all = df_all.groupby("user_id", as_index=False).apply(_alter_diff)
    sp = df_all.loc[df_all["type"] == "sp"].drop(columns=["type"])
    tpls = df_all.loc[df_all["type"] == "tpl"].drop(columns=["type"])

    sp = sp[["id", "user_id", "started_at", "finished_at", "geom", "duration", "is_activity"]]
    tpls = tpls[["id", "user_id", "started_at", "finished_at", "geom", "length_m", "duration", "mode"]]

    return sp.set_index("id"), tpls.set_index("id")


def _alter_diff(df):

    df.sort_values(by="started_at", inplace=True)
    df["diff"] = pd.NA
    df["st_next"] = pd.NA

    diff = df.iloc[1:]["started_at"].reset_index(drop=True) - df.iloc[:-1]["finished_at"].reset_index(drop=True)
    df["diff"][:-1] = diff.dt.total_seconds()
    df["st_next"][:-1] = df.iloc[1:]["started_at"].reset_index(drop=True)

    df.loc[df["diff"] < 0, "finished_at"] = df.loc[df["diff"] < 0, "st_next"]

    df["started_at"], df["finished_at"] = pd.to_datetime(df["started_at"]), pd.to_datetime(df["finished_at"])
    df["duration"] = (df["finished_at"] - df["started_at"]).dt.total_seconds()

    # print(df.loc[df["diff"] < 0])
    df.drop(columns=["diff", "st_next"], inplace=True)
    df.drop(index=df[df["duration"] <= 0].index, inplace=True)

    return df

def filter_within_swiss(stps, swissBound):
    """Spatial filtering of staypoints."""
    # save a copy of the original projection
    init_crs = stps.crs
    # project to projected system
    stps = stps.to_crs(swissBound.crs)

    ## parallel for speeding up
    stps["within"] = _apply_parallel(stps["geom"], _apply_extract, swissBound, n=-1)
    sp_swiss = stps[stps["within"] == True].copy()
    sp_swiss.drop(columns=["within"], inplace=True)

    return sp_swiss.to_crs(init_crs)

def _apply_extract(df, swissBound):
    """The func for _apply_parallel: judge whether inside a shp."""
    tqdm.pandas(desc="pandas bar")
    shp = swissBound["geometry"].to_numpy()[0]
    return df.progress_apply(lambda x: shp.contains(x))

def _apply_parallel(df, func, other, n=-1):
    """parallel apply for spending up."""
    if n is None:
        n = -1
    dflength = len(df)
    cpunum = multiprocessing.cpu_count()
    if dflength < cpunum:
        spnum = dflength
    if n < 0:
        spnum = cpunum + n + 1
    else:
        spnum = n or 1

    sp = list(range(dflength)[:: int(dflength / spnum + 0.5)])
    sp.append(dflength)
    slice_gen = (slice(*idx) for idx in zip(sp[:-1], sp[1:]))
    results = Parallel(n_jobs=n, verbose=0)(delayed(func)(df[slc], other) for slc in slice_gen)
    return pd.concat(results)