import geopandas as gpd

import pandas as pd

from shapely import wkt
from tqdm import tqdm


import datetime

from trackintel.analysis.tracking_quality import temporal_tracking_quality, _split_overlaps


def preprocess_to_ti(df):
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

    diff = df["started_at"].iloc[1:].reset_index(drop=True) - df["finished_at"].iloc[:-1].reset_index(drop=True)
    df["diff"].iloc[:-1] = diff.dt.total_seconds()
    df["st_next"].iloc[:-1] = df["started_at"].iloc[1:].reset_index(drop=True)

    df.loc[df["diff"] < 0, "finished_at"] = df.loc[df["diff"] < 0, "st_next"]

    df["started_at"], df["finished_at"] = pd.to_datetime(df["started_at"]), pd.to_datetime(df["finished_at"])
    df["duration"] = (df["finished_at"] - df["started_at"]).dt.total_seconds()

    # print(df.loc[df["diff"] < 0])
    df.drop(columns=["diff", "st_next"], inplace=True)
    df.drop(index=df[df["duration"] <= 0].index, inplace=True)

    return df


def enrich_time_info(sp):
    sp = sp.groupby("user_id", group_keys=False).apply(_get_time)
    sp.drop(columns={"finished_at", "started_at"}, inplace=True)
    sp.sort_values(by=["user_id", "start_day", "start_min"], inplace=True)
    sp = sp.reset_index(drop=True)

    # 
    sp["location_id"] = sp["location_id"].astype(int)
    sp["user_id"] = sp["user_id"].astype(int)

    # final cleaning, reassign ids
    sp.index.name = "id"
    sp.reset_index(inplace=True)
    return sp


def _get_time(df):
    min_day = pd.to_datetime(df["started_at"].min().date())
    df["started_at"] = df["started_at"].dt.tz_localize(tz=None)
    df["finished_at"] = df["finished_at"].dt.tz_localize(tz=None)

    df["start_day"] = (df["started_at"] - min_day).dt.days
    df["end_day"] = (df["finished_at"] - min_day).dt.days

    df["start_min"] = df["started_at"].dt.hour * 60 + df["started_at"].dt.minute
    df["end_min"] = df["finished_at"].dt.hour * 60 + df["finished_at"].dt.minute
    df.loc[df["end_min"] == 0, "end_min"] = 24 * 60

    df["weekday"] = df["started_at"].dt.weekday
    return df


def calculate_user_quality(sp, trips, file_path, quality_filter):

    trips["started_at"] = pd.to_datetime(trips["started_at"]).dt.tz_localize(None)
    trips["finished_at"] = pd.to_datetime(trips["finished_at"]).dt.tz_localize(None)
    sp["started_at"] = pd.to_datetime(sp["started_at"]).dt.tz_localize(None)
    sp["finished_at"] = pd.to_datetime(sp["finished_at"]).dt.tz_localize(None)

    # merge trips and staypoints
    print("starting merge", sp.shape, trips.shape)
    sp["type"] = "sp"
    trips["type"] = "tpl"
    df_all = pd.concat([sp, trips])
    df_all = _split_overlaps(df_all, granularity="day")
    df_all["duration"] = (df_all["finished_at"] - df_all["started_at"]).dt.total_seconds()
    print("finished merge", df_all.shape)
    print("*" * 50)

    if "min_thres" in quality_filter:
        end_period = datetime.datetime(2017, 12, 26)
        df_all = df_all.loc[df_all["finished_at"] < end_period]

    print(len(df_all["user_id"].unique()))

    # get quality
    total_quality = temporal_tracking_quality(df_all, granularity="all")
    # get tracking days
    total_quality["days"] = (
        df_all.groupby("user_id").apply(lambda x: (x["finished_at"].max() - x["started_at"].min()).days).values
    )
    # filter based on days
    user_filter_day = (
        total_quality.loc[(total_quality["days"] > quality_filter["day_filter"])]
        .reset_index(drop=True)["user_id"]
        .unique()
    )

    sliding_quality = (
        df_all.groupby("user_id")
        .apply(_get_tracking_quality, window_size=quality_filter["window_size"])
        .reset_index(drop=True)
    )

    filter_after_day = sliding_quality.loc[sliding_quality["user_id"].isin(user_filter_day)]

    if "min_thres" in quality_filter:
        # filter based on quanlity
        filter_after_day = (
            filter_after_day.groupby("user_id")
            .apply(_filter_user, min_thres=quality_filter["min_thres"], mean_thres=quality_filter["mean_thres"])
            .reset_index(drop=True)
            .dropna()
        )

    filter_after_user_quality = filter_after_day.groupby("user_id", as_index=False)["quality"].mean()

    print("final selected user", filter_after_user_quality.shape[0])
    filter_after_user_quality.to_csv(file_path, index=False)
    return filter_after_user_quality["user_id"].values


def _filter_user(df, min_thres, mean_thres):
    consider = df.loc[df["quality"] != 0]
    if (consider["quality"].min() > min_thres) and (consider["quality"].mean() > mean_thres):
        return df


def _get_tracking_quality(df, window_size):

    weeks = (df["finished_at"].max() - df["started_at"].min()).days // 7
    start_date = df["started_at"].min().date()

    quality_list = []
    # construct the sliding week gdf
    for i in range(0, weeks - window_size + 1):
        curr_start = datetime.datetime.combine(start_date + datetime.timedelta(weeks=i), datetime.time())
        curr_end = datetime.datetime.combine(curr_start + datetime.timedelta(weeks=window_size), datetime.time())

        # the total df for this time window
        cAll_gdf = df.loc[(df["started_at"] >= curr_start) & (df["finished_at"] < curr_end)]
        if cAll_gdf.shape[0] == 0:
            continue
        total_sec = (curr_end - curr_start).total_seconds()

        quality_list.append([i, cAll_gdf["duration"].sum() / total_sec])
    ret = pd.DataFrame(quality_list, columns=["timestep", "quality"])
    ret["user_id"] = df["user_id"].unique()[0]
    return ret


def split_dataset(totalData):
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

def get_valid_sequence(input_df, previous_day=14):

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