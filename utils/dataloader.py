import pandas as pd
import numpy as np
import geopandas as gpd
from tqdm import tqdm
from pathlib import Path
import pickle as pickle
from shapely import wkt

from sklearn.preprocessing import OrdinalEncoder
import os

from joblib import Parallel, delayed
from joblib import parallel_backend
import torch
from torch.nn.utils.rnn import pad_sequence

import trackintel as ti


class sp_loc_dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        source_root,
        user=None,
        dataset="geolife",
        data_type="train",
        previous_day=7,
        model_type="transformer",
        day_selection="default",
    ):
        self.root = source_root
        self.user = user
        self.data_type = data_type
        self.previous_day = previous_day
        self.model_type = model_type
        self.dataset = dataset
        self.day_selection = day_selection

        # check whether to train individual models
        if user is None:
            self.is_individual_model = False
        else:
            self.is_individual_model = True

        if self.is_individual_model:
            # create seperate folders for each individual
            self.data_dir = os.path.join(
                source_root, "temp", "individual", f"{self.dataset}_{self.model_type}_{previous_day}"
            )
            if not os.path.exists(self.data_dir):
                os.makedirs(self.data_dir)
            save_path = os.path.join(self.data_dir, f"{user}_{data_type}.pk")
        else:
            # define data storing dir
            self.data_dir = os.path.join(source_root, "temp")
            if day_selection == "default":
                save_path = os.path.join(
                    self.data_dir, f"{self.dataset}_{self.model_type}_{previous_day}_{data_type}.pk"
                )
            else:
                save_path = os.path.join(
                    self.data_dir, f"{self.dataset}_{''.join(str(x) for x in day_selection)}_{data_type}.pk"
                )

        # if the file is pre-generated we load the file, otherwise run self.generate_data()
        if Path(save_path).is_file():
            self.data = pickle.load(open(save_path, "rb"))
        else:
            parent = Path(save_path).parent.absolute()
            if not os.path.exists(parent):
                os.makedirs(parent)
            self.data = self.generate_data()

        self.len = len(self.data)

    def __len__(self):
        """Return the length of the current dataloader."""
        return self.len

    def __getitem__(self, idx):
        """Get a single sample."""
        selected = self.data[idx]

        return_dict = {}
        # [sequence_len]
        x = torch.tensor(selected["X"])
        # [1]
        y = torch.tensor(selected["Y"])

        # [1]
        return_dict["user"] = torch.tensor(selected["user_X"][0])
        # [sequence_len] in a quater of an hour
        return_dict["time"] = torch.tensor(selected["start_min_X"] // 15)
        #
        return_dict["diff"] = torch.tensor(selected["diff"])

        # [sequence_len]
        return_dict["duration"] = torch.tensor(selected["dur_X"] // 30, dtype=torch.long)
        # [sequence_len]
        return_dict["weekday"] = torch.tensor(selected["weekday_X"])

        if self.dataset == "gc":
            # [sequence_len * buffer_num]
            return_dict["poi"] = torch.tensor(np.array(selected["poi_X"]), dtype=torch.float32)

        return x, y, return_dict

    def generate_data(self):
        if self.dataset == "gc":
            # the poi vectors
            self.poi_data = load_pk_file(os.path.join(self.root, "poiValues_lda_16_500.pk"))
            # self.poi_data = load_pk_file(os.path.join(self.root, "poiValues_lda_16_single.pk"))
            # the valid location ids for unifying comparision
        self.valid_ids = load_pk_file(os.path.join(self.root, f"valid_ids_{self.dataset}.pk"))

        # the location data
        ori_data = pd.read_csv(os.path.join(self.root, f"dataset_{self.dataset}.csv"))

        ori_data.sort_values(by=["user_id", "start_day", "start_min"], inplace=True)

        # truncate too long duration: > 2days to 2 days
        ori_data.loc[ori_data["duration"] > 60 * 24 * 2 - 1, "duration"] = 60 * 24 * 2 - 1

        # classify the datasets, user dependent 0.6, 0.2, 0.2
        train_data, vali_data, test_data = self._splitDataset(ori_data)

        if self.is_individual_model:
            # save the total location number for each user

            # get the total number for each user
            total_num_location = train_data.groupby("user_id")["location_id"].max() + 1
            user_dict = total_num_location.to_dict()
            # define the data path to store
            save_path = os.path.join(self.data_dir, f"loc.pk")
            save_pk_file(save_path, user_dict)

        else:
            # encode unseen location in train as 1 (0 reserved for padding)
            # this saves (a bit of) #parameters when defining the model
            train_data, vali_data, test_data, enc = self._encode_loc(train_data, vali_data, test_data)

            if self.dataset == "gc":
                # re encode poi index, use the enc from self._encode_loc()
                self.poi_data["index"] = enc.transform(self.poi_data["index"].reshape(-1, 1)) + 2
                self.poi_data["index"] = np.squeeze(self.poi_data["index"], axis=-1)

                # change the poi values of unknown locations into 0
                idx = np.where(self.poi_data["index"] == 1)[0]
                self.poi_data["poiValues"][idx, :, :] = np.zeros_like(self.poi_data["poiValues"][idx, :, :])

        # preprocess the data into sequences
        train_records = self._preProcessDatasets(train_data, "train")
        validation_records = self._preProcessDatasets(vali_data, "validation")
        test_records = self._preProcessDatasets(test_data, "test")

        if self.data_type == "test":
            return test_records
        if self.data_type == "validation":
            return validation_records
        if self.data_type == "train":
            return train_records

    def _encode_loc(self, train, validation, test):
        """encode unseen locations in validation and test into 1 (0 reserved for padding)."""

        # fit to train
        enc = OrdinalEncoder(dtype=np.int64, handle_unknown="use_encoded_value", unknown_value=-1).fit(
            train["location_id"].values.reshape(-1, 1)
        )
        # apply to all. add 2 to account for unseen locations (1) and to account for 0 padding
        train["location_id"] = enc.transform(train["location_id"].values.reshape(-1, 1)) + 2
        validation["location_id"] = enc.transform(validation["location_id"].values.reshape(-1, 1)) + 2
        test["location_id"] = enc.transform(test["location_id"].values.reshape(-1, 1)) + 2

        return train, validation, test, enc

    def _splitDataset(self, totalData):
        """Split dataset into train, vali and test."""
        totalData = totalData.groupby("user_id", group_keys=False).apply(self.__getSplitDaysUser)

        train_data = totalData.loc[totalData["Dataset"] == "train"].copy()
        vali_data = totalData.loc[totalData["Dataset"] == "vali"].copy()
        test_data = totalData.loc[totalData["Dataset"] == "test"].copy()

        # final cleaning
        train_data.drop(columns={"Dataset"}, inplace=True)
        vali_data.drop(columns={"Dataset"}, inplace=True)
        test_data.drop(columns={"Dataset"}, inplace=True)

        return train_data, vali_data, test_data

    def __getSplitDaysUser(self, df):
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

        # 3. We encode the location ids for each user if user specific (tested)
        if self.is_individual_model:

            enc = OrdinalEncoder(
                dtype=np.int64,
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            ).fit(df.loc[df["Dataset"] == "train", "location_id"].values.reshape(-1, 1))
            # add 2 to account for unseen locations and to account for 0 padding
            df["location_id"] = enc.transform(df["location_id"].values.reshape(-1, 1)) + 2

        return df

    def _preProcessDatasets(self, data, dataset_type):
        """Generate the datasets and save to the disk."""
        valid_records = self.__getValidSequence(data)

        if self.is_individual_model:
            # we save each user individually
            for i, records in enumerate(valid_records):
                save_path = os.path.join(self.data_dir, f"{i+1}_{dataset_type}.pk")
                save_pk_file(save_path, records)
            return_data = valid_records[self.user - 1]

        else:
            valid_records = [item for sublist in valid_records for item in sublist]

            if self.day_selection == "default":
                save_path = os.path.join(
                    self.data_dir, f"{self.dataset}_{self.model_type}_{self.previous_day}_{dataset_type}.pk"
                )
            else:
                save_path = os.path.join(
                    self.data_dir, f"{self.dataset}_{''.join(str(x) for x in self.day_selection)}_{dataset_type}.pk"
                )

            save_pk_file(save_path, valid_records)

            return_data = valid_records

        return return_data

    def __getValidSequence(self, input_df):
        """Get the valid sequences.

        According to the input previous_day and day_selection.
        The length of the history sequence should >2, i.e., whole sequence >3.

        We use parallel computing on users (generation is independet of users) to speed up the process.
        """
        valid_user_ls = applyParallel(input_df.groupby("user_id"), self.___getValidSequenceUser, n_jobs=-1)
        return valid_user_ls

    def ___getValidSequenceUser(self, df):
        """Get the valid sequences per user.

        input df contains location history for a single user.
        """

        df.reset_index(drop=True, inplace=True)

        data_single_user = []

        # get the day of tracking
        min_days = df["start_day"].min()
        df["diff_day"] = df["start_day"] - min_days

        for index, row in df.iterrows():
            # exclude the first records that do not include enough previous_day
            if row["diff_day"] < self.previous_day:
                continue

            # get the history records [curr-previous, curr]
            hist = df.iloc[:index]
            hist = hist.loc[(hist["start_day"] >= (row["start_day"] - self.previous_day))]

            # should be in the valid user ids
            if not (row["id"] in self.valid_ids):
                continue

            if self.day_selection != "default":
                # get only records from selected days
                hist["diff"] = row["diff_day"] - hist["diff_day"]
                hist = hist.loc[hist["diff"].isin(self.day_selection)]
                if len(hist) < 2:
                    continue

            data_dict = {}
            # get the features: location, user, weekday, start time, duration, diff to curr day, and poi
            data_dict["X"] = hist["location_id"].values
            data_dict["user_X"] = hist["user_id"].values
            data_dict["weekday_X"] = hist["weekday"].values
            data_dict["start_min_X"] = hist["start_min"].values
            data_dict["dur_X"] = hist["duration"].values
            data_dict["diff"] = (row["diff_day"] - hist["diff_day"]).astype(int).values

            if self.dataset == "gc":
                data_dict["poi_X"] = self._getPOIRepresentation(data_dict["X"])

            # the next location is the target
            data_dict["Y"] = int(row["location_id"])

            # append the single sample to list
            data_single_user.append(data_dict)

        return data_single_user

    def _getPOIRepresentation(self, loc_ls):
        """Given a sequence of location index, return their poi vector representation.

        Return size of len(loc_ls) * dim_vec * buff_num
        """

        poi_rep = []
        for loc in loc_ls:
            idx = np.where(self.poi_data["index"] == loc)[0][0]
            matrix = self.poi_data["poiValues"][idx, :, :]
            poi_rep.append(matrix)

        return poi_rep


def save_pk_file(save_path, data):
    """Function to save data to pickle format given data and path."""
    with open(save_path, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pk_file(save_path):
    """Function to load data from pickle format given path."""
    return pickle.load(open(save_path, "rb"))


def applyParallel(dfGrouped, func, n_jobs, print_progress=True, **kwargs):
    """
    Funtion warpper to parallelize funtions after .groupby().
    Parameters
    ----------
    dfGrouped: pd.DataFrameGroupBy
        The groupby object after calling df.groupby(COLUMN).
    func: function
        Function to apply to the dfGrouped object, i.e., dfGrouped.apply(func).
    n_jobs: int
        The maximum number of concurrently running jobs. If -1 all CPUs are used. If 1 is given, no parallel
        computing code is used at all, which is useful for debugging. See
        https://joblib.readthedocs.io/en/latest/parallel.html#parallel-reference-documentation
        for a detailed description
    print_progress: boolean
        If set to True print the progress of apply.
    **kwargs:
        Other arguments passed to func.
    Returns
    -------
    pd.DataFrame:
        The result of dfGrouped.apply(func)
    """
    with parallel_backend("threading", n_jobs=n_jobs):
        df_ls = Parallel()(delayed(func)(group, **kwargs) for _, group in tqdm(dfGrouped, disable=not print_progress))
    return df_ls


def collate_fn(batch):
    """function to collate data samples into batch tensors."""
    x_batch, y_batch = [], []

    # get one sample batch
    x_dict_batch = {"len": []}
    for key in batch[0][-1]:
        x_dict_batch[key] = []

    for src_sample, tgt_sample, return_dict in batch:
        x_batch.append(src_sample)
        y_batch.append(tgt_sample)

        x_dict_batch["len"].append(len(src_sample))
        for key in return_dict:
            x_dict_batch[key].append(return_dict[key])

    x_batch = pad_sequence(x_batch)
    y_batch = torch.tensor(y_batch, dtype=torch.int64)

    # x_dict_batch
    x_dict_batch["user"] = torch.tensor(x_dict_batch["user"], dtype=torch.int64)
    x_dict_batch["len"] = torch.tensor(x_dict_batch["len"], dtype=torch.int64)
    for key in x_dict_batch:
        if key in ["user", "len", "history_count"]:
            continue
        x_dict_batch[key] = pad_sequence(x_dict_batch[key])

    return x_batch, y_batch, x_dict_batch


def test_dataloader(train_loader):

    batch_size = train_loader.batch_size
    # print(batch_size)

    x_shape = 0
    x_dict_shape = 0
    for batch_idx, (x, y, x_dict) in tqdm(enumerate(train_loader)):
        # print("batch_idx ", batch_idx)
        x_shape += x.shape[0]
        x_dict_shape += x_dict["duration"].shape[0]
        # print(x_dict["user"].shape)
        # print(x_dict["poi"].shape)

        # print(, batch_len)

        # print(data)
        # print(target)
        # print(dict)
        # if batch_idx > 10:
        #     break
    print(x_shape / len(train_loader))
    print(x_dict_shape / len(train_loader))


if __name__ == "__main__":
    source_root = r"./data/"

    dataset_train = sp_loc_dataset(source_root, dataset="geolife", data_type="train", previous_day=7)
    kwds_train = {"shuffle": False, "num_workers": 0, "batch_size": 2}
    train_loader = torch.utils.data.DataLoader(dataset_train, collate_fn=collate_fn, **kwds_train)

    test_dataloader(train_loader)
