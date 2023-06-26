import pandas as pd
import numpy as np
import geopandas as gpd

import networkx as nx
from networkx.algorithms import isomorphism

import json
import os
from sklearn.preprocessing import OrdinalEncoder

import trackintel as ti
from trackintel.analysis.tracking_quality import _split_overlaps

from entropy import random_entropy, uncorrelated_entropy, real_entropy
from utils import filter_sp_history, preprocess_to_trackintel, filter_duplicates, filter_within_swiss

import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as stats

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
matplotlib.rcParams["figure.dpi"] = 300
matplotlib.rcParams["xtick.labelsize"] = 13
matplotlib.rcParams["ytick.labelsize"] = 13
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)


def plot_entropy(if_gc=True):

    plt.figure(figsize=(8, 5))
    if if_gc:
        gc = _calculate_entropy(r".\data\dataSet_gc.csv")

        realDensity = stats.gaussian_kde(gc)
        x = np.linspace(np.min(gc) - 0.5, np.max(gc) + 0.5, 50)
        plt.plot(x, realDensity(x), color=colors[0], label="GC")
        plt.fill_between(x, 0, realDensity(x), facecolor="blue", alpha=0.2)

    geolife = _calculate_entropy(r".\data\dataSet_geolife.csv")

    realDensity = stats.gaussian_kde(geolife)
    x = np.linspace(np.min(geolife) - 0.5, np.max(geolife) + 0.5, 50)
    plt.plot(x, realDensity(x), color=colors[1], label="Geolife")
    plt.fill_between(x, 0, realDensity(x), facecolor="orange", alpha=0.2)

    plt.legend(loc="upper left", fontsize=13)
    plt.xlabel("Entropy", fontsize=16)
    plt.ylabel("PDF", fontsize=16)

    # plt.savefig(r'entropy.pdf', dpi=600, bbox_inches='tight')
    plt.show()


def _calculate_entropy(filename):
    sp = pd.read_csv(filename)
    sp.groupby("user_id")["location_id"]

    rand_entropy = random_entropy(sp, print_progress=False)
    uncorr_entropy = uncorrelated_entropy(sp, print_progress=False)
    rea_entropy = real_entropy(sp, print_progress=True)

    #     save
    res_df = rand_entropy.to_frame()
    res_df["uncorr_entropy"] = uncorr_entropy
    res_df["real_entropy"] = rea_entropy

    return res_df["real_entropy"].values


def get_stats(if_gc=True):
    if if_gc:
        file_name = "gc"
    else:
        file_name = "geolife"
    df = pd.read_csv(os.path.join(".", "data", f"dataSet_{file_name}.csv"))

    sp_num = df.groupby("user_id").apply(lambda x: len(x))
    print(f"{file_name}: staypoint number per user: {sp_num.mean():.2f}\t std: {sp_num.std():.2f}")

    loc_num = df.groupby("user_id").apply(lambda x: x["location_id"].unique().shape[0])
    print(f"{file_name}: location number per user: {loc_num.mean():.2f}\t std: {loc_num.std():.2f}")

    track_days = df.groupby("user_id").apply(lambda x: x["end_day"].max() - x["start_day"].min())
    print(f"{file_name}: tracked days per user: {track_days.mean():.2f}\t std: {track_days.std():.2f}")

    sp_num_day = df.groupby(["user_id", "start_day"]).size()
    print(f"{file_name}: staypoint number per user per day: {sp_num_day.mean():.2f}\t std: {sp_num_day.std():.2f}")


def plot_tracking_quality(if_gc=True):
    plt.figure(figsize=(8, 5))

    if if_gc:

        gc_quality = pd.read_csv(os.path.join(".", "data", "quality", "gc_slide_filtered.csv"))

        # get the user filter (same as preprocessing)
        gc = pd.read_csv(os.path.join(".", "data", "sp_time_temp_gc.csv"))
        filtered_user_gc = filter_sp_history(gc, list(np.arange(14) + 1))

        gc_quality = gc_quality.loc[gc_quality["user_id"].isin(filtered_user_gc), "quality"]
        print(f"GC: tracking quality: {gc_quality.mean():.2f}\t std: {gc_quality.std():.2f}")

        # kde and plotting
        density = stats.gaussian_kde(gc_quality)
        x = np.linspace(gc_quality.min(), 1, 50)
        plt.plot(x, density(x), color=colors[0], label="GC")
        plt.fill_between(x, 0, density(x), facecolor="blue", alpha=0.2)

    geo_quality = pd.read_csv(os.path.join(".", "data", "quality", "geolife_slide_filtered.csv"))

    # get the user filter (same as preprocessing)
    geolife = pd.read_csv(os.path.join(".", "data", "sp_time_temp_geolife.csv"))
    filtered_user_geo = filter_sp_history(geolife, list([7]))

    geo_quality = geo_quality.loc[geo_quality["user_id"].isin(filtered_user_geo), "quality"]
    print(f"Geolife: tracking quality: {geo_quality.mean():.2f}\t std: {geo_quality.std():.2f}")

    # kde and plotting
    density = stats.gaussian_kde(geo_quality)
    x = np.linspace(0, 1, 50)
    plt.plot(x, density(x), color=colors[1], label="Geolife")
    plt.fill_between(x, 0, density(x), facecolor="orange", alpha=0.2)

    plt.legend(loc="upper left", fontsize=13)
    plt.ylabel("PDF", fontsize=16)
    plt.xlabel("Tracking coverage", fontsize=16)

    # plt.savefig(r'track_quality.pdf', dpi=600, bbox_inches='tight')
    plt.show()


def motifs_preprocessing():

    # read file storage: we need the raw staypoints of the SBB GC1 dataset
    DBLOGIN_FILE = os.path.join(".", "paths.json")
    with open(DBLOGIN_FILE) as json_file:
        config = json.load(json_file)

    ## read and change name to trackintel format
    sp = pd.read_csv(os.path.join(config[f"raw_gc"], "stps.csv"))
    tpls = pd.read_csv(os.path.join(config[f"raw_gc"], "tpls.csv"))

    sp.rename(columns={"activity": "is_activity"}, inplace=True)

    sp = preprocess_to_trackintel(sp)
    tpls = preprocess_to_trackintel(tpls)

    # ensure the timeline of sp and tpls does not overlap
    sp, _ = filter_duplicates(sp.copy().reset_index(), tpls.reset_index())

    ## select valid user
    quality_path = os.path.join(".", "data", "quality", "gc_slide_filtered.csv")
    valid_users = pd.read_csv(quality_path)["user_id"].values

    sp = sp.loc[sp["user_id"].isin(valid_users)]

    # select only switzerland records
    swissBoundary = gpd.read_file(os.path.join(".", "data", "swiss", "swiss_1903+.shp"))
    print("Before spatial filtering: ", sp.shape[0])
    sp = filter_within_swiss(sp, swissBoundary)
    print("After spatial filtering: ", sp.shape[0])

    # filter activity staypoints
    drop_sp = sp.loc[sp["is_activity"] == True]

    # generate locations
    sp_loc, _ = drop_sp.as_staypoints.generate_locations(
        epsilon=20, num_samples=2, distance_metric="haversine", agg_level="dataset", n_jobs=-1
    )

    # filter noise staypoints
    sp_loc = sp_loc.loc[~sp_loc["location_id"].isna()].copy()
    print("After filter non-location staypoints: ", sp_loc.shape[0])

    # merge staypoints
    sp_merged = sp_loc.as_staypoints.merge_staypoints(
        triplegs=pd.DataFrame([]), max_time_gap="1min", agg={"location_id": "first", "geom": "first"}
    )
    print("After staypoints merging: ", sp_merged.shape[0])
    # recalculate staypoint duration
    sp_merged["duration"] = (sp_merged["finished_at"] - sp_merged["started_at"]).dt.total_seconds() // 60
    sp_merged = gpd.GeoDataFrame(sp_merged, geometry="geom")

    sp = _split_overlaps(sp_merged, granularity="day").sort_values(by=["user_id", "started_at"]).reset_index(drop=True)
    sp = sp[["user_id", "started_at", "location_id"]]
    sp["user_id"] = sp["user_id"].astype(int)
    sp["location_id"] = sp["location_id"].astype(int)
    sp["date"] = sp["started_at"].dt.date
    sp.drop(columns="started_at", inplace=True)

    # we delete the self transitions
    sp["loc_next"] = sp["location_id"].shift(-1)
    sp["date_next"] = sp["date"].shift(-1)
    sp = sp.loc[~((sp["loc_next"] == sp["location_id"]) & (sp["date_next"] == sp["date"]))].copy()
    sp.drop(columns=["loc_next", "date_next"], inplace=True)

    value_counts = sp.groupby(["user_id", "date"]).agg({"location_id": "nunique"}).value_counts()

    # the fraction of daily location visit <= 6 locations
    print(value_counts.head(6).sum() / value_counts.sum())

    # the mean daily location visit
    print(sp.groupby(["user_id", "date"]).agg({"location_id": "nunique"}).mean().values)

    valid_user_dates = sp.groupby(["user_id", "date"]).agg({"location_id": "nunique"})
    # we only select daily location visit < 7 records
    # valid_user_dates = valid_user_dates[valid_user_dates<7]
    valid_user_dates.rename(columns={"location_id": "daily_visit"}, inplace=True)

    sp_valid = sp.merge(valid_user_dates.reset_index(), on=["user_id", "date"], how="left")
    sp_valid = sp_valid.loc[~sp_valid["daily_visit"].isna()]

    # check daily location visit filter is correct
    print(sp_valid.groupby(["user_id", "date"]).agg({"location_id": "nunique"}).value_counts())
    sp_valid.to_csv("./data/sp_for_motifs.csv")


def get_motifs():
    sp = pd.read_csv("./data/sp_for_motifs.csv")

    graphs_ls = _motifs_classification(sp.copy())

    all_user_days = _label_classes(graphs_ls)
    sp_class = (
        sp.groupby(["user_id", "date"], as_index=False).size().merge(all_user_days, on=["user_id", "date"], how="left")
    )

    print("Dataset mean motifs proportion: {:.2f}".format(len(sp_class.loc[~sp_class["class"].isna()]) / len(sp_class)))

    motif_prop = sp_class.groupby("user_id").apply(_get_motif_prop_user).rename("motif_prop").reset_index()

    print("User mean motifs proportion: {:.2f}".format(motif_prop.mean()["motif_prop"]))

    return motif_prop


def plot_motifs(motif_prop):
    # plotting
    plt.plot()
    motif_prop.boxplot(column=["motif_prop"])
    plt.show()

    plt.figure(figsize=(8, 5))

    enc = OrdinalEncoder(dtype=np.int64)
    motif_prop["user_id"] = enc.fit_transform(motif_prop["user_id"].values.reshape(-1, 1)) + 1

    result = pd.read_csv(r".\outputs\gc_transformer_7_5\user_detail.csv")
    result["acc@1"] = result["correct@1"] / result["total"]
    result["acc@10"] = result["correct@10"] / result["total"]

    all_df = motif_prop.merge(result, left_on="user_id", right_on="user")
    all_df["acc@1"] = all_df["acc@1"] * 100

    plt.scatter(all_df["acc@1"], all_df["motif_prop"])

    pearsonr = stats.pearsonr(all_df["acc@1"], all_df["motif_prop"])[0]

    # Create sequence of 100 numbers from 0 to 100
    xseq = np.linspace(all_df["acc@1"].min() - 3, all_df["acc@1"].max() + 5, num=100)

    # Plot regression line
    b, a = np.polyfit(all_df["acc@1"], all_df["motif_prop"], deg=1)
    plt.plot(xseq, a + b * xseq, color="k", lw=1, label=f"pearson r: {pearsonr:.2f}")

    plt.ylabel("Motifs proportion in daily mobility", fontsize=16)
    plt.xlabel("Acc@1", fontsize=16)
    plt.legend(prop={"size": 13})

    # plt.savefig(r".\results\03_collective\motifs.pdf", dpi=600, bbox_inches="tight")
    plt.show()

    print("Correlation coefficient: {:.2f}".format(stats.pearsonr(all_df["acc@1"], all_df["motif_prop"])[0]))


def _motifs_classification(sp):
    graphs_ls = []
    for daily_visit in [1, 2, 3, 4, 5, 6]:
        curr_sp = sp.loc[sp["daily_visit"] == daily_visit].copy()
        curr_sp["next_loc"] = curr_sp["location_id"].shift(-1)

        if daily_visit == 1:
            curr_graph = curr_sp.groupby(["user_id", "date"]).size()
            graphs_ls.append(curr_graph.rename("size").reset_index())
            # daily_records.append(len(curr_graph))
            continue

        # the edge number shall be at least the node number
        curr_edge_num = curr_sp.groupby(["user_id", "date"]).size() - 1
        valid_user_days = curr_edge_num[curr_edge_num >= daily_visit].rename("edge_num")
        curr_sp = curr_sp.merge(valid_user_days.reset_index(), on=["user_id", "date"], how="left")
        curr_sp = curr_sp.loc[~curr_sp["edge_num"].isna()]

        if daily_visit == 2:
            curr_graph = curr_sp.groupby(["user_id", "date"]).size()
            graphs_ls.append(curr_graph.rename("size").reset_index())
            # daily_records.append(len(curr_graph))
            continue

        graph_df = curr_sp.groupby(["user_id", "date"]).apply(_construct_graph)
        # filter graphs that do not have an in-degree and out degree
        graph_df = graph_df.loc[~graph_df.isna()]
        graphs = graph_df.values

        motifs_groups = []
        for i in range(graphs.shape[0] - 1):
            # print(i)
            if i in [item for sublist in motifs_groups for item in sublist]:
                continue
            possible_match = [i]
            for j in range(i + 1, graphs.shape[0]):
                curr_graph = graphs[i]
                compare_graph = graphs[j]

                GM = isomorphism.GraphMatcher(curr_graph, compare_graph).is_isomorphic()
                if GM:
                    possible_match.append(j)
            motifs_groups.append(possible_match)
        # print(len(graphs))
        # print(len([item for sublist in motifs_groups for item in sublist]))

        graph_df = graph_df.rename("graphs").reset_index()
        class_arr = np.zeros(graph_df.shape[0]) - 1
        for i, classes in enumerate(motifs_groups):
            class_arr[classes] = i
        graph_df["class"] = class_arr

        graphs_ls.append(graph_df)

    return graphs_ls


def _construct_graph(df):
    G = nx.DiGraph()
    G.add_nodes_from(df["location_id"])

    G.add_edges_from(df.iloc[:-1][["location_id", "next_loc"]].astype(int).values)

    in_degree = np.all([False if degree == 0 else True for _, degree in G.in_degree])
    out_degree = np.all([False if degree == 0 else True for _, degree in G.out_degree])
    if in_degree and out_degree:
        return G


def _label_classes(graphs_ls):
    # we lable the classes

    total_motifs = np.sum([len(num) for num in graphs_ls])

    all_user_days = []
    for i, graph in enumerate(graphs_ls):
        if i == 0 or i == 1:
            res = graph.drop(columns={"size"})
            res["class"] = i * 10
            all_user_days.append(res)
        else:
            prop = graph["class"].value_counts() / total_motifs * 100
            valid_class = prop[prop > 0.5].index
            res = graph.loc[graph["class"].isin(valid_class)].copy()
            res = res.drop(columns={"graphs"})
            res["class"] = i * 10 + res["class"]
            all_user_days.append(res)

    return pd.concat(all_user_days)


def _get_motif_prop_user(df):
    return (len(df["class"]) - df["class"].isna().sum()) / len(df["class"])


if __name__ == "__main__":
    ## geolife and gc get entropy
    plot_entropy(if_gc=False)

    ## #staypoints, #locations, #tracking days
    get_stats(if_gc=False)

    ## tracking quality plots
    plot_tracking_quality(if_gc=False)

    ## motifs (working only for GC)
    # motifs_preprocessing()

    # motif_prop = get_motifs()
    # plot_motifs(motif_prop)
