import numpy as np

from tqdm import tqdm
import pandas as pd
from joblib import Parallel, delayed


def random_entropy(stps, print_progress=False):
    """Random entropy of individual visited locations.

    Parameters
    ----------
    stps : Geodataframe (as trackintel staypoints)
        Staypoints with column "location_id".

    print_progress: boolen, default False
        Show per-user progress if set to True.

    Returns
    -------
    s: pd.Series
        the random entropy calculated at the individual level.

    References
    ----------
    [1] Song, C., Qu, Z., Blumm, N. and Barabási, A.L., 2010. Limits of predictability in human mobility. Science, 327(5968), pp.1018-1021.

    """
    if print_progress:
        tqdm.pandas(desc="User random entropy calculation")
        s = stps.groupby("user_id").progress_apply(lambda x: _random_entropy_user(x))
    else:
        s = stps.groupby("user_id").apply(lambda x: _random_entropy_user(x))

    s.rename("randomEntropy", inplace=True)
    return s


def uncorrelated_entropy(stps, print_progress=False):
    """
    Uncorrelated entropy of individual visited locations.

    Parameters
    ----------
    stps : Geodataframe (as trackintel staypoints)
        Staypoints with column "location_id".

    print_progress: boolen, default False
        Show per-user progress if set to True.

    Returns
    -------
    pandas DataFrame
        the temporal-uncorrelated entropy of the individuals.

    References
    ----------
    [1] Song, C., Qu, Z., Blumm, N. and Barabási, A.L., 2010. Limits of predictability in human mobility. Science, 327(5968), pp.1018-1021.

    """
    if print_progress:
        tqdm.pandas(desc="User uncorrelated entropy calculation")
        s = stps.groupby("user_id").progress_apply(lambda x: _uncorrelated_entropy_user(x))
    else:
        s = stps.groupby("user_id").apply(lambda x: _uncorrelated_entropy_user(x))

    s.rename("uncorrelatedEntropy", inplace=True)
    return s


def real_entropy(stps, print_progress=False, n_jobs=-1):
    """
    Real entropy of individual visited locations.

    Parameters
    ----------
    stps : Geodataframe (as trackintel staypoints)
        Staypoints with column "location_id".

    print_progress: boolen, default False
        Show per-user progress if set to True.

    Returns
    -------
    pandas DataFrame
        the real entropy of the individuals.

    References
    ----------
    [1] Song, C., Qu, Z., Blumm, N. and Barabási, A.L., 2010. Limits of predictability in human mobility. Science, 327(5968), pp.1018-1021.

    """
    # if print_progress:
    #     tqdm.pandas(desc="User uncorrelated entropy calculation")

    #     s = stps.groupby("user_id").progress_apply(lambda x: _real_entropy_user(x))
    # else:
    #     s = stps.groupby("user_id").apply(lambda x: _real_entropy_user(x))

    s = applyParallel(stps.groupby("user_id"), _real_entropy_user, print_progress=print_progress, n_jobs=n_jobs)
    # s.rename("realEntropy", inplace=True)
    return s


def _random_entropy_user(stps_user):
    """
    User level random entropy calculation, see random_entropy() for details.

    Parameters
    ----------
    stps_user : Geodataframe (as trackintel staypoints)
        The staypoints from an individual, should contain column "location_id".

    Returns
    -------
    float
        the random entropy of the individual
    """
    locs_numb = len(stps_user["location_id"].unique())
    return np.log2(locs_numb)


def _uncorrelated_entropy_user(stps_user):
    """
    User level uncorrelated entropy calculation, see uncorrelated_entropy() for details.

    Parameters
    ----------
    stps_user : Geodataframe (as trackintel staypoints)
        The staypoints from an individual, should contain column "location_id".

    Returns
    -------
    float
        the temporal-uncorrelated entropy of the individual
    """
    locs_prob = stps_user["location_id"].value_counts(normalize=True, sort=False).values
    return -(locs_prob * np.log2(locs_prob)).sum()


def _real_entropy_user(stps_user):
    """
    User level real entropy calculation, see real_entropy() for details.

    Parameters
    ----------
    stps_user : Geodataframe (as trackintel staypoints)
        The staypoints from an individual, should contain column "location_id".

    Returns
    -------
    float
        the real entropy of the individual
    """
    locs_series = stps_user["location_id"].values

    n = len(locs_series)

    # 1 to ensure to consider the first situation from where
    # locs_series[i:j] = [] and locs_series[i:j] = locs_series[0:1]
    # TODO: sckitmobility also add 2 here to account for the "last elements", I am not sure why
    sum_lambda = 1

    for i in range(1, n - 1):
        j = i + 1

        while True:
            # if the locs_series[i:j] is longer than locs_series[:i],
            # we can no longer find it locs_series[i:j] in locs_series[:i]
            if j - i > i:
                break

            # if locs_series[i:j] exist in locs_series[:i], we increase j by 1
            # sliding_window_view creates sublist of length len(locs_series[i:j]) from locs_series[:i]
            ls = np.lib.stride_tricks.sliding_window_view(locs_series[:i], j - i).tolist()
            if tuple(locs_series[i:j]) in list(map(tuple, ls)):
                # if the subsequence already exist, we increase the sequence by 1, and check again
                j += 1
            else:
                # we find the "shortest substring" that does not exist in locs_series[:i]
                break

        # length of the substring
        sum_lambda += j - i

    # TODO: this is the function S5 from the suppl. material, sckitmobility uses np.log2
    return 1.0 / (sum_lambda * 1 / n) * np.log(n)


def applyParallel(dfGrouped, func, n_jobs, print_progress, **kwargs):

    df_ls = Parallel(n_jobs=n_jobs)(
        delayed(func)(group, **kwargs) for _, group in tqdm(dfGrouped, disable=not print_progress)
    )
    return df_ls


if __name__ == "__main__":
    import geopandas as gpd
    from shapely import wkt

    # import skmob
    import matplotlib
    import matplotlib.pyplot as plt
    import scipy.stats as stats

    all_df = pd.read_csv(r"D:\Code\HAI-IRMA\WP1\npp\data\input\sp.csv")

    all_df["geom"] = all_df["geom"].apply(wkt.loads)
    all_gdf = gpd.GeoDataFrame(all_df, geometry="geom", crs="EPSG:4326")

    # print(all_df)
    all_gdf.sort_values(by=["user_id", "started_at"], inplace=True)

    all_gdf["location_id"] = all_gdf["location_id"].astype(int)

    # rea_entropy = real_entropy(all_gdf[all_gdf.user_id.isin(all_gdf.user_id.unique()[:5])], print_progress=True)
    rea_entropy = real_entropy(all_gdf, print_progress=True)
    print(rea_entropy)

    # from skmob.measures.individual import real_entropy

    # locs_df = pd.read_csv(r"D:\Code\HAI-IRMA\WP1\npp\data\input\locs.csv")
    # locs_df = locs_df[["id", "user_id", "center"]]
    # locs_df["center"] = locs_df["center"].apply(wkt.loads)
    # locs_df = gpd.GeoDataFrame(locs_df, geometry="center", crs="EPSG:4326")

    # locs_df["id"] = locs_df["id"].astype(int)

    # locs_df.drop_duplicates(subset="id", inplace=True)
    # locs_df.drop(columns={"user_id"}, inplace=True)

    # all_df = all_df[["user_id", "started_at", "location_id"]]
    # merged_df = all_df.merge(locs_df, left_on="location_id", right_on="id", validate="many_to_one").drop(columns={"id"})
    # merged_df = gpd.GeoDataFrame(merged_df, geometry="center", crs="EPSG:4326")
    # merged_df["lon"] = merged_df["center"].y
    # merged_df["lat"] = merged_df["center"].x

    # # all_df["long"] =
    # from skmob_entropy import sk_real_entropy

    # tdf = skmob.TrajDataFrame(merged_df, latitude="lat", longitude="lon", datetime="started_at", user_id="user_id")
    # tdf.sort_values(by=["uid", "datetime"], inplace=True)
    # # print(tdf)
    # tdf["location_id"] = tdf["location_id"].astype(int)

    # rea_entropy = sk_real_entropy(tdf[tdf.uid.isin(tdf.uid.unique()[:5])])
    # print(rea_entropy["sk_real_entropy"].values)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    matplotlib.rcParams["figure.dpi"] = 300
    matplotlib.rcParams["xtick.labelsize"] = 13
    matplotlib.rcParams["ytick.labelsize"] = 13
    np.set_printoptions(precision=4)
    np.set_printoptions(suppress=True)

    # distribution pattern
    realDensity = stats.gaussian_kde(rea_entropy)
    x = np.linspace(0, np.max(rea_entropy) + 0.5, 50)
    plt.plot(x, realDensity(x), label="$S$")
    plt.fill_between(x, 0, realDensity(x), facecolor="blue", alpha=0.2)

    plt.legend(prop={"size": 13})
    plt.xlabel("Entropy", fontsize=16)
    plt.ylabel("PDF", fontsize=16)

    plt.show()
