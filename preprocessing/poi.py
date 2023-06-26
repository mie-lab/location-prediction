import os
import geopandas as gpd
import pandas as pd
import numpy as np
import pickle as pickle
from tqdm import tqdm

from gensim.corpora import Dictionary
from gensim.models import TfidfModel, LdaModel
import gensim

import trackintel as ti


def _read_poi_files():
    # get all the pofws, 1
    pofw_file = gpd.read_file(os.path.join("data", "poi", "ori", "gis_osm_pofw_free_1.shp"))

    # get all the transport, 4
    transport_file = gpd.read_file(os.path.join("data", "poi", "ori", "gis_osm_transport_free_1.shp"))

    # drop the trees and get the beaches, 1
    natural_file = gpd.read_file(os.path.join("data", "poi", "ori", "gis_osm_natural_free_1.shp"))
    natural_file = natural_file.loc[natural_file["fclass"] != "tree"]

    natural_a_file = gpd.read_file(os.path.join("data", "poi", "ori", "gis_osm_natural_a_free_1.shp"))
    natural_a_file = natural_a_file.loc[natural_a_file["fclass"].isin(["beach"])].reset_index(drop=True)
    natural_a_file["geometry"] = natural_a_file.to_crs("EPSG:2056").geometry.centroid.to_crs("EPSG:4326")

    # get all the pois, 11
    poi_file = gpd.read_file(os.path.join("data", "poi", "ori", "gis_osm_pois_free_1.shp"))

    # get the parking slots of traffic point file, 4
    traffic_file = gpd.read_file(os.path.join("data", "poi", "ori", "gis_osm_traffic_free_1.shp"))
    traffic_file = (
        traffic_file.loc[
            traffic_file["fclass"].isin(
                [
                    "parking",
                    "parking_bicycle",
                    "parking_underground",
                    "parking_multistorey",
                ]
            )
        ]
        .reset_index(drop=True)
        .copy()
    )

    # get the parking slots of traffic area file
    traffic_a_file = gpd.read_file(os.path.join("data", "poi", "ori", "gis_osm_traffic_a_free_1.shp"))
    traffic_a_file = (
        traffic_a_file.loc[
            traffic_a_file["fclass"].isin(
                [
                    "parking",
                    "parking_bicycle",
                    "parking_underground",
                    "parking_multistorey",
                ]
            )
        ]
        .reset_index(drop=True)
        .copy()
    )
    traffic_a_file["geometry"] = traffic_a_file.to_crs("EPSG:2056").geometry.centroid.to_crs("EPSG:4326")

    # buildings_file
    buildings_file = gpd.read_file(os.path.join("data", "poi", "ori", "gis_osm_buildings_a_free_1.shp"))
    buildings_file["code"] = buildings_file.groupby(["type"]).ngroup() + 1002
    buildings_file["geometry"] = buildings_file.to_crs("EPSG:2056").geometry.centroid.to_crs("EPSG:4326")

    # concat all the pois
    all_pois = pd.concat(
        [
            pofw_file,
            transport_file,
            natural_file,
            natural_a_file,
            poi_file,
            traffic_file,
            traffic_a_file,
            buildings_file,
        ]
    )
    # all_pois.drop(columns=["name"], inplace=True)

    return all_pois


def _assign_category(df):
    # 2018 Hong: Life services, Office building/space, Other facilities, Medical/Education, Entertainment, Government, Residence communities, Financial services
    # 2021 Yin : Residential, Hotel/resort, Mixed-use, K–12 schools, University/college, Office/workplace, Services, Civic/religious, Shopping/retail, Recreation/entertainment, Transportation, Others

    ### osm code -> 2018 Hong -> 2021 Yin
    # public 20xx  -> Residence communities ->  Residential
    #    university + school + kindergarten + college (208x) -> Medical/Education -> K–12 schools/University/college
    # health 21xx -> Medical/Education -> Services
    # leisure 22xx -> Entertainment -> Recreation/entertainment
    # catering 23xx -> Life services -> Residential
    # accommodation 24xx -> Entertainment -> Hotel/resort
    # shopping 25xx -> Life services -> Shopping/retail
    # money 26xx -> Financial services -> Services
    # tourism 27xx -> Entertainment -> Recreation/entertainment
    # pofw 3xxx -> Life services -> Civic/religious
    # natural 41xx -> Entertainment -> Recreation/entertainment
    # transport 56xx -> Other facilities -> Transportation
    # miscpoi 29xx -> Other facilities -> Others

    # note: miscpoi contains "bench" or "drinking_water" that might not reveal any landuse info

    # init
    df["category"] = "Unknown"

    # public 20xx  -> Residence communities ->  Residential
    #    university + school + kindergarten + college (208x) -> Medical/Education -> K–12 schools/University/college
    df.loc[(df["code"] > 2000) & (df["code"] < 2100), "category"] = "Residential"
    df.loc[(df["code"] > 2080) & (df["code"] < 2090), "category"] = "Schools"

    # health 21xx -> Medical/Education -> Services
    df.loc[(df["code"] > 2100) & (df["code"] < 2200), "category"] = "Services"

    # leisure 22xx -> Entertainment -> Recreation/entertainment
    df.loc[(df["code"] > 2200) & (df["code"] < 2300), "category"] = "Entertainment"

    # catering 23xx -> Life services -> Residential
    df.loc[(df["code"] > 2300) & (df["code"] < 2400), "category"] = "Residential"

    # accommodation 24xx -> Entertainment -> Hotel/resort
    df.loc[(df["code"] > 2400) & (df["code"] < 2500), "category"] = "Entertainment"

    # shopping 25xx -> Life services -> Shopping/retail
    df.loc[(df["code"] > 2500) & (df["code"] < 2600), "category"] = "Shopping"

    # money 26xx -> Financial services -> Services
    df.loc[(df["code"] > 2600) & (df["code"] < 2700), "category"] = "Services"

    # tourism 27xx -> Entertainment -> Recreation/entertainment
    df.loc[(df["code"] > 2700) & (df["code"] < 2800), "category"] = "Entertainment"

    # miscpoi 29xx -> Other facilities -> Others
    df.loc[(df["code"] > 2900) & (df["code"] < 3000), "category"] = "Others"

    # pofw 3xxx -> Life services -> Civic/religious
    df.loc[(df["code"] > 3000) & (df["code"] < 4000), "category"] = "Civic"

    # natural 41xx -> Entertainment -> Recreation/entertainment
    df.loc[(df["code"] > 4000) & (df["code"] < 5000), "category"] = "Entertainment"

    # transport 56xx -> Other facilities -> Transportation
    df.loc[(df["code"] > 5600) & (df["code"] < 5700), "category"] = "Transportation"

    # Unknown           2574292
    # Others            95253
    # Entertainment     77546
    # Shopping          40720
    # Residential       38335
    # Transportation    38164
    # Services           7728
    # Schools            2667
    # Civic               735

    # print(df["category"].value_counts())
    return df


def preprocess():
    gdf = _read_poi_files()

    # assign category for tf-idf calculation
    gdf = _assign_category(gdf)

    # final cleaning
    gdf.drop(columns=["osm_id", "fclass"], inplace=True)
    # reindex
    gdf.reset_index(drop=True, inplace=True)
    gdf.index.name = "id"
    gdf.reset_index(inplace=True)

    # change the projection and save
    gdf = gdf.to_crs("EPSG:2056")
    gdf.to_file(os.path.join("data", "poi", "final_pois.shp"))


def get_poi_representation(method="lda", categories=16):
    # checked: buffer method; transform to final_poi; vector values are different

    # read location and change the geometry columns
    locs = ti.read_locations_csv(os.path.join("data", f"locations_gc.csv"), index_col="id", crs="EPSG:4326")
    locs = gpd.GeoDataFrame(locs.drop(columns="center"), crs="EPSG:4326", geometry="extent").to_crs("EPSG:2056")

    # drop duplicate index
    locs.drop(columns="user_id", inplace=True)
    locs.reset_index(inplace=True)
    locs.rename(columns={"id": "loc_id"}, inplace=True)

    # read poi file
    poi = gpd.read_file(os.path.join("data", "poi", "final_pois.shp"))
    spatial_index = poi.sindex

    poi_dict_ls = []
    buffer_ls = np.arange(11) * 50
    # buffer_ls = [250]

    for distance in buffer_ls:
        curr_locs = locs.copy()
        ## create buffer for each location
        if distance != 0:
            curr_locs["extent"] = curr_locs["extent"].buffer(distance=distance)

        # get the inside poi within each location
        tqdm.pandas(desc="Generating poi within")
        curr_locs["poi_within"] = curr_locs["extent"].progress_apply(
            _get_inside_pois, poi=poi, spatial_index=spatial_index
        )

        # cleaning and expanding to location_id-poi_id pair
        curr_locs.drop(columns="extent", inplace=True)

        # explode preserves nan - preserves locs with no poi
        locs_poi = curr_locs.explode(column="poi_within")

        # get the poi info from original poi df
        locs_poi = locs_poi.merge(poi[["id", "category", "code"]], left_on="poi_within", right_on="id", how="left")
        locs_poi.drop(columns=["id"], inplace=True)

        # final cleaning
        valid_pairs = locs_poi.dropna(subset=["poi_within"]).copy()
        valid_pairs["code"] = valid_pairs["code"].astype(int).astype(str)

        # get the poi representation
        if method == "lda":
            poi_dict = _lda(valid_pairs, categories=categories)
        elif method == "tf_idf":
            poi_dict = _tf_idf(valid_pairs, categories=categories)
        else:
            raise AttributeError

        poi_dict_ls.append(poi_dict)

    ## create the matrix
    idx_max = np.max([poi_dict["index"].max() for poi_dict in poi_dict_ls])
    all_idx = np.arange(idx_max + 1)

    # num_loc*lda_vector*buffer_num
    all_poi = np.zeros([len(all_idx), categories, len(buffer_ls)])

    for i, poi_dict in enumerate(poi_dict_ls):
        for j, idx in enumerate(poi_dict["index"]):
            # print(i, idx)
            all_poi[idx, :, i] = poi_dict["poiValues"][j, :]

    all_poi_dict = {"index": all_idx, "poiValues": all_poi}

    ## save to disk
    with open(os.path.join("data", f"poiValues_{method}_{categories}.pk"), "wb") as handle:
        pickle.dump(all_poi_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _get_inside_pois(df, poi, spatial_index):
    """
    Given one extent (df), return the poi within this extent.
    spatial_index is obtained from poi.sindex to speed up the process.
    """
    possible_matches_index = list(spatial_index.intersection(df.bounds))
    possible_matches = poi.iloc[possible_matches_index]
    precise_matches = possible_matches[possible_matches.within(df)]["id"].values

    return precise_matches


def _tf_idf(df, categories=8):
    """Note: deal with the manually assigned "category" field."""
    texts = df.groupby("loc_id")["category"].apply(list).to_list()

    dct = Dictionary(texts)
    corpus = [dct.doc2bow(line) for line in texts]

    tfmodel = TfidfModel(corpus)
    vector = tfmodel[corpus]

    # the tf array
    dense_tfvector = gensim.matutils.corpus2dense(vector, num_terms=categories).T
    # the index arr
    index_arr = df.groupby("loc_id").count().reset_index()["loc_id"].values
    return {"index": index_arr, "poiValues": dense_tfvector}


def _lda(df, categories=16):
    """Note: deal with the osm assigned "code" field."""
    texts = df.groupby("loc_id")["code"].apply(list).to_list()

    dct = Dictionary(texts)
    corpus = [dct.doc2bow(line) for line in texts]

    lda = LdaModel(corpus, num_topics=categories)
    vector = lda[corpus]

    # the lda array
    dense_ldavector = gensim.matutils.corpus2dense(vector, num_terms=categories).T
    # the index arr
    index_arr = df.groupby("loc_id", as_index=False).count()["loc_id"].values
    return {"index": index_arr, "poiValues": dense_ldavector}


if __name__ == "__main__":
    preprocess()

    get_poi_representation(method="lda", categories=16)
