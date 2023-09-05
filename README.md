# Context-aware next location prediction

This repository represents the implementation of the paper:

## [Context-aware multi-head self-attentional neural network model for next location prediction](https://arxiv.org/abs/2212.01953)
[Ye Hong](https://hongyeehh.github.io/), [Yatao Zhang](https://frs.ethz.ch/people/researchers/yatao-zhang.html), [Konrad Schindler](https://prs.igp.ethz.ch/group/people/person-detail.schindler.html), [Martin Raubal](https://raubal.ethz.ch/)\
| [MIE, ETH Zurich](https://gis.ethz.ch/en/) | [FRS, Singapore-​ETH Centre](https://frs.ethz.ch/) | [PRS, ETH Zurich](https://prs.igp.ethz.ch/) |

![flowchart](fig/1_overview_flowchart.png?raw=True)

## Requirements and dependencies
This code has been tested on

- Python 3.9.12, Geopandas 0.12.1, trackintel 1.1.10, gensim 4.1.2, PyTorch 1.12.1, transformers 4.16.2, cudatoolkit 11.3, GeForce RTX 3090

To create a virtual environment and install the required dependencies, please run the following:
```shell
    git clone https://github.com/mie-lab/location-prediction.git
    cd location-prediction
    conda env create -f environment.yml
    conda activate loc-pred
```
in your working folder.

## Folder structure
The respective code files are stored in separate modules:
- `/preprocessing/*`. Functions that are used for preprocessing the dataset. It should be executed before training a model. `poi.py` includes POI preprocessing and embedding methods (**LDA** and **TF-IDF**).
- `/models/*`. Implementation of **Transformer** learning model.  
- `/baselines/*`. (Non-ML) Baseline methods that we implemented to compare with the proposed model. The methods include **persistent forecast**, **most frequent forecast** and **Markov models**. 
- `/config/*`. Hyperparameter settings are saved in the `.yml` files under the respective dataset folder under `config/`. For example, `/config/geolife/transformer.yml` contains hyperparameter settings of the transformer model for the geolife dataset. 
- `/utils/*`. Helper functions that are used for model training. 
- `/analysis/*`. Analysis function for getting dataset properties and visualizing training results of the model. `entropy.py` includes functions to calculate the **random**, **uncorrelated** and **real entropy**. `stats.py` includes functions to calculate the mobility **motifs**.

The main starting point for training a model is as follows:
- `main.py` for starting the deep learning model training. 
- `main_individual.py` for starting the training of individual models. 

## Model variations

The repo contains different model variations, which can be controlled as follows:

- Individual vs collective model - Running `main.py` or `main_individual.py`. Config files for individual models contain `ind_` as the prefix.
- Including different contexts - Whether to include a specific context can be controlled in the config files, with `if_embed_user`, `if_embed_poi`, `if_embed_time`, and `if_embed_duration` parameters.
- Including different previous days - The length of considered historical previous days can be controlled through the `previous_day` parameter in each config file.
- Including separate previous days - The selection of single historical previous days can be controlled through the `day_selection` parameter in each config file. `default` includes all days, and specific day selection can be passed in a list, e.g., `[0, 1, 7]` to include only the current, previous and one week before.

## Reproducing models on the Geolife dataset
To run the whole pipeline on the Geolife dataset, follow the steps below:

### 1. Install dependencies 
Download the repo, and install the necessary `Requirements and dependencies`.

### 2. Download Geolife 
Download the Geolife GPS tracking dataset from [here](https://www.microsoft.com/en-us/download/details.aspx?id=52367). Create a new folder in the repo root and name it `data`. Unzip and copy the Geolife `Data` folder into `data/`. The file structure should look like `data/Data/000/...`.

Create a file `paths.json` in the repo root, and define your working directories by writing:

```json
{
    "raw_geolife": "./data/Data"
}
```

### 3. Preprocess the dataset
run 
```shell
    python preprocessing/geolife.py 20
```
for executing the preprocessing script for the geolife dataset. The process takes 15-30min. `dataSet_geolife.csv`, `sp_time_temp_geolife.csv` and `valid_ids_geolife.pk` will be created under the `data/` folder, `geolife_slide_filtered.csv` will be created under `data/quality` folder.

### 4. Run the proposed transformer model
run 
```shell
    python main.py config/geolife/transformer.yml
```
for starting the training process. The dataloader will create intermediate data files and save them under `data/temp/` folder. The configuration of the current run, the network paramters and the performance indicators will be stored under the `outputs/` folder.

### 5. Get dataset statistics
run 
```shell
    python analysis/stats.py
```
for generating the mobility entropy plot, the basic statistics of the Geolife dataset, and generating the tracking quality plot. 

## Reproducing models on the check-in dataset
To run the whole pipeline on Gowalla or Foursquare New York City (NYC) datasets, follow the steps below:

### 1. Switch branch and install dependencies 
Switch to `lbsn` branch. Download the repo, and install the necessary `Requirements and dependencies`.

### 2. Download the datasets 
Download the Gowalla dataset from [here](https://snap.stanford.edu/data/loc-gowalla.html) or the Foursquare NYC dataset from [here](https://sites.google.com/site/yangdingqi/home/foursquare-dataset). Create a new folder in the repo root and name it `data`.  Unzip and copy the Gowalla `Gowalla_totalCheckins.txt` file into a new folder `data/gowalla`. The file structure should look like `data/gowalla/Gowalla_totalCheckins.txt` for Gowalla. Alternatively, unzip and copy the Foursquare `dataset_TSMC2014_NYC.txt` file into a new folder `data/tsmc2014`. The file structure should look like `data/tsmc2014/dataset_TSMC2014_NYC.txt` for Foursquare.


Create a file `paths.json` in the repo root, and define your working directories by writing:
```json
{
    "raw_gowalla": "./data/gowalla"
}
```
or
```json
{
    "raw_foursquare": "./data/tsmc2014"
}
```

### 3. Preprocess the dataset

run 
```shell
    python preprocessing/gowalla.py
```
or 
```shell
    python preprocessing/foursquare.py
```
for executing the preprocessing script for the datasets. `dataSet_*.csv`, `locations_*.csv`, `sp_time_temp_*.csv` and `valid_ids_*.pk` will be created under the `data/` folder, 

### 4. Run the proposed transformer model
- run 
```shell
    python main.py config/gowalla/transformer.yml
```
or
```shell
    python main.py config/foursquare/transformer.yml
```

for starting the training process. The dataloader will create intermediate data files and save them under the `data/temp/` folder. The configuration of the current run, the network paramters and the performance indicators will be stored under the `outputs/` folder.

## Citation
If you find this code useful for your work or use it in your project, please consider citing:

```shell

@article{hong_context_2023,
  title   = {Context-aware multi-head self-attentional neural network model for next location prediction},
  journal = {Transportation Research Part C: Emerging Technologies},
  author  = {Hong, Ye and Zhang, Yatao and Schindler, Konrad and Raubal, Martin},
  year    = {2023},
}

```

## Contact
If you have any questions, please open an issue or let me know: 
- Ye Hong {hongy@ethz.ch}
