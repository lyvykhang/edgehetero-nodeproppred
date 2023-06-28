from pathlib import Path
import os
import yaml
import utils

import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import combinations
from collections import Counter
import random
import torch
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset


def generate_edges(data, kind, seed, threshold=None):
    edge_list = []
    random.seed(seed)

    for ids in tqdm(data, desc=f"Generating '{kind}' Edges"):
        if len(ids) > 1:
            if threshold is not None:
                if len(ids) > threshold:
                    ids = random.sample(ids, threshold)
            edge_list.extend([edge for edge in combinations(sorted(ids), 2)])

    counts = Counter(edge_list)
    edge_list = {kind : list(counts.keys())}
    edge_stores = {kind : list(counts.values())}

    return edge_list, edge_stores


def preprocess(df, col, attr):
    temp_df = df.copy().dropna(subset=[col]).reset_index()[["index", col]]

    if col == "venue":
        temp_df[col] = pd.Series([x[attr] if attr in x.keys() else None for x in temp_df[col]], dtype=object)
    else:
        temp_df[col] = [[x[attr] for x in xs] for xs in temp_df[col]]

    grouped = temp_df.drop_duplicates(subset=["index"]).explode(col).groupby(col)["index"].apply(lambda x: x.tolist())
    return grouped


if __name__ == "__main__":
    project_root: Path = utils.get_project_root()
    os.chdir(project_root)
    with open(str(project_root / "config/data_generation_config.yaml")) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    
    dataset = PygNodePropPredDataset("ogbn-arxiv", root="data/")
    data = dataset[0]
    splits = dataset.get_idx_split()
    data = data.to_heterogeneous(node_type_names=["paper"], 
                                 edge_type_names=[("paper", "references", "paper")])
    data["paper"].train_idx = splits["train"]
    data["paper"].val_idx = splits["valid"]
    data["paper"].test_idx = splits["test"]

    path_to_metadata = str(Path(params["data"]["path_to_metadata"]))
    df = pd.read_parquet(path_to_metadata)

    authorship_input = preprocess(df, "authors", "id")
    authorship_output = generate_edges(authorship_input, kind="authorship", seed=params["seed"])

    fos_input = preprocess(df, "fos", "name")
    fos_output = generate_edges(fos_input, kind="fos", seed=params["seed"], 
                                threshold=int(np.mean(fos_input.apply(len))))

    venue_input = preprocess(df, "venue", "id")
    venue_output = generate_edges(venue_input, kind="venue", seed=params["seed"], 
                                  threshold=int(np.mean(venue_input.apply(len))))

    edge_lists = [authorship_output[0], fos_output[0], venue_output[0]]
    edge_stores = [authorship_output[1], fos_output[1], venue_output[1]]
    num_nodes = data["paper"].y.shape[0]

    edge_lists = {k : torch.tensor(v, dtype=torch.long) for d in edge_lists for k, v in d.items()}
    edge_stores = {k : torch.tensor(v, dtype=torch.float32) for d in edge_stores for k, v in d.items()}
    edge_lists_stores = {k : [v1.T, v2] for k, v1, v2 in utils.zip_dicts(edge_lists, edge_stores)}

    data[("paper", "shares_author_with", "paper")].edge_index = edge_lists_stores["authorship"][0].contiguous()
    data[("paper", "shares_author_with", "paper")].edge_stores = edge_lists_stores["authorship"][1].sigmoid()

    data[("paper", "shares_fos_with", "paper")].edge_index = edge_lists_stores["fos"][0].contiguous()
    data[("paper", "shares_fos_with", "paper")].edge_stores = edge_lists_stores["fos"][1].sigmoid()

    data[("paper", "shares_venue_with", "paper")].edge_index = edge_lists_stores["venue"][0].contiguous()
    data[("paper", "shares_venue_with", "paper")].edge_stores = edge_lists_stores["venue"][1]

    data[("paper", "references", "paper")].edge_stores = torch.ones(data[("paper", "references", "paper")].edge_index.shape[1], dtype=float)

    transform = T.Compose([T.ToUndirected(reduce="mean"), T.ToSparseTensor(attr="edge_stores", remove_edge_index=False)])
    data = transform(data)

    save_path = str(Path(params["data"]["save_path"]))
    torch.save(data, save_path)
    









