from pathlib import Path
import os
import yaml
import csv

import numpy as np
import pandas as pd
import random
from itertools import combinations
import torch
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from ogb.nodeproppred import PygNodePropPredDataset
from tqdm import tqdm

import utils


def preprocess(df, col, id, attr=None):
    """Outputs all unique entities (e.g. author IDs) and a list of 
    their associated papers (e.g. papers that author (co-)authored).

    Args:
        df (pd.DataFrame): DF containing the metadata.
        col (str): name of the column for the entity of interest.
        id (str): name of the index column, "index" (arxiv)/"pmid" (pubmed).
        attr (str): only used for ogbn-arxiv - name of nested attribute.

    Output:
        grouped (pd.Series): as described above.
    """
    temp_df = df.copy().dropna(subset=[col]).reset_index()[[id, col]]

    if attr is not None:
        if col == "venue":
            temp_df[col] = pd.Series([x[attr] if attr in x.keys() else None for x in temp_df[col]], dtype=object)
        else:
            temp_df[col] = [[x[attr] for x in xs] for xs in temp_df[col]]

    grouped = temp_df.drop_duplicates(subset=[id]).explode(col).groupby(col)[id].apply(lambda x: x.tolist())
    return grouped


def generate_edges(data, kind, seed, threshold=None):
    """For each unique attribute entity, e.g. author/journal/etc., 
    generates edges via pairwise combinations of their associated paper IDs.

    Args:
        data (pd.Series): output of `preprocess()`.
        kind (str): "references", "authorship", "fos", "mesh", "venue", "journal".
        seed (int): random seed (only used if `threshold` is specified).
        threshold (int): sample `threshold` paper IDs and only create pairwise combinations 
            between at most that many papers per unique entity. 
    
    Output:
        edge_list (dict): dict value = list of unique edges.
    """
    edge_list = set()
    random.seed(seed)

    for ids in tqdm(data, desc=f"Generating '{kind}' Edges"):
        if len(ids) > 1:
            if threshold is not None:
                if len(ids) > threshold:
                    ids = random.sample(ids, threshold)
            edge_list.update([edge for edge in combinations(sorted(ids), 2)])

    edge_list = {kind : list(edge_list)}

    return edge_list


def multigraph_dataset(df, edge_lists, device, data=None, node_map=None):
    """Instantiates a HeteroData object (PubMed) or updates an existing one (OGBN-arXiv) based on given edge lists.

    Args:
        df (pd.DataFrame): Dataframe representation of chosen dataset.
        edge_lists (list): list of remapped edge dicts {edge type : edge list}.
        device (torch.device): device for storing data.
        data (torch_geometric.data.HeteroData): existing HeteroData object (for OGBN-arXiv). Leave as None for PubMed.
        node_map (dict): mapping of node EIDs to new index value (only needed for PubMed).

    Output:
        data (torch_geometric.data.HeteroData): data object describing a heterogeneous graph, holding multiple edge types in disjunct storage objects.
    """
    edge_lists = {k : torch.tensor(v, dtype=torch.long).T.contiguous() for d in edge_lists for k, v in d.items()}

    if data is not None: # Data object already exists.
        data[("paper", "shares_author_with", "paper")].edge_index = edge_lists["authorship"]
        data[("paper", "shares_fos_with", "paper")].edge_index = edge_lists["fos"]
        data[("paper", "shares_venue_with", "paper")].edge_index = edge_lists["venue"]
    else: # instantiate new Data object for PubMed.
        assert node_map is not None, "Must specify `node_map` for PubMed."
        data = HeteroData(
            paper={
                "y" : torch.tensor(df["label"].values, dtype=torch.long, device=device).unsqueeze(0).T, 
                "num_nodes" : len(node_map)
            },
            paper__references__paper={"edge_index" : edge_lists["references"]},
            paper__shares_author_with__paper={"edge_index" : edge_lists["authorship"]},
            paper__shares_journal_with__paper={"edge_index" : edge_lists["srcid"]},
            paper__shares_mesh_with__paper={"edge_index" : edge_lists["mesh"]}
        ).to(device)

    transform = T.Compose([T.ToUndirected(), T.ToSparseTensor(remove_edge_index=True)])
    data = transform(data)

    return data


def main_ogbnarxiv():
    dataset = PygNodePropPredDataset("ogbn-arxiv", root="data/")
    data = dataset[0]
    splits = dataset.get_idx_split()
    data = data.to_heterogeneous(node_type_names=["paper"], 
                                 edge_type_names=[("paper", "references", "paper")])
    data["paper"].train_idx = splits["train"]
    data["paper"].val_idx = splits["valid"]
    data["paper"].test_idx = splits["test"]

    authorship_input = preprocess(df, "authors", "index", "id")
    authorship_output = generate_edges(authorship_input, "authorship", params["seed"])

    fos_input = preprocess(df, "fos", "index", "name")
    fos_output = generate_edges(fos_input, "fos", params["seed"], int(np.mean(fos_input.apply(len))))

    venue_input = preprocess(df, "venue", "index", "id")
    venue_output = generate_edges(venue_input, "venue", params["seed"], int(np.mean(venue_input.apply(len))))

    edge_lists = [authorship_output, fos_output, venue_output]

    data = multigraph_dataset(df, edge_lists, torch.device("cpu"), data)

    save_path = str(Path(params["data"]["save_path"]["ogbnarxiv"]))
    torch.save(data, save_path)


def main_pubmed():
    records = []
    with open(params["data"]["path_to_metadata"]["pubmed_refs"]) as tsv: # Citation edge list stored in separate TSV. 
        for line in csv.reader(tsv, delimiter="\t"):
            records.append(line)
    refs_edgelist = [(int(r[1].split(":")[1]), int(r[-1].split(":")[1])) for r in records[2:]]
    refs_edgelist = [(e1, e2) for (e1, e2) in refs_edgelist if e1 != 17874530 and e2 != 17874530] # 17874530 is the one PMID with no available metadata; manually remove.
    refs_output = {"references" : refs_edgelist}

    authorship_input = preprocess(df, "authors", "pmid")
    authorship_output = generate_edges(authorship_input, "authorship", params["seed"])

    srcid_input = preprocess(df, "journal", "pmid")
    srcid_output = generate_edges(srcid_input, "srcid", params["seed"])

    mesh_input = preprocess(df, "mesh", "pmid")
    mesh_output = generate_edges(mesh_input, "mesh", params["seed"], int(np.mean(mesh_input.apply(len))))

    def remapper(internal_eids, edge_lists):
        """Remaps node names (and subsequently edge lists) into a sequential index (0 ... no. nodes). 

        Args:
            internal_eids (list): list of all unique supervised EIDs.
            edge_lists (list): list of edge dicts {edge type : edge list} for all relationship types. 
        
        Output:
            node_map (dict): mapping of node EIDs (across all relationship types) to new index value.
            edge_lists_remapped (list): list of remapped edge dicts. 
        """
        node_map = dict(zip(range(len(internal_eids)), internal_eids))
        inv_node_map = {v : k for k, v in node_map.items()}
        edge_lists_remapped = [{k : [(inv_node_map[eid_1], inv_node_map[eid_2]) 
            for (eid_1, eid_2) in tqdm(v)]} for edge_list in edge_lists for k, v in edge_list.items()]
        
        return node_map, edge_lists_remapped

    node_map, edge_lists = remapper(df["pmid"].to_list(), [refs_output, authorship_output, mesh_output, srcid_output])

    data = multigraph_dataset(df, edge_lists, torch.device("cpu"), node_map=node_map)
    data["paper"].x = torch.tensor(np.stack(df["tfidf"])).float() # attach the default TFIDF features.

    if params["pubmed_fixed_split"]:
        splits = utils.per_class_idx_split(data, params["seed"])
        data["paper"].train_idx, data["paper"].val_idx, data["paper"].test_idx = splits

    save_path = str(Path(params["data"]["save_path"]["pubmed"]))
    torch.save(data, save_path)


if __name__ == "__main__":
    project_root: Path = utils.get_project_root()
    os.chdir(project_root)
    with open(str(project_root / "config/data_generation_config.yaml")) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    
    path_to_metadata = str(Path(params["data"]["path_to_metadata"][params["dataset"]]))
    df = pd.read_parquet(path_to_metadata)

    if params["dataset"] == "ogbnarxiv":
        main_ogbnarxiv()
    elif params["dataset"] == "pubmed":
        main_pubmed()