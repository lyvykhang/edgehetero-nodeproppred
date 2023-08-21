from pathlib import Path
import os
import yaml
import utils

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
import csv


def remapper(internal_eids, edge_lists):
    """Remaps node names (and subsequently edge lists) into a sequential index (0 ... no. nodes). 

    Args:
        internal_eids (set): set of all supervised EIDs.
        edge_lists (list): list of edge dicts {edge type : edge list} for all relationship types. 
    
    Output:
        node_map (dict): mapping of node EIDs (across all relationship types) to new index value.
        inv_node_map (dict): inverse of the above mapping.
        edge_lists_remap (list): list of remapped edge dicts. 
    """
    node_map = dict(zip(range(len(internal_eids)), internal_eids))
    inv_node_map = {v : k for k, v in node_map.items()}
    edge_lists_remap = [{k : [(inv_node_map[eid_1], inv_node_map[eid_2]) 
        for (eid_1, eid_2) in tqdm(v)]} for edge_list in edge_lists for k, v in edge_list.items()]
    
    return node_map, inv_node_map, edge_lists_remap


def generate_data(df, node_map, inv_node_map, edge_lists_remap, edge_stores, device):
    """Computes the relevant tensors and generates the HeteroData object. 

    Args:
        df (pd.DataFrame): Dataframe representation of chosen dataset.
        node_map (dict): mapping of node EIDs to new index value.
        inv_node_map (dict): inverse of the above mapping.
        edge_lists_remap (list): list of remapped edge dicts {edge type : edge list}.
        edge_stores (list): list of edge weight dicts {edge type : edge weights}.
        device (torch.device): device for storing data.

    Output:
        data (torch_geometric.data.HeteroData): data object describing a heterogeneous graph, holding multiple edge types in disjunct storage objects.
    """
    edge_lists = {k : torch.tensor(v, dtype=torch.long) for d in edge_lists_remap for k, v in d.items()}
    edge_stores = {k : torch.tensor(v, dtype=torch.float32) for d in edge_stores for k, v in d.items()}
    edge_lists_stores = {k : [v1.T, v2] for k, v1, v2 in utils.zip_dicts(edge_lists, edge_stores)}

    y = list(df.set_index("pmid").loc[list(inv_node_map.keys())[:len(df)]].reset_index()["label"].values)
    y_tensor = torch.tensor(y, dtype=torch.long, device=device)
    y_tensor = torch.unsqueeze(y_tensor, 0).T
    
    data = HeteroData(paper={"y" : y_tensor, 
                    "num_nodes" : len(node_map)},
        paper__references__paper={"edge_index" : edge_lists_stores["references"][0].contiguous(),
                    "edge_stores" : edge_lists_stores["references"][1]},
        paper__shares_author_with__paper={"edge_index" : edge_lists_stores["authorship"][0].contiguous(),
                    "edge_stores" : edge_lists_stores["authorship"][1].sigmoid()},
        paper__shares_journal_with__paper={"edge_index" : edge_lists_stores["srcid"][0].contiguous(),
                    "edge_stores" : edge_lists_stores["srcid"][1]},
        paper__shares_mesh_with__paper={"edge_index" : edge_lists_stores["mesh"][0].contiguous(),
                    "edge_stores" : edge_lists_stores["mesh"][1].sigmoid()}).to(device)

    transform = T.Compose([T.ToUndirected(reduce="mean"), T.ToSparseTensor(attr="edge_stores", remove_edge_index=False)])
    data = transform(data)

    return data


if __name__ == "__main__":
    project_root: Path = utils.get_project_root()
    os.chdir(project_root)
    with open(str(project_root / "config/data_generation_config.yaml")) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    path_to_metadata = str(Path(params["data"]["path_to_metadata"]["pubmed"]))
    df = pd.read_parquet(path_to_metadata)

    records = []
    with open(params["data"]["path_to_metadata"]["pubmed_refs"]) as tsv: # Citation edge list stored in separate TSV. 
        for line in csv.reader(tsv, delimiter="\t"):
            records.append(line)
    refs_edgelist = [(int(r[1].split(":")[1]), int(r[-1].split(":")[1])) for r in records[2:]]
    refs_edgelist = [(e1, e2) for (e1, e2) in refs_edgelist if e1 != 17874530 and e2 != 17874530] # 17874530 is the one PMID with no available metadata.
    refs_output = [{"references" : refs_edgelist}, {"references" : np.ones(len(refs_edgelist)).tolist()}]

    authorship_output = utils.generate_edges(utils.preprocess(df, "authors", "pmid"), "authorship", params["seed"])

    srcid_output = utils.generate_edges(utils.preprocess(df, "journal", "pmid"), "srcid", params["seed"])

    mesh_input = utils.preprocess(df, "mesh", "pmid")
    mesh_output = utils.generate_edges(mesh_input, "mesh", params["seed"], int(np.mean(mesh_input.apply(len))))

    node_map, inv_node_map, edge_lists_remap = remapper(set(df["pmid"]), [refs_output[0], authorship_output[0], mesh_output[0], srcid_output[0]])

    data = generate_data(df, node_map, inv_node_map, edge_lists_remap, 
        [refs_output[1], authorship_output[1], mesh_output[1], srcid_output[1]], torch.device("cpu"))
    data.inv_node_map = [inv_node_map] # Needed for re-ordering embeddings.
    data["paper"].x = torch.tensor(np.stack(df.set_index("pmid").loc[list(data.inv_node_map[0].keys())].tfidf)).float()

    save_path = str(Path(params["data"]["save_path"]["pubmed"]))
    torch.save(data, save_path)
    









