from pathlib import Path
from itertools import combinations
import random
from collections import Counter
from tqdm import tqdm
import pandas as pd
import torch
from math import floor


def get_project_root() -> Path:
    return Path(__file__).parent.parent


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
        edge_stores (dict): dict value = edge weight. 
    """
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


def zip_dicts(*dcts):
    """Yields tuples aggregating values from common keys (similar to
    the zip() function, but for dicts).
    Helper function for generate_data() (see generate_graph_data.py).

    Args:
        dcts (*dict): any number of dicts to zip.

    Output:
        (k, v1, v2, ...), where v1, ..., vn indicates the value
        of key k in the respective input dicts.
    """
    if not dcts:
        return
    for i in set(dcts[0]).intersection(*dcts[1:]):
        yield (i,) + tuple(d[i] for d in dcts)


def edge_type_selection(selection):
    result = []
    for i in selection:
        if i == "references":
            result.append(('paper', 'references', 'paper'))
        else:
            result.append(('paper', f'shares_{i}_with', 'paper'))
    return result


def train_valid_test(nodes, test_size, valid_size, random_state):
    # nodes: set of node IDs on which to perform the splitting.
    random.seed(random_state)
    test = set(random.sample(nodes, floor(test_size * len(nodes))))
    rest = nodes - test
    valid = set(random.sample(rest, floor(valid_size * len(rest))))
    train = rest - valid

    train_mask = torch.tensor(list(train), dtype=torch.long)
    valid_mask = torch.tensor(list(valid), dtype=torch.long)
    test_mask = torch.tensor(list(test), dtype=torch.long)

    return train_mask, valid_mask, test_mask