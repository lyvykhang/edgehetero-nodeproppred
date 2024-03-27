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


def zip_dicts(*dcts):
    """
    UNUSED: Yields tuples aggregating values from common keys (similar to
    the zip() function, but for dicts).

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


def per_class_idx_split(data, random_state):
    splits = [torch.empty(0, dtype=torch.long)]*3
    for i in range(0, data["paper"].y.unique().shape[0]):
        per_class = torch.tensor(range(data["paper"].y.shape[0]))[(data["paper"].y == i).squeeze()]
        per_class_split = train_valid_test(set(per_class.tolist()), 0.2, 0.2, random_state)
        for j in range(0, 3):
            splits[j] = torch.cat((splits[j], per_class_split[j]))
    return splits # train, val, test.