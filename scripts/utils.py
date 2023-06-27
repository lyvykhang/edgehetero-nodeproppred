from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).parent.parent


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
        elif i == "authorship":
            result.append(('paper', 'shares_author_with', 'paper'))
        elif i == "fos":
            result.append(('paper', 'shares_fos_with', 'paper'))
        elif i == "venue":
            result.append(('paper', 'shares_venue_with', 'paper'))
    return result