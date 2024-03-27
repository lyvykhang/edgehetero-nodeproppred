from pathlib import Path
import os
import yaml

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import to_hetero
from tqdm import tqdm
from ogb.nodeproppred import Evaluator

import utils
import models


def train(model, data, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(data.x_dict, data.adj_t_dict)["paper"]
    loss = F.nll_loss(out[train_idx], data["paper"].y[train_idx].squeeze())
    loss.backward()
    optimizer.step()
    
    return float(loss), out


@torch.no_grad()
def test(model, data, idx, dataset, out=None):
    model.eval()
    
    out = model(data.x_dict, data.adj_t_dict)['paper'] if out is None else out
    loss = F.nll_loss(out[idx], data["paper"].y[idx].squeeze())
    y_pred = out[idx].argmax(dim=-1, keepdim=True)
    if dataset == "ogbnarxiv":
        evaluator = Evaluator(name="ogbn-arxiv")
        acc = evaluator.eval({"y_true": data["paper"].y[idx], "y_pred": y_pred})["acc"]
    else:
        acc = int((y_pred == data["paper"].y[idx]).sum()) / int(idx.shape[0])

    return acc, loss


def get_model(verbose=False):
    num_layers = params["model"]["num_layers"]
    in_channels = data["paper"].x.shape[1]
    out_channels = len(torch.unique(data["paper"].y))
    hidden_channels = params["model"]["hidden_channels"]
    dropout = params["model"]["dropout"]

    if params["model"]["name"] == "GCN":
        model = models.GCN(num_layers, in_channels, out_channels, hidden_channels, dropout).to(DEVICE)
    elif params["model"]["name"] == "SAGE":
        model = models.SAGE(num_layers, in_channels, out_channels, hidden_channels, dropout).to(DEVICE)
    elif params["model"]["name"] == "GCNJKNet":
        model = models.GCNJKNet(num_layers, in_channels, out_channels, hidden_channels, dropout, mode="max").to(DEVICE)
    elif params["model"]["name"] == "SGC":
        model = models.SGC(num_layers, in_channels, out_channels).to(DEVICE)
    
    model = to_hetero(model, data.metadata(), aggr="mean").to(DEVICE)
    if verbose:
        print("No. parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    return model


if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Will run on: {DEVICE}.")

    project_root: Path = utils.get_project_root()
    os.chdir(project_root)
    with open(str(project_root / "config/experiments_config.yaml")) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    with open(str(project_root / "config/data_generation_config.yaml")) as f:
        data_gen_params = yaml.load(f, Loader=yaml.FullLoader)

    path_to_data = str(Path(params["data"]["graph_dataset"][params["dataset"]]))
    data = torch.load(path_to_data)

    data = data.edge_type_subgraph(utils.edge_type_selection(params["edge_type_selection"][params["dataset"]]))

    if any(i in params["node_embs"] for i in ["simtg", "tape"]):
        path_embs = str(Path(params["data"][f"{params['node_embs']}_embs"][params["dataset"]]))
        if params["node_embs"] == "tape":
            init_x_shape = (19717, 768) if params["dataset"] == "pubmed" else (data["paper"].num_nodes, 768)
            features = np.array(np.memmap(path_embs, mode='r', dtype=np.float16, shape=init_x_shape))
            if params["dataset"] == "pubmed":
                features = np.delete(features, [2459], axis=0) # manually remove index corresponding to ID with no metadata from the precomputed embeddings.
            data["paper"].x = torch.from_numpy(features).to(torch.float32)
        else:
            data["paper"].x = torch.load(path_embs).type(torch.float32)

    print("Loaded pre-trained node embeddings of type={} and shape={}.".format(params["node_embs"], data["paper"].x.shape))

    data.to(DEVICE)

    path_to_model = f"models/{params['dataset']}_{params['model']['name']}"
    all_runs_accs = []

    for run in range(params["runs"]):
        model = get_model(verbose=True) if run == 0 else get_model()
        
        if params["dataset"] == "ogbnarxiv" or (params["dataset"] == "pubmed" and data_gen_params["pubmed_fixed_split"]):
            train_idx, val_idx, test_idx = data["paper"].train_idx, data["paper"].val_idx, data["paper"].test_idx
        else: # Randomly split nodes of each class; not compatible with SimTG (fixed split is required to finetune the LM).
            data.to(torch.device("cpu"))
            train_idx, val_idx, test_idx = utils.per_class_idx_split(data, run) # use run no. as seed.
            data.to(DEVICE)

        optimizer = torch.optim.Adam(params=model.parameters(), 
            weight_decay=params["optimizer"]["weight_decay"], 
            lr=params["optimizer"]["lr"])
        
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)

        best_acc = best_epoch = -1
        for epoch in tqdm(range(params["epochs"]), desc=f"Run {run:02d}"):
            train_loss, out = train(model, data, train_idx, optimizer)

            val_acc, val_loss = test(model, data, val_idx, params["dataset"], out=out)
            scheduler.step()

            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
                torch.save(model.state_dict(), f"{path_to_model}_run{run}_best.pth")
            elif epoch - best_epoch > params["early_stop_threshold"]:
                tqdm.write(f"Early stopped training for run {run:02d} at epoch {epoch:02d}.")
                break

        model.load_state_dict(torch.load(f"{path_to_model}_run{run}_best.pth"))
        test_acc, test_loss = test(model, data, test_idx, params["dataset"])

        tqdm.write(f"Run {run:02d}: Best Epoch {best_epoch:02d}, Best Val Acc {best_acc:.4f}, Test Acc {test_acc:.4f}.")
        all_runs_accs.append([best_acc, test_acc])
        torch.cuda.empty_cache()
    
    # data.to(torch.device("cpu"))

    all_runs_accs = torch.tensor(all_runs_accs)
    print("* ============================= ALL RUNS =============================")
    print(f"Best Val Acc: {all_runs_accs[:, 0].max().item()*100:.2f}, Best Test Acc: {all_runs_accs[:, 1].max().item()*100:.2f}.")
    print(f"Avg. Val Acc: {all_runs_accs[:, 0].mean().item()*100:.2f} ± {all_runs_accs[:, 0].std().item()*100:.2f}", 
        f"Avg. Test Acc: {all_runs_accs[:, 1].mean().item()*100:.2f} ± {all_runs_accs[:, 1].std().item()*100:.2f}.")