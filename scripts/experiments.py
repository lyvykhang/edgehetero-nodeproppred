from pathlib import Path
import os
import yaml
import utils
import models

import torch
import torch.nn.functional as F
from torch_geometric.nn import to_hetero
from tqdm import tqdm
from ogb.nodeproppred import Evaluator


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


if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Will run on: {DEVICE}.")

    project_root: Path = utils.get_project_root()
    os.chdir(project_root)
    with open(str(project_root / "config/experiments_config.yaml")) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    path_to_data = str(Path(params["data"]["graph_dataset"][params["dataset"]]))
    data = torch.load(path_to_data)

    data = data.edge_type_subgraph(utils.edge_type_selection(params["edge_type_selection"][params["dataset"]]))

    if params["use_scibert"]:
        path_scibert = str(Path(params["data"]["embeddings"][params["dataset"]]))
        data["paper"].x = torch.load(path_scibert)

    data.to(DEVICE)

    path_to_model = f"models/{params['dataset']}_{params['model']['name']}"
    all_runs_accs = []

    for run in range(params["runs"]):
        if params["model"]["name"] == "GCN":
            model = models.GCN(params["model"]["num_layers"], data["paper"].x.shape[1], 
                len(torch.unique(data["paper"].y)), params["model"]["hidden_channels"], 
                params["model"]["dropout"]).to(DEVICE)
        elif params["model"]["name"] == "SAGE":
            model = models.SAGE(params["model"]["num_layers"], data["paper"].x.shape[1], 
                len(torch.unique(data["paper"].y)), params["model"]["hidden_channels"], 
                params["model"]["dropout"]).to(DEVICE)
        elif params["model"]["name"] == "GCNJKNet":
            model = models.GCNJKNet(params["model"]["num_layers"], data["paper"].x.shape[1], 
                len(torch.unique(data["paper"].y)), params["model"]["hidden_channels"], 
                params["model"]["dropout"], mode="cat").to(DEVICE)
        elif params["model"]["name"] == "SGC":
            model = models.SGC(params["model"]["num_layers"], data["paper"].x.shape[1], 
                len(torch.unique(data["paper"].y))).to(DEVICE)
        
        model = to_hetero(model, data.metadata(), aggr="mean").to(DEVICE)
        if run == 0:
            print("No. parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
        
        if params["dataset"] == "ogbnarxiv":
            train_idx, val_idx, test_idx = data["paper"].train_idx, data["paper"].val_idx, data["paper"].test_idx
        else: # Randomly split nodes of each class.
            splits = [torch.empty(0, dtype=torch.long)]*3
            for i in range(0, data["paper"].y.unique().shape[0]):
                per_class = torch.tensor(range(data["paper"].y.shape[0]))[(data["paper"].y == i).squeeze()]
                per_class_split = utils.train_valid_test(set(per_class.tolist()), 0.2, 0.2, run) # use run no. as seed.
                for j in range(0, 3):
                    splits[j] = torch.cat((splits[j], per_class_split[j]))
            train_idx, val_idx, test_idx = splits

        optimizer = torch.optim.Adam(params=model.parameters(), 
            weight_decay=params["optimizer"]["weight_decay"], 
            lr=params["optimizer"]["lr"])
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=params["scheduler"]["factor"], 
            patience=params["scheduler"]["patience"], 
            threshold=params["scheduler"]["threshold"], 
            min_lr=params["scheduler"]["min_lr"], 
            cooldown=params["scheduler"]["cooldown"])

        best_acc = best_epoch = -1
        for epoch in tqdm(range(params["epochs"]), desc=f"Run {run:02d}"):
            train_loss, out = train(model, data, train_idx, optimizer)

            val_acc, val_loss = test(model, data, val_idx, params["dataset"], out=out)
            scheduler.step(val_loss)

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
    
    # data.to(torch.device("cpu"))

    all_runs_accs = torch.tensor(all_runs_accs)
    print(f"##### ALL RUNS #####")
    print(f"Best Val Acc: {all_runs_accs[:, 0].max().item():.4f}, Best Test Acc: {all_runs_accs[:, 1].max().item():.4f}.")
    print(f"Avg. Val Acc: {all_runs_accs[:, 0].mean().item():.4f} ± {all_runs_accs[:, 0].std().item():.4f}", 
        f"Avg. Test Acc: {all_runs_accs[:, 1].mean().item():.4f} ± {all_runs_accs[:, 1].std().item():.4f}.")