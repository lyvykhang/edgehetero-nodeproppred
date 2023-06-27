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
def test(model, data, idx, out=None):
    model.eval()
    
    out = model(data.x_dict, data.adj_t_dict)['paper'] if out is None else out
    loss = F.nll_loss(out[idx], data["paper"].y[idx].squeeze())
    y_pred = out[idx].argmax(dim=-1, keepdim=True)
    evaluator = Evaluator(name="ogbn-arxiv")
    acc = evaluator.eval({"y_true": data["paper"].y[idx], "y_pred": y_pred})["acc"]

    return acc, loss


if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Will run on: {DEVICE}.")

    project_root: Path = utils.get_project_root()
    os.chdir(project_root)
    with open(str(project_root / "config/gcn_config.yaml")) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    path_to_data = str(Path(params["data"]["graph_dataset"]))
    data = torch.load(path_to_data)

    data = data.edge_type_subgraph(utils.edge_type_selection(params["edge_type_selection"]))

    print("Loading embeddings...")
    path_scibert = str(Path(params["data"]["embeddings"]))
    data["paper"].x = torch.load(path_scibert)
    
    model = models.GCN(params["model"]["num_layers"], data["paper"].x.shape[1], 
                       len(torch.unique(data["paper"].y)), params["model"]["hidden_features"])
    model = to_hetero(model, data.metadata(), aggr="mean").to(DEVICE)
    print("No. parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    data.to(DEVICE)

    path_to_model = str(Path(params["model_path_prefix"]))
    all_runs_accs = []

    for run in range(params["runs"]):
        models.GCN.reset_parameters(model)

        optim = [dict(params=conv.parameters(), weight_decay=params["optimizer"]["weight_decay"]) \
                 if i == 0 else dict(params=conv.parameters(), weight_decay=0) for i, conv in enumerate(model.convs)]
        optimizer = torch.optim.Adam(optim, lr=params["optimizer"]["lr"]) # weight decay on first conv. only.
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=params["scheduler"]["factor"], 
            patience=params["scheduler"]["patience"], 
            threshold=params["scheduler"]["threshold"], 
            min_lr=params["scheduler"]["min_lr"], 
            cooldown=params["scheduler"]["cooldown"])

        best_acc = best_epoch = -1
        for epoch in tqdm(range(params["epochs"]), desc=f"Run {run:02d}"):
            train_loss, out = train(model, data, data["paper"].train_idx, optimizer)

            val_acc, val_loss = test(model, data, data["paper"].val_idx, out=out)
            scheduler.step(val_loss)

            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
                torch.save(model.state_dict(), f"{path_to_model}_run{run}_best.pth")
            elif epoch - best_epoch > params["early_stop_threshold"]:
                tqdm.write(f"Early stopped training for run {run:02d} at epoch {epoch:02d}.")
                break

        model.load_state_dict(torch.load(f"{path_to_model}_run{run}_best.pth"))
        test_acc, test_loss = test(model, data, data["paper"].test_idx)

        tqdm.write(f"Run {run:02d}: Best Epoch {best_epoch:02d}, Best Val Acc {best_acc:.4f}, Test Acc {test_acc:.4f}.")
        all_runs_accs.append([best_acc, test_acc])
    
    # data.to(torch.device("cpu"))

    all_runs_accs = torch.tensor(all_runs_accs)
    print(f"##### ALL RUNS #####")
    print(f"Best Val Acc: {all_runs_accs[:, 0].max().item():.4f}, Best Test Acc: {all_runs_accs[:, 1].max().item():.4f}.")
    print(f"Avg. Val Acc: {all_runs_accs[:, 0].mean().item():.4f} ± {all_runs_accs[:, 0].std().item():.4f}", f"Avg. Test Acc: {all_runs_accs[:, 1].mean().item():.4f} ± {all_runs_accs[:, 1].std().item():.4f}.")