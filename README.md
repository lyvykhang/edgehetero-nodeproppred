# edgehetero-nodeproppred

Repository for the "Improving Article Classification Using Edge-Heterogeneous Graph Convolutional Networks" 2022-2023 UvA MSc DS thesis project, in collaboration with Elsevier. Repo only covers the `ogbn-arxiv` benchmark use case. See the `technical_report.pdf` for more details.

```
edgehetero-nodeproppred/
├─ config/
│  ├─ data_generation_config.yaml
│  ├─ gcn_config.yaml
├─ data/
│  ├─ embeddings/
│  │  ├─ ogbnarxiv_scibert_tensor_ordered.pt
│  ├─ tables/
│  │  ├─ ogbnarxiv_mag_metadata.parquet.gzip
├─ models/
├─ notebooks/
│  ├─ ogbnarxiv_process_mag_data.ipynb
├─ scripts/
│  ├─ gcn_experiments.py
│  ├─ models.py
│  ├─ ogbnarxiv_hetero_transform.py
│  ├─ utils.py
├─ technical_report.pdf
```

## Reproduce Experiments ##
Clone the repository with Git LFS. For environment setup:
```
conda create --name EHGNN python=3.10
conda activate EHGNN
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 cudatoolkit=11.7 -c pytorch
conda install -c pyg pytorch-sparse
conda install -c pyg pytorch-scatter
conda install -c pyg pytorch-cluster
conda install -c pyg pyg
pip install tqdm
pip install ogb
pip install PyYAML
```
The DataFrame `data/tables/ogbnarxiv_mag_metadata.parquet.gzip` contains the extracted MAG metadata fields, along with the aforementioned raw titles and abstracts.

The SciBERT embeddings are pre-generated and provided at `data/embeddings/ogbnarxiv_scibert_tensor_ordered.pt`. 

**Generate data**: run `scripts/ogbnarxiv_hetero_transform.py`, which downloads the dataset and saves out the transformed `HeteroData` object.

**To reproduce**: run `scripts/experiments.py` to train model and print results. Model choice and all relevant parameters can be specified in `config/experiments_config.yaml`.

## Results ##
| Model  	| Val. Acc.     	| Test Acc.     	| # Params  	|
|--------	|---------------	|---------------	|-----------	|
| GCN    	| 75.86% ± 0.12 	| 74.61% ± 0.06 	| 621,944   	|
| GCN+JK 	| 76.29% ± 0.07 	| 74.72% ± 0.24 	| 809,512   	|
| SAGE   	| 76.05% ± 0.07 	| 74.61% ± 0.13 	| 1,242,488 	|
| SGC    	| 75.15% ± 0.05 	| 74.19% ± 0.04 	| 92,280    	|