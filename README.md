# edgehetero-nodeproppred

Repository for the ["Improving Article Classification Using Edge-Heterogeneous Graph Convolutional Networks"](https://elsevier.atlassian.net/wiki/spaces/RCODS/pages/119601060404383/Improving+Article+Classification+with+Edge-Heterogeneous+Graph+Convolutional+Networks) 2022-2023 UvA MSc DS thesis project. For prior work on this topic, see [this repo](https://github.com/elsevier-research/ds-graph-clf-experiments).

The code only covers the `ogbn-arxiv` benchmark use case. Work on internal datasets is documented on Databricks, Confluence, and the thesis itself [(Overleaf read-only link)](https://www.overleaf.com/read/fqjgcfrkqtzy). Results and instructions for reproducibility at the end.

The goal is to implement a data preparation pipeline with edge heterogeneity and BERT-based node features to boost accuracy while keeping the model architecture itself relatively lightweight. 

## File Structure ##
```
edgehetero-nodeproppred/
├─ config/
│  ├─ data_generation_config.yaml
│  ├─ gcn_config.yaml
├─ data/
│  ├─ embeddings/
│  ├─ processed/
│  ├─ tables/
├─ documents/
├─ models/
├─ notebooks/
│  ├─ ogbnarxiv_process_mag_data.ipynb
├─ scripts/
│  ├─ gcn_experiments.py
│  ├─ models.py
│  ├─ utils.py
```

## ETL ##
The dataset provides the mapping from node IDs to MAG IDs, which is used to retrieve additional metadata fields. Since MAG has been discontinued, we use a July 2020 snapshot of MAG, made available by the Open Academic Graph project, hosted on AMiner [(link)](https://www.aminer.cn/oag-2-1). The ~240M MAG papers are split into 17 ~10GB chunks, each chunk containing 3 ~10GB text files of records adhering to the schema listed under *Data Description*. All chunks were downloaded locally beforehand, and records corresponding to IDs in `ogbn-arxiv` were saved out.
  
The raw texts of titles and abstracts linked under the dataset description [on OGB](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv) are also used.

See the notebook `ogbnarxiv_process_mag_data.ipynb` for relevant code.

### Edge Types ###

Using the `authors`, `venue`, and `fos` features (fields of study), the following  additional edge types are constructed: 
- `(paper, shares_author_with, paper)`: 2 papers are connected if they share an author, weighted with the no. of shared authors.
- `(paper, shares_venue_with, paper)`: 2 papers are connected if they were published at the same venue, all edges 1-weighted.
- `(paper, shares_fos_with, paper)`: 2 papers are connected if they share a field of study, weighted with the no. of shared fields. 

For the latter two, we sample the mean number of IDs associated with a FoS/venue and only create edges between that many nodes for each unique FoS/venue. Otherwise, they lead to massive subgraphs, as the no. of edges explodes due to the combination function. All subgraphs are *undirected*.

| Type      | \|E\|     | Edge Homophily | Non-Isolated Nodes |
|-----------|-----------|----------------|--------------------|
| Citations | 2,315,598 | 0.654          | **169,343**        |
| Authors   | 6,749,335 | 0.58           | 157,067            |
| FoS       | 8,279,687 | 0.319          | 144,714            |
| Venue     | 600,930   | 0.077          | 17,848             |

See the script `ogbnarxiv_hetero_transform.py` for relevant code.

### Node Features ###

The tensor `ogbnarxiv_scibert_tensor_ordered.pt` contains the inferred SciBERT embeddings for the concatenated titles and abstracts of all IDs, using the base `scibert-scivocab-uncased` without any tuning. Node2Vec was also tested for the thesis, but did not improve metrics on `ogbn-arxiv`, hence it was excluded here.

Done on [this Databricks notebook](https://elsevier.cloud.databricks.com/?o=0#notebook/5431542/command/5467567).

### Ablation Results ###

| Model   | Embeddings         | Subgraphs                  | Acc.      | F1        |
|---------|--------------------|----------------------------|-----------|-----------|
| GCNConv | SkipGram (Default) | Refs                       | 69.9%     | 0.694     |
| GCNConv | SciBERT            | Refs                       | 73.1%     | 0.722     |
| GCNConv | SciBERT            | Refs, Authors              | 74.7%     | 0.739     |
| GCNConv | SciBERT            | Refs, Authors, FoS         | **75.1%** | **0.743** |
| GCNConv | SciBERT            | Refs, Authors, FoS, Venues | 75%       | 0.743     |
| GCNConv | SciBERT, N2V       | Refs, Authors, FoS         | 74.8%     | 0.739     |

### Instructions for Use ###
Clone the repository with Git LFS. 
#### For environment setup: ####
```
conda create --name EHRGCN python=3.10
conda activate EHRGCN
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 cudatoolkit=11.7 -c pytorch
conda install -c pyg pytorch-sparse
conda install -c pyg pytorch-scatter
conda install -c pyg pytorch-cluster
conda install -c pyg pyg
pip install tqdm
pip install ogb
pip install PyYAML
```

Run `scripts/gcn_experiments.py` to train model and print results. All relevant parameters can be specified in `config/gcn_config.yaml`. 

The data preprocessing code is included for reference, but the SciBERT embeddings (`data/embeddings/ogbnarxiv_scibert_tensor_ordered.pt`) and HeteroData object (`data/data_ogbnarxiv_ref_au_fos_venue.pt`) have both been pre-generated. Also, the DataFrame `data/tables/ogbnarxiv_mag_metadata.parquet.gzip` contains the extracted MAG metadata fields, along with the titles and abstracts from `titleabs.tsv`. 


