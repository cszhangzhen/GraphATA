[![DOI](https://zenodo.org/badge/925059335.svg)](https://doi.org/10.5281/zenodo.14777073)

# GraphATA
Aggregate to Adapt: Node-Centric Aggregation for Multi-Source-Free Graph Domain Adaptation (WWW-2025).

![](https://github.com/cszhangzhen/GraphATA/blob/main/fig/model.png)

This is a PyTorch implementation of the GraphATA algorithm, which tries to address the multi-source domain adaptation problem without accessing the labelled source graph. Unlike previous multi-source domain adaptation approaches that aggregate predictions at model level, we introduce a novel model named GraphATA which conducts adaptation at node granularity. Specifically, we parameterize each node with its own graph convolutional matrix by automatically aggregating weight matrices from multiple source models according to its local context, thus realizing dynamic adaptation over graph structured data. We also demonstrate the capability of GraphATA to generalize to both model-centric and layer-centric methods.

## Requirements
* python3.8
* pytorch==1.13.1
* torch-scatter==2.1.0
* torch-sparse==0.6.15
* torch-cluster==1.6.0
* torch-geometric==2.4.0
* numpy==1.23.4
* scipy==1.9.3

## Datasets
Datasets used in the paper are all publicly available datasets.

## Quick Start For Node Classification:
Just execuate the following command for source model pre-training:
```
python train_source_node.py
```
Then, execuate the following command for adaptation:
```
python train_target_node.py
```

## Quick Start For Graph Classification:
Just execuate the following command for source model pre-training:
```
python train_source_graph.py
```
Then, execuate the following command for adaptation:
```
python train_target_graph.py
```

## Citing
If you find GraphATA useful for your research, please consider citing the following paper:
```
@inproceedings{zhang2025aggregate,
  title={Aggregate to Adapt: Node-Centric Aggregation for Multi-Source-Free Graph Domain Adaptation},
  author={Zhang, Zhen and He, Bingsheng},
  booktitle={Proceedings of the ACM on Web Conference 2025},
  pages={4420--4431},
  year={2025}
}
```

