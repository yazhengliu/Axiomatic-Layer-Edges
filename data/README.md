# Explanations of GNN on Evolving Graphs via Axiomatic Layer Edges
This is the Pytorch Implementation of [Explanations of GNN on Evolving Graphs via Axiomatic  Layer Edges](https://openreview.net/forum?id=pXN8T5RwNN&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2025%2FConference%2FAuthors%23your-submissions))
# Train a Model
You need to train a model first. To train the model, use th following command, replacing $(dataset) with the name of dataset. **Pheme**, **Weibo**, **Chi**, and **NYC** are the node classification datasets. **Bitcoinalpha**, **bitcoinotc**, and **UCI** are the link prediction datasets. **Mutag**, **Clintox**, **IMDB-BINARY**, and **REDDIT-BINARY** are the graph classification datasets. 
```bash
python train.py --data $(dataset)
```
# Provide an explanation
Once you have a  model, the next step is to provide an explanation. You can find the important input edges or the layer edges as the explanations. Use the following command to obtain the explanation:
```bash
python main_explain.py --data $(dataset)
```
# Citation
```bash
@article{liu2025axiomatic,
  title={Explanations of GNN on Evolving Graphs via Axiomatic Layer Edges},
  author={Yazheng Liu and Sihong Xie},
  conference={International Conference on Learning Representations},
  year={2025}
}
```
