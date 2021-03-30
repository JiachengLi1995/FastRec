# FastRec

Pipeline to obtain item embeddings of [Amazon Review Data (2018)](https://nijianmo.github.io/amazon/index.html).

Model is based on sequential recommender SASRec. [[paper](https://cseweb.ucsd.edu/~jmcauley/pdfs/icdm18.pdf)][[github](https://github.com/kang205/SASRec)]

## Dependencies

```bash
pip install -r SASRec/requirements.txt
```

## Data Preprocess

Our pipeline take .gz file from [Amazon Review Data (2018)](https://nijianmo.github.io/amazon/index.html) as input. 

```bash
python data_process.py --file_path [gz file path] --output_path [output directory]
```

data_process.py will generate train/val/test json files with leave-one-out strategy as SASRec paper mentioned. "umap.json" and "smap.json" contain the dictionary that convert user id and item id in raw data into numbers used in model.

Please output all processed files under path `SASRec/datasets/data/dataset_name/`.

| Name        | Description     | Example |
|-------------|--------------|---------------|
| train.json  | Training set of item sequence, using the n-2 interactions       | {0: [0, 1, 2, 3, 4]}   |
| val.json    | Validation set of item sequence, using the (n-1)th interaction  | {0: [5]}               |
| test.json   | Testing set of item sequence, using the (n)th interaction       | {0: [6]}               |
| umap.json   | Mapping real user/sess id to consecutive integer id             |  {"A2HOI48JK8838M": 0} |
| smap.json   | Mapping real item id to consecutive integer id                  | {"B00004U9V2": 1}      |




## Model Training

Model training with our default hyper-parameter configure.

```bash
cd SASRec
bash scripts/train.sh ${gpu_id} ${data_name}
```
Evaluation methods: NDCG@K, Recall@K, MRR, AUC


## Amazon Dataset Statistics

| #Users | #Items | #Interactions |  Sparsity |
|--------|--------|---------------|-----------|
| ~44 Million |  ~15 Million  | ~0.2 Billion |  99.99997% |

## Results

For evaluation, we uniformly sample 100,000 users and for each user we uniformly sample 1000 items as negative candidates, which will be ranked with the single positive item (form a ranking list containing 1001 items).
Training time is about 25 hours.


|        | NDCG@10 | Recall@10 | MRR | AUC |
|--------|---------|-----------|-----|-----|
| Validation Set| 0.35626 | 0.52921 | 0.31500 | 0.90279 |
| Test Set| 0.33591 | 0.50647 | 0.29566 | 0.89410 |

## Notebook

We also provide a notebook `Amazon_Beauty.ipynb` to demostrate the whole training process with a toy Amazon dataset.









