import gc

import torch

from src.models import model_factory
from src.dataloaders import dataloader_factory
from src.datasets import dataset_factory
from src.trainers import trainer_factory
from src.utils.options import parser
from src.utils.utils import *

if __name__ == '__main__':

    # hyper-parameter config
    args = parser.parse_args()
    ckpt_root = setup_train(args)

    # dataset and data loader
    print("dataset and data loader ...")
    dataset = dataset_factory(args)
    train_loader, val_loader, test_loader, dataset = dataloader_factory(args, dataset)

    # model setup
    print("model setup ...")
    model = model_factory(args)
    if args.load_pretrained_weights is not None:
        print("weights loading from %s ..." % args.load_pretrained_weights)
        model = load_pretrained_weights(model, args.load_pretrained_weights)
    print("Model size:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    # # trainer setup
    # trainer = trainer_factory(args, model, train_loader, val_loader, test_loader, ckpt_root, dataset.data)

    # # test
    # trainer.validate_test(0, 0, mode='test')

    # faiss

    del train_loader
    del val_loader
    del dataset
    gc.collect()

    import faiss
    import numpy as np

    d = args.trm_hidden_dim                                 # dimension
    nb = model.item_emb.weight.size(0)                      # item pool size
    xb = model.item_emb.weight.data.cpu().numpy()           # item embeddings table

    print("build index")
    index = faiss.index_factory(d, 'HNSW32', faiss.METRIC_INNER_PRODUCT)   # build the index
    index.add(xb)                                                          # add vectors to the index
    print("index built")
    import pdb; pdb.set_trace()

    for batch_idx, batch in enumerate(test_loader):      
        users, seqs, candidates, labels, length = batch
        feats = self.model(seqs, length=length, mode="serving")
        print(feats.size())
        exit()