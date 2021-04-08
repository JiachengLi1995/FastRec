import gc
import os

import torch
import numpy as np

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
    
    # trainer setup
    trainer = trainer_factory(args, model, train_loader, val_loader, test_loader, ckpt_root, dataset.data)

    # test
    trainer.validate_test(0, 0, mode='test')

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

    index_path = os.path.join(ckpt_root, 'models', 'test.idx')
    if os.path.exist(index_path):
        print("read index")
        index = faiss.read_index(index_path)
    else:
        print("build index")
        index = faiss.index_factory(d, 'HNSW32', faiss.METRIC_INNER_PRODUCT)   # build the index
        index.add(xb)                                                          # add vectors to the index
        print("index built")
        faiss.write_index(index, index_path)
        print("index stored")

    k = 10
    xb_tensor = torch.Tensor(xb).transpose()
    for batch_idx, batch in enumerate(test_loader):      
        users, seqs, candidates, labels, length = batch
        xq = self.model(seqs, length=length, mode="serving").detach().cpu()
        V, I = torch.topk((xq @ xb_tensor), k)
        print(I)
        V, I = index.search(xq.numpy().astype('float32'), k)
        print(I)
        break