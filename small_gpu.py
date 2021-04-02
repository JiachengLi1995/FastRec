import os
import torch

from src.models import model_factory
from src.dataloaders import dataloader_factory
from src.datasets import dataset_factory
from src.trainers import trainer_factory
from src.utils.options import parser
from src.utils.loggers import save_state_dict
from src.utils.utils import *

if __name__ == '__main__':
    print("If GPU is too small for the large-scale dataset...")

    # hyper-parameter config
    args = parser.parse_args()
    ckpt_root = setup_train(args)

    # dataset 
    dataset = dataset_factory(args)

    # large embedding table (on cpu)
    global_num_items = set(list(dataset.smap.keys()))
    local_num_items = set([])
    large_embed = torch.zeros(len(dataset.smap)+2, args.trm_hidden_dim) # embeddings + [cloze, pad] 
    torch.nn.init.xavier_uniform_(large_embed[:-2])

    args.num_epochs = args.local_epochs

    for idx, i in enumerate(range(args.global_epochs)):
        # subdataset and data loader
        subdataset = dataset.subdataset(k=args.subset_size)
        train_loader, val_loader, test_loader, subdataset = dataloader_factory(args, subdataset)
        temp_num_items = set(list(subdataset.smap.keys()))
        local_num_items = local_num_items | temp_num_items
        print("[%d] #user %d, #item %d, local cover rate %.4f, total cover rate %.4f start negative sampling ..." % \
            (idx, len(subdataset.umap), len(subdataset.smap), len(temp_num_items)/len(global_num_items), len(local_num_items)/len(global_num_items)))
        # bridge large embed table with local embed table
        # eg. smap = {5:1} --> 5(in large)==1(in local)
        smap_r = {subdataset.smap[s]:s for s in subdataset.smap}
        # so local_embed[1] = large_embed[samp_r[1]] 
        mapping = [smap_r[k] for k in range(len(smap_r))] + [-2, -1] # include cloze & pad token
        # model setup
        if idx == 0:
            model = model_factory(args)
            print("Model size:", sum(p.numel() for p in model.parameters() if p.requires_grad))
        else:
            model.item_emb = torch.nn.Embedding.from_pretrained(large_embed[mapping], freeze=False, padding_idx=-1)
            model.pad_token = model.item_emb.padding_idx
        # trainer setup
        trainer = trainer_factory(args, model, train_loader, val_loader, test_loader, ckpt_root, subdataset.data)
        
        # model training
        trainer.train()
        model = trainer.model.to('cpu')
        large_embed[mapping] = model.item_emb.weight.data

        # load large embedding table
        torch.save(large_embed, os.path.join(ckpt_root, 'models', 'best_emb.pth'))

    # model testing and saving
    train_loader, val_loader, test_loader, subdataset = dataloader_factory(args, dataset)

    # load large embedding table
    large_embed = torch.load(os.path.join(ckpt_root, 'models', 'best_emb.pth'))
    model.item_emb = torch.nn.Embedding.from_pretrained(large_embed, freeze=False, padding_idx=-1)
    model.pad_token = model.item_emb.padding_idx
    
    # loading and saving
    args.device = 'cpu' # assume that we cannot load this table to gpu
    trainer = trainer_factory(args, model, train_loader, val_loader, test_loader, ckpt_root, dataset.data)
    save_state_dict(trainer._create_state_dict(), os.path.join(ckpt_root, 'models'), 'best_acc_model.pth')
    
    # testing
    trainer.test() 
    trainer.logger_service.complete({'state_dict': (trainer._create_state_dict())})
    trainer.writer.close()
