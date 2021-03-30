import torch

from options import args
from models import model_factory
from dataloaders import dataloader_factory
from trainers import trainer_factory
from utils import *

def train():
    ckpt_root = setup_train(args)
    train_loader, val_loader, test_loader, dataset = dataloader_factory(args)
    model = model_factory(args)
    print(model)
    if args.load_pretrained_weights is not None:
        print("weights loading from %s ..." % args.load_pretrained_weights)
        model = load_pretrained_weights(model, args.load_pretrained_weights)
    print("Model size:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    trainer = trainer_factory(args, model, train_loader, val_loader, test_loader, ckpt_root, dataset.data)
    trainer.train()

if __name__ == '__main__':
    train()
