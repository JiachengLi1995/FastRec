from datasets import dataset_factory
from .sasrec import SASRecDataloader


DATALOADERS = {
    SASRecDataloader.code(): SASRecDataloader,
}


def dataloader_factory(args):
    dataset = dataset_factory(args)
    dataloader = DATALOADERS[args.dataloader_code]
    dataloader = dataloader(args, dataset)
    train, val, test = dataloader.get_pytorch_dataloaders()
    return train, val, test, dataset
