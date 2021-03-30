from .sasrec import SASRecAllTrainer, SASRecSampleTrainer


TRAINERS = {
    SASRecAllTrainer.code(): SASRecAllTrainer,
    SASRecSampleTrainer.code(): SASRecSampleTrainer,
}


def trainer_factory(args, model, train_loader, val_loader, test_loader, ckpt_root, user2seq):
    trainer = TRAINERS[args.trainer_code]
    if "cpu" not in args.trainer_code:
        return trainer(args, model, train_loader, val_loader, test_loader, ckpt_root, user2seq)
    else:
        return trainer(args, model, encoder, train_loader, val_loader, test_loader, ckpt_root, user2seq)
