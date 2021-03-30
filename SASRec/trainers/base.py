from loggers import *
from utils import AverageMeterSet, STATE_DICT_KEY, OPTIMIZER_STATE_DICT_KEY

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .optmization import AdamW, get_linear_schedule_with_warmup
from .parallel import DataParallelModel
from .utils import recalls_and_ndcgs_for_ks

import json
from abc import *
from pathlib import Path


class AbstractTrainer(metaclass=ABCMeta):
    def __init__(self, args, model, train_loader, val_loader, test_loader, ckpt_root, user2seq):
        self.user2seq = user2seq
        self.args = args
        self.device = args.device
        self.model = model.to(self.device)
        self.is_parallel = args.num_gpu > 1
        if self.is_parallel:
            self.model = DataParallelModel(self.model)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = self._create_optimizer()

        if type(args.enable_lr_schedule) is str and args.enable_lr_schedule.lower() == "linear":
            t_total = len(train_loader) * args.num_epochs
            self.lr_scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

        self.num_epochs = args.num_epochs
        self.metric_ks = args.metric_ks
        self.best_metric = args.best_metric

        self.ckpt_root = ckpt_root
        self.writer, self.train_loggers, self.val_loggers, self.test_loggers = self._create_loggers()
        self.add_extra_loggers()
        self.logger_service = LoggerService(self.train_loggers, self.val_loggers, self.test_loggers)

        self.cases = {}

    @abstractmethod
    def add_extra_loggers(self):
        pass

    @abstractmethod
    def log_extra_train_info(self, log_data):
        pass

    @abstractmethod
    def log_extra_val_info(self, log_data):
        pass

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @abstractmethod
    def calculate_loss(self, batch):
        pass

    @abstractmethod
    def calculate_metrics(self, batch):
        pass

    def train(self):
        accum_iter = 0
        best_target = 0
        best_num = 0
        # self.validate_test(0, accum_iter, mode='val')
        import time
        t = time.time()
        self.validate_test(0, accum_iter, mode='test')
        print(time.time() - t)

        for epoch in range(self.num_epochs):
            if best_num >= 10:
                break
            if self.args.stop_epochs is not None:
                if epoch > self.args.stop_epochs: break
            accum_iter = self.train_one_epoch(epoch, accum_iter)
            if (epoch % self.args.verbose) == (self.args.verbose - 1):
                target = self.validate_test(epoch, accum_iter, mode='val', target=self.args.best_metric)
                self.validate_test(epoch, accum_iter, mode='test')
                best_num = best_num + 1 if target < best_target else 1
                best_target = max(target, best_target)
        self.test()
        self.logger_service.complete({'state_dict': (self._create_state_dict())})
        self.writer.close()

    def train_one_epoch(self, epoch, accum_iter):

        self.model.train()

        average_meter_set = AverageMeterSet()
        tqdm_dataloader = tqdm(self.train_loader)

        for batch_idx, batch in enumerate(tqdm_dataloader):

            batch_size = batch[0].size(0)
            batch = [x.to(self.device) for x in batch]

            self.optimizer.zero_grad()
            loss = self.calculate_loss(batch)
            
            loss.backward()
            nn.utils.clip_grad_norm(self.model.parameters(), 5.0)
                
            self.optimizer.step()
            if self.args.enable_lr_schedule:
                self.lr_scheduler.step()

            average_meter_set.update('loss', loss.item())
            tqdm_dataloader.set_description(
                'Epoch {}, loss {:.5f} '.format(epoch+1, average_meter_set['loss'].avg))

            accum_iter += batch_size

            log_data = {
                'state_dict': (self._create_state_dict()),
                'epoch': epoch+1,
                'accum_iter': accum_iter,
            }

        log_data.update(average_meter_set.averages())
        self.log_extra_train_info(log_data)
        self.logger_service.log_train(log_data)

        torch.cuda.empty_cache()

        return accum_iter

    def validate_test(self, epoch, accum_iter, mode="val", target=None):
        self.model.eval()

        average_meter_set = AverageMeterSet()

        with torch.no_grad():
            tqdm_dataloader = tqdm(self.val_loader) if mode == "val" else tqdm(self.test_loader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                batch = [x.to(self.device) for x in batch]

                metrics = self.calculate_metrics(batch, mode)

                for k, v in metrics.items():
                    average_meter_set.update(k, v)

                description_metrics = ['NDCG@%d' % k for k in self.metric_ks[:3]] +\
                                      ['Recall@%d' % k for k in self.metric_ks[:3]] + ['MRR'] + ['AUC'] + ['loss']
                description = mode.upper() + " " + ': , '.join(s + ' {:.5f}' for s in description_metrics)
                description = description.replace('NDCG', 'N').replace('Recall', 'R').replace('MRR', 'M').replace('Jaccard', 'J')
                description = description.format(*(average_meter_set[k].avg for k in description_metrics))
                tqdm_dataloader.set_description(description)

            log_data = {
                'state_dict': (self._create_state_dict()),
                'epoch': epoch+1,
                'accum_iter': accum_iter,
            }
            log_data.update(average_meter_set.averages())


        if self.args.case_study and mode == 'val':
            with open(os.path.join(self.ckpt_root, 'cases_%s.json' % self.args.dataloader_code), 'w') as f:
                json.dump(self.cases, f)
            print("save cases in: " + os.path.join(self.ckpt_root, 'cases_%s.json' % self.args.dataloader_code))
            exit()

        torch.cuda.empty_cache()

        if mode == "val":
            self.logger_service.log_val(log_data)
        else:
            self.logger_service.log_test(log_data)

        if target is not None:
            return average_meter_set[target].avg

    def test(self):
        print('Test best model with test set!')

        best_model = torch.load(os.path.join(self.ckpt_root, 'models', 'best_acc_model.pth')).get('model_state_dict')
        self.model.load_state_dict(best_model)
        self.model.eval()

        average_meter_set = AverageMeterSet()

        with torch.no_grad():
            tqdm_dataloader = tqdm(self.test_loader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                batch = [x.to(self.device) for x in batch]

                metrics = self.calculate_metrics(batch, mode='test')

                for k, v in metrics.items():
                    average_meter_set.update(k, v)
                description_metrics = ['NDCG@%d' % k for k in self.metric_ks[:3]] +\
                                      ['Recall@%d' % k for k in self.metric_ks[:3]] + ['MRR'] + ['AUC'] + ['loss']
                description = 'FINAL TEST: ' + ', '.join(s + ' {:.5f}' for s in description_metrics)
                description = description.replace('NDCG', 'N').replace('Recall', 'R').replace('MRR', 'M').replace('Jaccard', 'J')
                description = description.format(*(average_meter_set[k].avg for k in description_metrics))
                tqdm_dataloader.set_description(description)

            average_metrics = average_meter_set.averages()
            with open(os.path.join(self.ckpt_root, 'logs', 'test_metrics.json'), 'w') as f:
                json.dump(average_metrics, f, indent=4)
            print(average_metrics)

    def ensemble_test(self, root1, root2):
        print('Test ensembled best model with test set!')

        item_model = torch.load(os.path.join(root1, 'models', 'best_acc_model.pth')).get('model_state_dict')
        query_model = torch.load(os.path.join(root2, 'models', 'best_acc_model.pth')).get('model_state_dict')
        self.model.load_state_dict(item_model)
        self.item_model = self.model
        self.item_model.alpha = 1
        self.item_model.eval()

        import pickle
        self.query_model = pickle.loads(pickle.dumps(self.item_model))
        self.query_model.load_state_dict(query_model)
        self.query_model.alpha = 0
        self.query_model.eval()

        average_meter_set = AverageMeterSet()

        with torch.no_grad():
            tqdm_dataloader = tqdm(self.test_loader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                batch = [x.to(self.device) for x in batch]
                if self.args.trainer_code == 'bert':
                    users, seqs, candidates, labels, lengths = batch
                    scores_item = self.item_model(seqs, candidates=candidates)  # B x T x C
                    scores_query = self.query_model(seqs, candidates=candidates)  # B x T x C
                    scores = (scores_item + scores_query) / 2
                    # scores = scores_item
                    # scores = scores_query
                    # gather the [MASK] outputs ranking metrics
                    metrics = recalls_and_ndcgs_for_ks(scores, labels, self.metric_ks)
                else:
                    users, seqs, labels, lengths = batch
                    scores_item = self.item_model(seqs)  # B x T x C
                    scores_query = self.query_model(seqs)  # B x T x C
                    scores = (scores_item + scores_query) / 2
                    # scores = scores_item
                    # scores = scores_query
                    res = self.ranker(scores, labels, lengths)
                    metrics = {}
                    for i, k in enumerate(self.args.metric_ks):
                        metrics["NDCG@%d" % k] = res[2*i]
                        metrics["Recall@%d" % k] = res[2*i+1]
                    metrics["loss"] = res[-1]

                for k, v in metrics.items():
                    average_meter_set.update(k, v)
                description_metrics = ['NDCG@%d' % k for k in self.metric_ks[:3]] +\
                                      ['Recall@%d' % k for k in self.metric_ks[:3]]
                description = 'FINAL TEST: ' + ', '.join(s + ' {:.5f}' for s in description_metrics)
                description = description.replace('NDCG', 'N').replace('Recall', 'R')
                description = description.format(*(average_meter_set[k].avg for k in description_metrics))
                tqdm_dataloader.set_description(description)

            average_metrics = average_meter_set.averages()
            with open(os.path.join(self.ckpt_root, 'logs', 'test_metrics.json'), 'w') as f:
                json.dump(average_metrics, f, indent=4)
            print(average_metrics)

    def _create_optimizer(self):
        args = self.args
        if args.optimizer.lower() == 'adam':
            no_decay = ["bias", "norm"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": args.weight_decay,
                },
                {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
            ]
            return optim.Adam(optimizer_grouped_parameters, lr=args.lr, betas=(0.9, 0.98))
            # return optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        elif args.optimizer.lower() == 'adamw':
            no_decay = ["bias", "norm"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": args.weight_decay,
                },
                {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
            ]
            return AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)

        elif args.optimizer.lower() == 'sgd':
            return optim.SGD(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
        
        else:
            raise ValueError

    def _create_loggers(self):
        root = Path(self.ckpt_root)
        writer = SummaryWriter(root.joinpath('logs'))
        model_checkpoint = root.joinpath('models')

        train_loggers = [
            MetricGraphPrinter(writer, key='epoch', graph_name='Epoch', group_name='Train'),
            MetricGraphPrinter(writer, key='loss', graph_name='Loss', group_name='Train'),
        ]

        val_loggers = []
        for k in self.metric_ks:
            val_loggers.append(
                MetricGraphPrinter(writer, key='NDCG@%d' % k, graph_name='NDCG@%d' % k, group_name='Validation'))
            val_loggers.append(
                MetricGraphPrinter(writer, key='Recall@%d' % k, graph_name='Recall@%d' % k, group_name='Validation'))
        val_loggers.append(MetricGraphPrinter(writer, key='loss', graph_name='loss', group_name='Validation'))
        val_loggers.append(MetricGraphPrinter(writer, key='MRR', graph_name='MRR', group_name='Validation'))
        val_loggers.append(MetricGraphPrinter(writer, key='Jaccard', graph_name='Jaccard', group_name='Validation'))
        # val_loggers.append(RecentModelLogger(model_checkpoint))
        val_loggers.append(BestModelLogger(model_checkpoint, metric_key=self.best_metric))

        test_loggers = []
        for k in self.metric_ks:
            test_loggers.append(
                MetricGraphPrinter(writer, key='NDCG@%d' % k, graph_name='NDCG@%d' % k, group_name='Test'))
            test_loggers.append(
                MetricGraphPrinter(writer, key='Recall@%d' % k, graph_name='Recall@%d' % k, group_name='Test'))
        test_loggers.append(MetricGraphPrinter(writer, key='loss', graph_name='loss', group_name='Test'))
        test_loggers.append(MetricGraphPrinter(writer, key='MRR', graph_name='MRR', group_name='Test'))
        test_loggers.append(MetricGraphPrinter(writer, key='Jaccard', graph_name='Jaccard', group_name='Test'))
        
        return writer, train_loggers, val_loggers, test_loggers

    def _create_state_dict(self):
        return {
            STATE_DICT_KEY: self.model.state_dict(),
            OPTIMIZER_STATE_DICT_KEY: self.optimizer.state_dict(),
        }
