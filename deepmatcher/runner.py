import logging
import os
import pdb
import time

import torch

from . import optim as optim
from .data import MatchingIterator
from .loss import SoftNLLLoss

logger = logging.getLogger(__name__)


class Statistics(object):
    """
    Accumulator for loss statistics, inspired by ONMT.
    Currently calculates:
    * F1
    * Precision
    * Recall
    * Accuracy
    """

    def __init__(self):
        self.loss_sum = 0
        self.examples = 0
        self.tps = 0
        self.tns = 0
        self.fps = 0
        self.fns = 0
        self.start_time = time.time()

    def update(self, loss=0, tps=0, tns=0, fps=0, fns=0):
        examples = tps + tns + fps + fns
        self.loss_sum += loss * examples
        self.tps += tps
        self.tns += tns
        self.fps += fps
        self.fns += fns
        self.examples += examples

    def loss(self):
        return self.loss_sum / self.examples

    def f1(self):
        prec = self.precision()
        recall = self.recall()
        return 2 * prec * recall / max(prec + recall, 1)

    def precision(self):
        return 100 * self.tps / max(self.tps + self.fps, 1)

    def recall(self):
        return 100 * self.tps / max(self.tps + self.fns, 1)

    def accuracy(self):
        return 100 * (self.tps + self.tns) / self.examples

    def examples_per_sec(self):
        return self.examples / (time.time() - self.start_time)


class Runner(object):

    @staticmethod
    def print_stats(name, epoch, batch, n_batches, stats, cum_stats):
        """Write out statistics to stdout.
        """
        print((' | {name} | [{epoch}][{batch:4d}/{n_batches}] || Loss: {loss:7.4f} |'
               ' F1: {f1:7.2f} | Prec: {prec:7.2f} | Rec: {rec:7.2f} ||'
               ' Cum. F1: {cf1:7.2f} | Cum. Prec: {cprec:7.2f} | Cum. Rec: {crec:7.2f} ||'
               ' Ex/s: {eps:6.1f}').format(
                   name=name,
                   epoch=epoch,
                   batch=batch,
                   n_batches=n_batches,
                   loss=stats.loss(),
                   f1=stats.f1(),
                   prec=stats.precision(),
                   rec=stats.recall(),
                   cf1=cum_stats.f1(),
                   cprec=cum_stats.precision(),
                   crec=cum_stats.recall(),
                   eps=cum_stats.examples_per_sec()))

    @staticmethod
    def print_final_stats(epoch, runtime, datatime, stats):
        """Write out statistics to stdout.
        """
        print(('Finished Epoch {epoch} || Run Time: {runtime:7f} | '
               'Load Time: {datatime:7f} | F1: {f1:7.2f} | Prec: {prec:7.2f} | '
               'Rec: {rec:7.2f} || Ex/s: {eps:6.1f}\n').format(
                   epoch=epoch,
                   runtime=runtime,
                   datatime=datatime,
                   f1=stats.f1(),
                   prec=stats.precision(),
                   rec=stats.recall(),
                   eps=stats.examples_per_sec()))

    @staticmethod
    def compute_scores(output, target):
        predictions = output.max(1)[1].data
        correct = (predictions == target.data).float()
        incorrect = (1 - correct).float()
        positives = (target.data == 1).float()
        negatives = (target.data == 0).float()

        tp = torch.dot(correct, positives)
        tn = torch.dot(correct, negatives)
        fp = torch.dot(incorrect, negatives)
        fn = torch.dot(incorrect, positives)

        return tp, tn, fp, fn

    @staticmethod
    def tally_parameters(model):
        n_params = sum([p.nelement() for p in model.parameters() if p.requires_grad])
        print('* Number of trainable parameters:', n_params)

    @staticmethod
    def _run(run_type,
             model,
             dataset,
             criterion=None,
             optimizer=None,
             train=False,
             retain_predictions=False,
             device=None,
             save_path=None,
             batch_size=32,
             num_data_workers=2,
             batch_callback=None,
             epoch_callback=None,
             log_freq=5,
             sort_in_buckets=None,
             **kwargs):

        sort_in_buckets = train
        train_iter = MatchingIterator(
            dataset,
            model.train_dataset,
            batch_size=batch_size,
            device=device,
            sort_in_buckets=sort_in_buckets)

        if device == 'cpu':
            model = model.cpu()
            if criterion:
                criterion = criterion.cpu()
        elif torch.cuda.is_available():
            model = model.cuda()
            if criterion:
                criterion = criterion.cuda()
        elif device == 'gpu':
            raise ValueError('No GPU available.')

        if train:
            model.train()
        else:
            model.eval()

        epoch = model.epoch
        datatime = 0
        runtime = 0
        cum_stats = Statistics()
        stats = Statistics()

        epoch_str = 'epoch ' + str(epoch + 1) + ' :'
        print('=> ', run_type, epoch_str)
        batch_end = time.time()
        for batch_idx, batch in enumerate(train_iter):
            batch_start = time.time()
            datatime += batch_start - batch_end

            output = model(batch)
            if epoch == 0 and batch_idx == 0 and train:
                Runner.tally_parameters(model)

            loss = float('NaN')
            if criterion:
                loss = criterion(output, batch.label)

            scores = Runner.compute_scores(output, batch.label)
            cum_stats.update(float(loss), *scores)
            stats.update(float(loss), *scores)

            if (batch_idx + 1) % log_freq == 0:
                Runner.print_stats(run_type, epoch + 1, batch_idx + 1, len(train_iter), stats,
                                   cum_stats)
                stats = Statistics()

            if train:
                model.zero_grad()
                loss.backward()

                if not optimizer.params:
                    optimizer.set_parameters(model.named_parameters())
                optimizer.step()

            batch_end = time.time()
            runtime += batch_end - batch_start

        Runner.print_final_stats(epoch + 1, runtime, datatime, cum_stats)
        return cum_stats.f1()

    @staticmethod
    def train(model,
              train_dataset,
              validation_dataset,
              epochs=50,
              criterion=None,
              optimizer=None,
              pos_weight=1,
              label_smoothing=False,
              save_prefix=None,
              save_every=None,
              **kwargs):
        model.initialize(train_dataset)

        model.register_train_buffer('optimizer_state')
        model.register_train_buffer('best_score')
        model.register_train_buffer('epoch')

        if criterion is None:
            assert pos_weight < 2
            neg_weight = 2 - pos_weight

            criterion = SoftNLLLoss(label_smoothing,
                                    torch.Tensor([neg_weight, pos_weight]))

        optimizer = optimizer or optim.Optimizer()
        if model.optimizer_state is not None:
            model.optimizer.base_optimizer.load_state_dict(model.optimizer_state)

        if model.epoch is None:
            epochs_range = range(epochs)
        else:
            epochs_range = range(model.epoch + 1, epochs)

        if model.best_score is None:
            model.best_score = -1
        optimizer.last_acc = model.best_score

        for epoch in epochs_range:
            model.epoch = epoch
            Runner._run(
                'TRAIN',
                model,
                train_dataset,
                criterion,
                optimizer,
                train=True,
                **kwargs)

            score = Runner._run('EVAL', model, validation_dataset, train=False, **kwargs)

            optimizer.update_learning_rate(score, epoch)
            model.optimizer_state = optimizer.base_optimizer.state_dict()

            new_best_found = False
            if score > model.best_score:
                print('* Best F1:', score)
                model.best_score = score
                new_best_found = True

            if save_prefix and new_best_found:
                print('Saving best model...')
                save_path = save_prefix + '_best.pth'
                model.save_state(save_path)

                if save_every is not None and (epoch + 1) % save_every == 0:
                    save_path = '{prefix}_ep{epoch}.pth'.format(
                        prefix=save_prefix, epoch=epoch)
                    model.save_state(save_path)

    def eval(model, dataset, **kwargs):
        Runner._run('EVAL', model, dataset, train=False, **kwargs)
