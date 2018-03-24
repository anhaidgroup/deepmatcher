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
        return 2 * self.precision * self.recall / (self.precision + self.recall)

    def precision(self):
        return 100 * self.tps / (self.tps + self.fps)

    def recall(self):
        return 100 * self.tps / (self.tps + self.fns)

    def accuracy(self):
        return 100 * (self.tps + self.tns) / self.examples

    def examples_per_sec(self):
        return self.examples / (time.time() - self.start_time)


class Runner(object):

    @staticmethod
    def print_stats(name, epoch, batch, n_batches, stats, cum_stats):
        """Write out statistics to stdout.
        """
        print((' | {name} | [{epoch}][{batch}/{n_batches}] || F1: {f1} | Prec: {prec} |'
               ' Rec: {rec} || Cum. F1: {cf1} | Cum. Prec: {cprec} | Cum. Rec: {crec} ||'
               ' Ex/s: {bps}').format(
                   name=name,
                   epoch=epoch,
                   batch=batch,
                   n_batches=n_batches,
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
        print(('Finished Epoch {epoch} || Run Time: {runtime} | Load Time: {datatime} | '
               'F1: {f1} | Prec: {prec} |  Rec: {rec} || Ex/s: {bps}\n').format(
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
        correct = predictions == target.data
        incorrect = 1 - correct
        positives = target.data == 1
        negatives = target.data == 0

        tp = torch.dot(correct, positives)
        tn = torch.dot(correct, negatives)
        fp = torch.dot(incorrect, negatives)
        fn = torch.dot(incorrect, positives)

        return tp, tn, fp, fn

    @staticmethod
    def _run(run_type,
             model,
             dataset,
             criterion,
             optimizer,
             train,
             epoch=None,
             retain_predictions=False,
             device=None,
             save_path=None,
             batch_size=32,
             num_data_workers=2,
             batch_callback=None,
             epoch_callback=None,
             log_freq=1,
             **kwargs):

        train_iter = MatchingIterator(dataset, model.train_dataset, batch_size=batch_size, device=device)

        if device == 'cpu':
            model = model.cpu()
            criterion = criterion.cpu()
        elif torch.cuda.is_available():
            model = model.cuda()
            criterion = criterion.cuda()
        elif device == 'gpu':
            raise ValueError('No GPU available.')

        if train:
            model.train_mode()
        else:
            model.eval_mode()

        datatime = 0
        runtime = 0
        cum_stats = Statistics()
        stats = Statistics()

        epoch_str = 'epoch ' + str(epoch) + ' :' if epoch else ':'
        print('=> ', run_type, epoch_str)
        batch_end = time.time()
        for batch_idx, batch in enumerate(train_iter):
            batch_start = time.time()
            datatime += batch_start - batch_end

            output = model(batch)

            loss = float('NaN')
            if criterion:
                loss = criterion(output, batch.label)

            scores = Runner.compute_scores(output, batch.label)
            cum_stats.update(loss, *scores)
            stats.update(loss, *scores)

            if (batch_idx + 1) % log_freq == 0:
                Runner.print_stats(run_type, epoch, batch, len(train_iter), stats,
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

        Runner.print_final_stats(epoch, runtime, datatime, cum_stats)
        return cum_stats.f1

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
              save_every=5,
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
            model.optimizer.load_state_dict(model.optimizer_state)

        if model.epoch is None:
            epochs_range = range(epochs)
        else:
            epochs_range = range(model.epoch + 1, epochs)

        if model.best_score is None:
            model.best_score or -1
        for epoch in epochs_range:
            model.epoch = epoch
            Runner._run(
                'TRAIN',
                model,
                train_dataset,
                criterion,
                optimizer,
                epoch=epoch,
                train=True,
                **kwargs)

            score = Runner._run(
                'EVAL', model, validation_dataset, epoch=epoch, train=False, **kwargs)

            model.optimizer_state = optimizer.state_dict()

            if save_prefix:
                if score > model.best_score:
                    model.best_score = score
                    save_path = save_prefix + '_best.pth'
                    model.save_state(save_path)
                if (epoch + 1) % save_every == 0:
                    save_path = '{prefix}_ep{epoch}.pth'.format(
                        prefix=save_prefix, epoch=epoch)
                    model.save_state(save_path)

    def evaluate(model, dataset, **kwargs):
        Runner._run(model, dataset, train=False, **kwargs)
