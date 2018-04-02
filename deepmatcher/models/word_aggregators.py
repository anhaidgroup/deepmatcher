import pdb

import six

import deepmatcher as dm
import torch
import torch.nn as nn
from torch.autograd import Variable

from . import _utils
from ..batch import AttrTensor


class Pool(dm.modules.Pool, dm.WordAggregator):
    pass


class AttentionWithRNN(dm.WordAggregator):

    def _init(self,
              hidden_size=None,
              input_dropout=0,
              rnn='gru',
              rnn_pool_style='birnn-last',
              alignment_network='concat',
              score_dropout=0,
              scale=False,
              transform_network=None,
              transform_dropout=0,
              input_size=None):

        self.alignment_network = dm.modules._alignment_module(
            alignment_network, hidden_size=hidden_size)

        assert rnn is not None
        self.rnn = _utils.get_module(dm.modules.RNN, rnn, hidden_size=hidden_size)
        self.rnn.expect_signature('[AxBxC] -> [AxBx{D}]'.format(D=hidden_size))

        self.rnn_pool = dm.modules.Pool(rnn_pool_style)

        self.transform_network = dm.modules._transform_module(
            transform_network, hidden_size=hidden_size)

        self.input_dropout = nn.Dropout(input_dropout)
        self.transform_dropout = nn.Dropout(transform_dropout)
        self.score_dropout = nn.Dropout(score_dropout)

        self.softmax = nn.Softmax(dim=2)

        self.scale = scale

    def _forward(self, input_with_meta):
        input = self.input_dropout(input_with_meta.data)

        rnn_output = self.rnn(AttrTensor.from_old_metadata(input, input_with_meta))

        # batch x 1 x hidden_size
        pool_output = self.rnn_pool(rnn_output).data.unsqueeze(1)

        alignment_scores = self.score_dropout(self.alignment_network(pool_output, input))

        if self.scale:
            alignment_scores = alignment_scores / torch.sqrt(input.size(2))

        if input_with_meta.lengths is not None:
            mask = _utils.sequence_mask(input_with_meta.lengths)
            mask = mask.unsqueeze(1)  # Make it broadcastable.
            alignment_scores.data.masked_fill_(1 - mask, -float('inf'))

        normalized_scores = self.softmax(alignment_scores)

        transformed = input
        if self.transform_network is not None:
            transformed = self.transform_dropout(self.transform_network(input))

        output = torch.bmm(normalized_scores, transformed).squeeze(1)
        return AttrTensor.from_old_metadata(output, input_with_meta)
