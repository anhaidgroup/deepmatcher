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
              score_dropout=0,
              scale=False,
              input_context_comparison_network='1-layer-highway',
              input_transform_network=None,
              transform_dropout=0,
              input_size=None):

        # self.alignment_network = dm.modules._alignment_module(
        #     alignment_network, hidden_size=hidden_size)

        assert rnn is not None
        self.rnn = _utils.get_module(dm.modules.RNN, rnn, hidden_size=hidden_size)
        self.rnn.expect_signature('[AxBxC] -> [AxBx{D}]'.format(D=hidden_size))

        self.rnn_pool = dm.modules.Pool(rnn_pool_style)

        self.input_context_comparison_network = dm.modules._transform_module(
            input_context_comparison_network, hidden_size=hidden_size)
        self.scoring_network = dm.modules._transform_module('1-layer', hidden_size=1)
        self.input_transform_network = dm.modules._transform_module(
            input_transform_network, hidden_size=hidden_size)

        self.input_dropout = nn.Dropout(input_dropout)
        self.transform_dropout = nn.Dropout(transform_dropout)
        self.score_dropout = nn.Dropout(score_dropout)

        self.softmax = nn.Softmax(dim=1)

        self.scale = scale

    def _forward(self, input_with_meta, context_with_meta):
        input = self.input_dropout(input_with_meta.data)
        context = self.input_dropout(context_with_meta.data)

        context_rnn_output = self.rnn(
            AttrTensor.from_old_metadata(context, context_with_meta))

        # Dims: batch x 1 x hidden_size
        context_pool_output = self.rnn_pool(context_rnn_output).data.unsqueeze(1)

        # Dims: batch x len1 x hidden_size
        context_pool_repeated = context_pool_output.repeat(1, input.size(1), 1)

        # Dims: batch x len1 x (hidden_size * 2)
        concatenated = torch.cat((input, context_pool_repeated), dim=2)

        # Dims: batch x len1
        raw_scores = self.scoring_network(
            self.input_context_comparison_network(concatenated)).squeeze(2)

        alignment_scores = self.score_dropout(raw_scores)

        if self.scale:
            alignment_scores = alignment_scores / torch.sqrt(input.size(2))

        if input_with_meta.lengths is not None:
            mask = _utils.sequence_mask(input_with_meta.lengths)
            alignment_scores.data.masked_fill_(1 - mask, -float('inf'))

        # Make values along dim 2 sum to 1.
        normalized_scores = self.softmax(alignment_scores)

        transformed = input
        if self.input_transform_network is not None:
            transformed = self.transform_dropout(self.input_transform_network(input))

        weighted_sum = torch.bmm(normalized_scores.unsqueeze(1), transformed).squeeze(1)
        return AttrTensor.from_old_metadata(weighted_sum, input_with_meta)
