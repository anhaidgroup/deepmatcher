import six

import deepmatcher as dm
import torch
import torch.nn as nn
from torch.autograd import Variable

from . import _utils
from ..batch import AttrTensor


class RNN(dm.modules.RNN, dm.WordContextualizer):
    pass


# class CNN(dm.WordContextualizer):
#     pass


class SelfAttention(dm.WordContextualizer):

    def _init(self,
              heads=1,
              hidden_size=None,
              input_dropout=0,
              alignment_network='decomposable',
              scale=False,
              score_dropout=0,
              value_transform_network=None,
              value_merge='concat',
              transform_dropout=0,
              output_transform_network=None,
              output_dropout=0,
              bypass_network='highway',
              normalization=None,
              input_size=None):
        hidden_size = hidden_size if hidden_size is not None else input_size[0]

        self.alignment_network = dm.modules._alignment_module(
            alignment_network, hidden_size=hidden_size)

        if value_transform_network is None and heads > 1:
            value_transform_network = 'linear'
        self.value_transform_network = dm.modules._transform_module(
            value_transform_network, hidden_size=hidden_size // heads)

        self.value_merge = dm.modules._merge_module(value_merge)

        self.softmax = dm.modules.Softmax(dim=2)

        if output_transform_network is None and heads > 1:
            output_transform_network = 'linear'
        self.output_transform_network = dm.modules._transform_module(
            output_transform_network, hidden_size=hidden_size)

        self.input_dropout = nn.Dropout(input_dropout)
        self.transform_dropout = nn.Dropout(transform_dropout)
        self.score_dropout = nn.Dropout(output_dropout)
        self.output_dropout = nn.Dropout(output_dropout)

        self.bypass_network = dm.modules._bypass_module(bypass_network)

        if normalization:
            raise NotImplementedError()

        self.heads = heads
        self.scale = scale

    def _forward(self, input_with_meta):
        input = self.input_dropout(input_with_meta.data)

        values_aligned = []
        for _ in range(self.heads):
            # Dims: batch x len1 x len2
            alignment_scores = self.score_dropout(self.alignment_network(input, input))

            if self.scale:
                alignment_scores = alignment_scores / torch.sqrt(input.size(2))

            if input_with_meta.lengths is not None:
                mask = _utils.sequence_mask(input_with_meta.lengths)
                mask = mask.unsqueeze(1)  # Make it broadcastable.
                alignment_scores.data.masked_fill_(1 - mask, -float('inf'))

            normalized_scores = self.softmax(alignment_scores)

            if self.value_transform_network is not None:
                values_transformed = self.transform_dropout(
                    self.value_transform_network(input))

            # Dims: batch x len1 x channels
            values_aligned.append(torch.bmm(normalized_scores, values_transformed))

        values_merged = self.value_merge(*values_aligned)

        output = values_merged
        if self.output_transform_network:
            output = self.output_transform_network(output)
        output = self.output_dropout(output)

        final_output = self.bypass_network(output, input)
        if self.normalization_network:
            final_output = self.normalization_network(final_output)

        return AttrTensor.from_old_metadata(final_output, input_with_meta)
