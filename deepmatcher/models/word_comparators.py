import pdb

import six

import deepmatcher as dm
import torch
import torch.nn as nn
from torch.autograd import Variable

from . import _utils
from ..batch import AttrTensor


class Attention(dm.WordComparator):
    r"""Attention module with multi-head support.

    This module does the following:

    * Computes an alignment matrix between the primary input and the context input.
    """

    def _init(self,
              heads=1,
              hidden_size=None,
              raw_alignment=False,
              input_dropout=0,
              alignment_network='decomposable',
              scale=False,
              score_dropout=0,
              input_transform_network=None,
              value_transform_network=None,
              value_merge='concat',
              transform_dropout=0,
              comparison_merge='concat',
              comparison_network='2-layer-highway',
              input_size=None):
        hidden_size = hidden_size if hidden_size is not None else input_size[0]

        self.alignment_network = dm.modules._alignment_module(alignment_network,
                                                              hidden_size)

        if value_transform_network is None and heads > 1:
            value_transform_network = 'linear'
        self.value_transform_network = dm.modules._transform_module(
            value_transform_network, hidden_size // heads)

        if input_transform_network is None:
            self.input_transform_network = self.value_transform_network
        else:
            self.input_transform_network = dm.modules._transform_module(
                input_transform_network, hidden_size // heads)

        self.value_merge = dm.modules._merge_module(value_merge)
        self.comparison_merge = dm.modules._merge_module(comparison_merge)
        self.comparison_network = dm.modules._transform_module(comparison_network,
                                                               hidden_size)

        self.input_dropout = nn.Dropout(input_dropout)
        self.transform_dropout = nn.Dropout(transform_dropout)
        self.score_dropout = nn.Dropout(score_dropout)
        self.softmax = nn.Softmax(dim=2)

        self.raw_alignment = raw_alignment
        self.heads = heads
        self.scale = scale

    def _forward(self,
                 input_with_meta,
                 context_with_meta,
                 raw_input_with_meta=None,
                 raw_context_with_meta=None):
        input = self.input_dropout(input_with_meta.data)
        context = self.input_dropout(context_with_meta.data)
        raw_input = self.input_dropout(raw_input_with_meta.data)
        raw_context = self.input_dropout(raw_context_with_meta.data)

        queries = input
        keys = context
        values = context
        if self.raw_alignment:
            queries = raw_input
            keys = raw_context

        inputs_transformed = []
        values_aligned = []
        for _ in range(self.heads):
            # Dims: batch x len1 x len2
            alignment_scores = self.score_dropout(self.alignment_network(queries, keys))
            if self.scale:
                alignment_scores = alignment_scores / torch.sqrt(queries.size(2))

            if context_with_meta.lengths is not None:
                mask = _utils.sequence_mask(context_with_meta.lengths)
                mask = mask.unsqueeze(1)  # Make it broadcastable.
                alignment_scores.data.masked_fill_(1 - mask, -float('inf'))

            # Make values along dim 2 sum to 1.
            normalized_scores = self.softmax(alignment_scores)

            if self.input_transform_network is not None:
                inputs_transformed.append(
                    self.transform_dropout(self.input_transform_network(input)))
            if self.value_transform_network is not None:
                values_transformed = self.transform_dropout(
                    self.value_transform_network(values))
            else:
                values_transformed = values

            # Dims: batch x len1 x channels
            values_aligned.append(torch.bmm(normalized_scores, values_transformed))

        inputs_merged = input
        if inputs_transformed:
            inputs_merged = self.value_merge(*inputs_transformed)
        values_merged = self.value_merge(*values_aligned)

        comparison_merged = self.comparison_merge(inputs_merged, values_merged)
        comparison = self.comparison_network(comparison_merged)

        return AttrTensor.from_old_metadata(comparison, input_with_meta)
