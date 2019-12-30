import pdb

import six

import deepmatcher as dm
import torch
import torch.nn as nn
from torch.autograd import Variable

from . import _utils
from ..batch import AttrTensor


class Pool(dm.modules.Pool, dm.WordAggregator):
    """Pooling based Word Aggregator.

    Takes the same parameters as the :class:`~deepmatcher.modules.Pool` module.
    """
    pass


class AttentionWithRNN(dm.WordAggregator):
    r"""__init__(hidden_size=None, input_dropout=0, rnn='gru', rnn_pool_style='birnn-last', score_dropout=0, input_context_comparison_network='1-layer-highway', value_transform_network=None, transform_dropout=0, input_size=None)

    Attention and RNN based Word Aggregator. This class can be used when the
    aggregation on the primary input also needs the information from the context.
    Specifically, the :class:`~deepmatcher.attr_summarizers.Hybrid` attribute summarizer
    uses this aggregation approach by default. This module takes a primary input sequence
    and a context input sequence and computes a single summary vector for the primary
    input sequence based on the information in the context input. To do so, it does the
    following:

    1. Applies an :ref:`rnn-op` over the context input.
    2. Uses a :ref:`pool-op` operation over the RNN to obtain a single vector summarizing
       the information in the context input.
    3. Based on this context summary vector, uses attention to score the relevance of each
       vector in the primary input sequence.
    4. Performs a weighted average of the vectors in the primary input sequence based on
       the computed scores to obtain a context dependent summary of the primary input
       sequence.

    Args:
        hidden_size (int):
            The default hidden size to use for the RNN,
            `input_context_comparison_network`, and the `value_transform_network`.
        input_dropout (float):
            If non-zero, applies dropout to the input to this module. Dropout probability
            must be between 0 and 1.
        rnn (string or :class:`~deepmatcher.modules.RNN` or callable):
            The RNN used in Step 1 described above. Argument must specify an
            :ref:`rnn-op` operation.
        rnn_pool_style (string):
            The pooling operation used in Step 2 described above. Argument must specify a
            :ref:`pool-op` operation.
        score_dropout (float):
            If non-zero, applies dropout to the attention scores computed in Step 3
            described above. Dropout probability must be between 0 and 1.
        input_context_comparison_network (string or :class:`~deepmatcher.modules.Transform` or callable):
            The neural network that takes each vector in the primary input sequence
            concatenated with the context summary vector to obtain a hidden vector
            representing the primary vector's relevance to the context input.
            Argument must specify a :ref:`transform-op` operation.
        value_transform_network (string or :class:`~deepmatcher.modules.Transform` or callable):
            The neural network to transform the primary input sequence before taking its
            weighted average in Step 4 described above. Argument must be None or specify a
            :ref:`transform-op` operation.
        transform_dropout (float):
            If non-zero, applies dropout to the output of the `value_transform_network`,
            if applicable. Dropout probability must be between 0 and 1.
        input_size (int):
            The number of features in the input to the module. This parameter will be
            automatically specified by :class:`LazyModule`.
    """

    def _init(self,
              hidden_size=None,
              input_dropout=0,
              rnn='gru',
              rnn_pool_style='birnn-last',
              score_dropout=0,
              input_context_comparison_network='1-layer-highway',
              value_transform_network=None,
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
        self.value_transform_network = dm.modules._transform_module(
            value_transform_network, hidden_size=hidden_size)

        self.input_dropout = nn.Dropout(input_dropout)
        self.transform_dropout = nn.Dropout(transform_dropout)
        self.score_dropout = nn.Dropout(score_dropout)

        self.softmax = nn.Softmax(dim=1)

    def _forward(self, input_with_meta, context_with_meta):
        r"""
        The forward function of the attention-with-RNN netowrk.

        Args:
        input_with_meta ():
            The input sequence with metadata information.
        context_with_meta ():
            The context sequence with metadata information.
        """
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

        if input_with_meta.lengths is not None:
            mask = _utils.sequence_mask(input_with_meta.lengths)
            alignment_scores.data.masked_fill_(~mask, -float('inf'))

        # Make values along dim 2 sum to 1.
        normalized_scores = self.softmax(alignment_scores)

        transformed = input
        if self.value_transform_network is not None:
            transformed = self.transform_dropout(self.value_transform_network(input))

        weighted_sum = torch.bmm(normalized_scores.unsqueeze(1), transformed).squeeze(1)
        return AttrTensor.from_old_metadata(weighted_sum, input_with_meta)
