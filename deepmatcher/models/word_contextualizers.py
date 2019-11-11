import six

import deepmatcher as dm
import torch
import torch.nn as nn
from torch.autograd import Variable

from . import _utils
from ..batch import AttrTensor


class RNN(dm.modules.RNN, dm.WordContextualizer):
    """Multi layered RNN based Word Contextualizer.

    Supports dropout and residual / highway connections. Takes the same parameters as the
    :class:`~deepmatcher.modules.RNN` module.
    """
    pass


# class CNN(dm.WordContextualizer):
#     pass


class SelfAttention(dm.WordContextualizer):
    """__init__(heads=1, hidden_size=None, input_dropout=0, alignment_network='decomposable', scale=False, score_dropout=0, value_transform_network=None, value_merge='concat', transform_dropout=0, output_transform_network=None, output_dropout=0, bypass_network='highway', input_size=None)

    Self Attention based Word Contextualizer.

    Supports `vanilla self attention <https://arxiv.org/abs/1606.01933>`__ and `multi-head
    self attention <https://arxiv.org/abs/1706.03762>`__.

    Args:
        heads (int):
            Number of attention heads to use. Defaults to 1.
        hidden_size (int):
            The default hidden size of the `alignment_network` and transform networks, if
            they are not disabled.
        input_dropout (float):
            If non-zero, applies dropout to the input to this module. Dropout probability
            must be between 0 and 1.
        alignment_network (string or :class:`deepmatcher.modules.AlignmentNetwork` or callable):
            The neural network takes the input sequence, aligns the words in the sequence
            with other words in the sequence, and returns the corresponding alignment
            score matrix. Argument must specify a :ref:`align-op` operation.
        scale (bool):
            Whether to scale the alignment scores by the square root of the
            `hidden_size` parameter. Based on `scaled dot-product attention
            <https://arxiv.org/abs/1706.03762>`__
        score_dropout (float):
            If non-zero, applies dropout to the alignment score matrix. Dropout
            probability must be between 0 and 1.
        value_transform_network (string or :class:`~deepmatcher.modules.Transform` or callable):
            For each word embedding in the input sequence, SelfAttention takes a weighted
            average of the aligning values, i.e., the aligning word embeddings based on
            the alignment scores. This parameter specifies the neural network to transform
            the values (word embeddings) before taking the weighted average. Argument must
            be None or specify a :ref:`transform-op` operation. If the argument is a
            string, the hidden size of the transform operation is computed as
            :code:`hidden_size // heads`. If argument is None, and `heads` is 1, then the
            values are not transformed. If argument is None and `heads` is > 1, then a 1
            layer highway network without any non-linearity is used. The hidden size for
            this is computed as mentioned above.
        value_merge (string or :class:`~deepmatcher.modules.Merge` or callable):
            For each word embedding in the input sequence, each SelfAttention head
            produces one corresponding vector as output. This parameter specifies how
            to merge the outputs of all attention heads for each word embedding.
            Concatenates the outputs of all heads by default. Argument must specify a
            :ref:`merge-op` operation.
        transform_dropout (float):
            If non-zero, applies dropout to the output of the `value_transform_network`,
            if applicable. Dropout probability must be between 0 and 1.
        output_transform_network (string or :class:`~deepmatcher.modules.Transform` or callable):
            For each word embedding in the input sequence, SelfAttention produces one
            corresponding vector as output. This neural network specifies how to transform
            each of these output vectors to obtain a hidden representation of size
            `hidden_size`. Argument must be None or specify a :ref:`transform-op`
            operation. If argument is None, and `heads` is 1, then the output vectors are
            not transformed. If argument is None and `heads` is > 1, then a 1 layer
            highway network without any non-linearity is used.
        output_dropout (float):
            If non-zero, applies dropout to the output of the `output_transform_network`,
            if applicable. Dropout probability must be between 0 and 1.
        bypass_network (string or :class:`Bypass` or callable):
            The bypass network (e.g. residual or highway network) to use. The input word
            embedding sequence to this module is considered as the raw input to the bypass
            network and the final output vector sequence (output of `value_merge` or
            `output_transform_network` if applicable) is considered as the transformed
            input. Argument must specify a :ref:`bypass-op` operation. If None, does not
            use a bypass network.
        input_size (int):
            The number of features in the input to the module. This parameter will be
            automatically specified by :class:`LazyModule`.
    """

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
              input_size=None):
        hidden_size = hidden_size if hidden_size is not None else input_size

        self.alignment_networks = nn.ModuleList()
        for head in range(heads):
            self.alignment_networks.append(
                dm.modules._alignment_module(alignment_network, hidden_size))

        if value_transform_network is None and heads > 1:
            value_transform_network = dm.modules.Transform(
                '1-layer-highway', non_linearity=None, hidden_size=hidden_size // heads)
        self.value_transform_network = dm.modules._transform_module(
            value_transform_network, hidden_size // heads)

        self.value_merge = dm.modules._merge_module(value_merge)

        self.softmax = nn.Softmax(dim=2)

        if output_transform_network is None and heads > 1:
            output_transform_network = dm.modules.Transform(
                '1-layer-highway', non_linearity=None, hidden_size=hidden_size)
        self.output_transform_network = dm.modules._transform_module(
            output_transform_network, hidden_size)

        self.input_dropout = nn.Dropout(input_dropout)
        self.transform_dropout = nn.Dropout(transform_dropout)
        self.score_dropout = nn.Dropout(output_dropout)
        self.output_dropout = nn.Dropout(output_dropout)

        self.bypass_network = dm.modules._bypass_module(bypass_network)

        self.heads = heads
        self.scale = scale
        self.hidden_size = hidden_size

    def _forward(self, input_with_meta):
        input = self.input_dropout(input_with_meta.data)

        values_aligned = []
        for head in range(self.heads):
            # Dims: batch x len1 x len2
            alignment_scores = self.score_dropout(self.alignment_networks[head](input,
                                                                                input))

            if self.scale:
                alignment_scores = alignment_scores / torch.sqrt(self.hidden_size)

            if input_with_meta.lengths is not None:
                mask = _utils.sequence_mask(input_with_meta.lengths)
                mask = mask.unsqueeze(1)  # Make it broadcastable.
                alignment_scores.data.masked_fill_(~mask, -float('inf'))

            normalized_scores = self.softmax(alignment_scores)

            if self.value_transform_network is not None:
                values_transformed = self.transform_dropout(
                    self.value_transform_network(input))
            else:
                values_transformed = input

            # Dims: batch x len1 x channels
            values_aligned.append(torch.bmm(normalized_scores, values_transformed))

        values_merged = self.value_merge(*values_aligned)

        output = values_merged
        if self.output_transform_network:
            output = self.output_transform_network(output)
        output = self.output_dropout(output)

        final_output = self.bypass_network(output, input)

        return AttrTensor.from_old_metadata(final_output, input_with_meta)
