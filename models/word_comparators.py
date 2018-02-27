import deepmatcher as dm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from . import _utils


class Attention(dm.WordComparator):

    def _init(
            self,
            heads=1,
            input_size=None,
            hidden_size=None,
            raw_alignment=False,
            scale=False,
            alignment_network=None,
            value_transform=False,
            comparison_network=None,
            comparison_merge='concat',
    ):
        hidden_size = hidden_size if hidden_size is not None else input_size

        if alignment_network is None:
            self.alignment_network = dm.nn.AlignmentNetwork(
                type='decomposable', input_size=input_size, hidden_size=hidden_size)
        else:
            self.alignment_network = alignment_network(input_size=input_size)
        dm.utils.assert_signature('[AxBxC, AxDxC] -> [AxBxD]', self.alignment_network)

        if value_transform_network:
            value_size = hidden_size
        else:
            value_size = input_size

        self.value_transform_network = None
        if value_transform_network is not None:
            self.value_transform_network = value_transform_network(input_size=input_size)
            dm.utils.assert_signature('[AxBxC] -> [AxBxD]', self.value_transform_network)

        if comparison_network is None:
            concat_layer = dm.nn.Lambda(lambda x, y: torch.cat(x, y))
            self.comparison_network = nn.Sequential(concat_layer,
                                                    dm.nn.NonLinearTransform(
                                                        type='decomposable',
                                                        input_size=value_size * 2,
                                                        hidden_size=hidden_size))
        else:
            self.comparison_network = comparison_network(input_size=input_size)
        dm.utils.assert_signature('[AxBxC, AxDxC] -> [AxBxD]', self.comparison_network)

        self.softmax = nn.Softmax(dim=2)

    def forward(self, input, context, raw_input=None, raw_context=None):
        queries = input
        keys = context
        values = context
        if self.args.raw_alignment:
            queries = raw_input
            keys = raw_context

        # Dims: batch x len1 x len2
        alignment_scores = self.alignment_network(queries.data, keys.data)

        if context.lengths is not None:
            mask = Variable(_utils.sequence_mask(context.lengths))
            mask = mask.unsqueeze(1)  # Make it broadcastable.
            alignment_scores.masked_fill_(1 - mask, -float('inf'))

        normalized_scores = self.softmax(alignment_scores)

        if self.value_transform_network is not None:
            input = self.value_transform_network(input)
            values = self.value_transform_network(values)

        # Dims: batch x len1 x channels
        aligned = torch.bmm(normalized_scores, values)
        comparison = self.comparison_network(input, aligned)

        return output


class MultiHeadAttention(dm.WordComparator):

    def _init(self,
              input_dims,
              heads=2,
              scale=False,
              values_merge='concat',
              query_key_dims='auto',
              value_dims='auto'):
        if isinstance(query_key_dims, six.integer_types):
            self.query_key_dims = query_key_dims
        elif query_key_dims == 'auto':
            self.query_key_dims = input_dims // heads
        elif query_key_dims is None:
            self.query_key_dims = None
        else:
            raise utils.bad_arg('query_key_dims')

        if isinstance(value_dims, six.integer_types):
            self.value_dims = value_dims
        elif value_dims == 'auto':
            self.value_dims = input_dims // heads
        elif value_dims is None:
            self.value_dims = None
        else:
            raise utils.bad_arg('value_dims')
