from __future__ import division

import abc
import math
import pdb

import six

import deepmatcher as dm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from . import _utils
from ..data import AttrTensor


@six.add_metaclass(abc.ABCMeta)
class LazyModule(nn.Module):

    def __init__(self, *args, **kwargs):
        super(LazyModule, self).__init__()
        self._init_args = args
        self._init_kwargs = kwargs
        self._initialized = False
        self._fns = []
        self.signature = None

    def forward(self, input, *args, **kwargs):
        if not self._initialized:
            try:
                self._init(
                    *self._init_args,
                    input_size=self._get_input_size(input, *args, **kwargs),
                    **self._init_kwargs)
            except TypeError:
                self._init(*self._init_args, **self._init_kwargs)
            for fn in self._fns:
                super(LazyModule, self)._apply(fn)

            if self.signature is not None:
                self._verify_signature(input, *args)

            if dm._check_nan:
                self.register_forward_hook(LazyModule._check_nan_hook)
                self.register_backward_hook(LazyModule._check_nan_hook)

            self._initialized = True

        return self._forward(input, *args, **kwargs)

    def expect_signature(self, signature):
        self.signature = signature

    def _verify_signature(self, *args):
        # TODO: Implement this.
        return True

    def _get_input_size(self, input, *args, **kwargs):
        if isinstance(input, AttrTensor):
            return input.data.size(-1)
        elif isinstance(input, torch.Tensor):
            return input.size(-1)
        else:
            return None

    def _apply(self, fn):
        self._fns.append(fn)

    @staticmethod
    def _check_nan_hook(m, *tensors):
        _utils.check_nan(*tensors)

    def _init(self):
        pass

    @abc.abstractmethod
    def _forward(self):
        pass


class NoMeta(nn.Module):

    def __init__(self, module):
        self.module = module

    def forward(self, *args):
        module_args = []
        for arg in args:
            module_args.append(arg.data if isinstance(arg, AttrTensor) else arg)

        result = self.module(*module_args)
        if isinstance(arg[0], AttrTensor):
            return AttrTensor.from_old_metadata(result, input)
        return result


class ModuleMap(nn.Module):

    def __getitem__(self, name):
        return getattr(self, name)

    def __setitem__(self, name, module):
        setattr(self, name, module)

    def __delitem__(self, name):
        delattr(self, name)


class LazyModuleFn(LazyModule):

    def __init__(self, fn):
        self.fn = fn
        self.module = None
        self._initialized = False

    def forward(self, input, *args, **kwargs):
        if not self._initialized:
            input_size = self.get_input_size(input, *args, **kwargs)
            try:
                self.module = self.fn(input_size)
            except TypeError:
                self.module = self.fn()
            self._initialized = True
        return self.module.forward(input, *args, **kwargs)


class RNN(LazyModule):
    supported_styles = ['RNN', 'GRU', 'LSTM']

    def _init(self,
              unit_type='gru',
              hidden_size=None,
              extra_config=None,
              layers=1,
              bidirectional=True,
              dropout=0,
              input_dropout=0,
              last_layer_dropout=0,
              bypass_network=None,
              connect_num_layers=1,
              disable_packing=True,
              input_size=None):
        hidden_size = input_size if hidden_size is None else hidden_size
        last_layer_dropout = dropout if last_layer_dropout is None else last_layer_dropout

        if bypass_network is not None:
            assert layers % connect_num_layers == 0
            rnn_groups = layers // connect_num_layers
            layers_per_group = connect_num_layers
        else:
            rnn_groups = 1
            layers_per_group = layers

        bad_args = [
            'input_size', 'input_size', 'num_layers', 'batch_first', 'dropout',
            'bidirectional'
        ]
        assert not any([a in extra_config for a in bad_args])

        self.rnn_groups = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.bypass_networks = nn.ModuleList()
        self.input_dropout = NoMeta(nn.Dropout(input_dropout))

        rnn_in_size = input_size
        for g in range(rnn_groups):
            self.rnn_groups.append(
                self._get_rnn_module(
                    unit_type,
                    input_size=rnn_in_size,
                    hidden_size=hidden_size,
                    num_layers=layers_per_group,
                    batch_first=True,
                    dropout=dropout,
                    bidirectional=bidirectional,
                    **extra_config))
            if g != rnn_groups:
                self.dropouts.append(NoMeta(nn.Dropout(dropout)))
            else:
                self.dropouts.append(NoMeta(nn.Dropout(last_layer_dropout)))
            self.bypass_networks.append(NoMeta(_bypass_module(bypass_network)))
            rnn_in_size = hidden_size

    def _forward(self, input):
        input = self.input_dropout(input)

        rnn_input = input
        # if input.lengths is not None and _utils.rnn_supports_packed(self.args.unit_type):
        #     rnn_input = nn.utils.rnn.pad_packed_sequence(input.data, input.lengths)

        output = rnn_input

        for rnn, dropout, bypass in zip(self.rnn_groups, self.dropouts,
                                        self.bypass_networks):
            new_output = dropout(rnn(output))
            if bypass:
                new_output = bypass(new_output, output)
            output = new_output

        return output

    def _get_rnn_module(self, unit_type, *args, **kwargs):
        return getattr(nn, unit_type)(*args, **kwargs)


class AlignmentNetwork(LazyModule):
    supported_styles = ['dot', 'bilinear', 'decomposable', 'concat', 'concat_dot']

    def _init(self,
              style='decomposable',
              hidden_size=None,
              transform_network='multilayer',
              input_size=None):
        if style in ['bilinear' or 'decomposable']:
            if style == 'bilinear':
                assert hidden_size is None or hidden_size == input_size
            self.transform = _transform_module(
                transform_network, hidden_size=hidden_size, output_size=hidden_size)
        elif style in ['concat' or 'concat_dot']:
            output_size = 1 if style == 'concat' else hidden_size
            self.input_transform = _transform_module(
                transform_network, hidden_size=hidden_size, output_size=output_size)
            self.context_transform = _transform_module(
                transform_network, hidden_size=hidden_size, output_size=output_size)
            if style == 'concat_dot':
                self.output_transform = Transform(output_size=1)

        self.style = style

    def _forward(self, input, context):
        if self.style == 'dot':
            return torch.bmm(
                input,  # batch x len1 x input_size
                context.transpose(1, 2))  # batch x ch x input_size
        elif self.style == 'bilinear':
            return torch.bmm(
                input,  # batch x len1 x input_size
                self.transform(context).transpose(1, 2))  # batch x input_size x len2
        elif self.style == 'decomposable':
            return torch.bmm(
                self.transform(input),  # batch x hidden_size x len2
                self.transform(context).transpose(1, 2))  # batch x hidden_size x len2
        elif self.style in ['concat', 'concat_dot']:
            # batch x len1 x 1 x output_size
            input_transformed = self.input_transform(input).unsqueeze(2)

            # batch x 1 x len2 x output_size
            context_transformed = self.context_transform(input).unsqueeze(1)

            # batch x len1 x len2 x output_size
            pairwise_transformed = input_transformed + context_transformed

            if self.style == 'concat':
                # batch x len1 x len2
                return pairwise_transformed.squeeze(3)

            # batch x len1 x len2
            return self.output_transform(pairwise_transformed).squeeze(3)


class Lambda(LazyModule):

    def __init__(self, lambd):
        super(Lambda, self).__init__()
        self.lambd = lambd

    def forward(self, *args):
        return self.lambd(*args)


class Pool(LazyModule):
    supported_styles = [
        'avg', 'divsqrt', 'inv-freq-avg', 'sif', 'max', 'last', 'birnn-last'
    ]

    def _init(self, style, alpha=0.0001):
        assert style in Pool.supported_styles
        self.style = style
        self.alpha = alpha

    def _forward(self, input_with_meta):
        input = input_with_meta.data

        if self.style == 'last':
            lengths = input_with_meta.lengths.type_as(input)
            lengths = lengths.expand(-1, 1, 1).repeat(input.size(2), 2)
            return input.gather(1, lengths).squeeze(1)
        elif self.style == 'birnn-last':
            hsize = input.size(2) // 2
            lengths = input_with_meta.lengths.type_as(input)
            lengths = lengths.expand(-1, 1, 1).repeat(hsize, 2)
            print(lengths.size())
            pdb.set_trace()

            forward_last = input.gather(1, lengths).squeeze(1)
            print(forward_last.size())
            pdb.set_trace()

            backward_last = input[:, 0, hsize:]
            return torch.cat((forward_last, backward_last), 1)
        elif self.style == 'max':
            if input_with_meta.lengths is not None:
                mask = Variable(_utils.sequence_mask(input_with_meta.lengths))
                mask = mask.unsqueeze(2)  # Make it broadcastable.
                input.masked_fill_(1 - mask, -float('inf'))
        else:
            if input_with_meta.lengths is not None:
                mask = Variable(_utils.sequence_mask(input_with_meta.lengths))
                mask = mask.unsqueeze(2)  # Make it broadcastable.
                input.masked_fill_(1 - mask, 0)
                print('inspect input')
                pdb.set_trace()

            lengths = input_with_meta.lengths.clamp(min=1).type_as(input).unsqueeze(1)
            if self.style == 'avg':
                return input.sum(1) / lengths
            elif self.style == 'divsqrt':
                return input.sum(1) / lengths.sqrt()
            elif self.style == 'inv-freq-avg':
                if not isinstance(self.alpha, torch.Tensor):
                    self.alpha = input.new([self.alpha])
                inv_probs = self.alpha / (input_with_meta.probs + self.alpha)
                weighted = input * inv_probs.unsqueeze(2)
                return weighted.sum(1) / lengths
            elif self.style == 'sif':
                if not isinstance(self.alpha, torch.Tensor):
                    self.alpha = input.new([self.alpha])
                inv_probs = self.alpha / (input_with_meta.probs + self.alpha)
                weighted = input * inv_probs.unsqueeze(2)
                return (weighted.sum(1) / lengths) - input_with_meta.pc
            else:
                raise NotImplementedError(self.style + ' is not implemented.')


class Merge(LazyModule):

    _style_map = {
        'concat': lambda *args: torch.cat(args, args[0].dim() - 1),
        'diff': lambda x, y: torch.sub(x, y),
        'absdiff': lambda x, y: torch.sub(x, y).abs(),
        'mul': lambda x, y: torch.mul(x, y)
    }
    supported_styles = _style_map.keys()

    def _init(self, style):
        assert style in Merge.supported_styles
        self.op = Merge._style_map[style]

    def _forward(self, *args):
        return self.op(*args)


class Bypass(LazyModule):
    supported_styles = ['residual', 'highway']

    def _init(self, style, residual_scale=True, highway_bias=-2, input_size=None):
        assert style in Bypass.supported_styles
        self.style = style
        self.residual_scale = residual_scale
        self.highway_bias = highway_bias
        self.highway_gate = None
        self.highway_bias

    def _forward(self, transformed, raw):
        assert transformed.shape[:-1] == raw.shape[:-1]

        tsize = transformed.shape[-1]
        rsize = raw.shape[-1]
        adjusted_raw = raw
        if tsize < rsize:
            padded = F.pad(raw, (0, tsize - rsize % tsize))
            adjusted_raw = padded.view(*raw.shape[:-1], -1, tsize).sum(-2)
            pdb.set_trace()
        elif tsize > rsize:
            multiples = math.ceil(tsize / rsize)
            adjusted_raw = raw.repeat([1] * (raw.dim() - 1), multiples).narrow(
                -1, 0, tsize)

        if self.style == 'residual':
            res = transformed + adjusted_raw
            if self.residual_scale:
                res *= math.sqrt(0.5)
            return res
        elif self.style == 'highway':
            if self.highway_gate is None:
                self.highway_gate = nn.Linear(rsize, tsize)
            transform_gate = F.sigmoid(self.highway_gate(raw) + self.highway_bias)
            carry_gate = 1 - transform_gate
            return transform_gate * transformed + carry_gate * adjusted_raw


class Transform(LazyModule):
    supported_nonlinearities = [
        'sigmoid', 'tanh', 'relu', 'elu', 'selu', 'glu', 'leaky_relu'
    ]

    def _init(self,
              style,
              layers=1,
              bypass_network=None,
              non_linearity='leaky_relu',
              hidden_size=None,
              input_size=None,
              output_size=None):
        hidden_size = hidden_size or input_size
        output_size = output_size or hidden_size

        parts = style.split('-')

        if 'layer' in parts:
            layers = int(parts[parts.index('layer') - 1])

        for bstyle in Bypass.supported_styles:
            if bstyle in parts:
                bypass_network = bstyle
                break

        for nlstyle in Transform.supported_nonlinearities:
            if nlstyle in parts:
                non_linearity = nlstyle
                break

        self.transforms = nn.ModuleList()
        self.bypass_networks = nn.ModuleList()
        self.non_linearity = non_linearity
        assert (non_linearity is None or
                non_linearity in Transform.supported_nonlinearities)

        transform_in_size = input_size
        transform_out_size = hidden_size
        for layer in range(layers):
            if layer == layers - 1:
                transform_out_size = output_size
            self.transforms.append(nn.Linear(transform_in_size, transform_out_size))
            self.bypass_networks.append(_bypass_module(bypass_network))

    def _forward(self, input):
        output = input

        for transform, bypass in zip(self.transforms, self.bypass_networks):
            new_output = transform(output)
            if self.non_linearity:
                new_output = getattr(F, self.non_linearity)
            if bypass:
                new_output = bypass(new_output, output)
            output = new_output

        return output


class Identity(LazyModule):
    def _forward(self, *args):
        return args


def _merge_module(op):
    module = _utils.get_module(Merge, op)
    if module:
        module.expect_signature('[AxB, AxB] -> [AxC]')
    return module


def _bypass_module(op):
    module = _utils.get_module(Bypass, op)
    if module:
        module.expect_signature('[AxB, AxC] -> [AxB]')
    return module


def _transform_module(op, hidden_size, output_size=None):
    module = _utils.get_module(
        Transform, op, hidden_size=hidden_size, output_size=output_size)
    if module:
        module.expect_signature('[AxB] -> [AxC]')
        module.expect_signature('[AxBxC] -> [AxBxD]')
    return module


def _alignment_module(op, hidden_size):
    module = _utils.get_module(
        AlignmentNetwork, op, hidden_size=hidden_size, required=True)
    module.expect_signature('[AxBxC, AxDxC] -> [AxBxD]')
    return module
