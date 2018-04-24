from __future__ import division

import abc
import math
import numbers
import pdb

import six

import deepmatcher as dm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from . import _utils
from ..batch import AttrTensor


@six.add_metaclass(abc.ABCMeta)
class LazyModule(nn.Module):
    r"""A lazily initialized module.

    This module is an extension of `nn.Module` with the following features:
    * Constructing an instance this module does not immediately initialize it. This means
        that if the module has paramters, they will not be instantiated immediately after
        construction.
    * The module is initialized the first time `forward` is called. This allows automatic
        input size inference. Refer to description of `_init` for details.
    * Lazy initialization also means this module can be safely deep copied to create
        structural clones that do not share parameters. E.g. deep copying a LazyModule
        consisting of a 2 layer Linear NN will produce another LazyModule with 2 layer
        Linear NN that 1) do not share parameters and 2) have different weight
        initializations.
    * Signature verification is also supported.
    * NaN checks
    """

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

    def _get_input_size(self, *args, **kwargs):
        if len(args) > 1:
            return [self._get_input_size(input) for input in args]
        elif isinstance(args[0], (AttrTensor, Variable)):
            return args[0].data.size(-1)
        else:
            return None

    def _apply(self, fn):
        if not self._initialized:
            self._fns.append(fn)
        else:
            super(LazyModule, self)._apply(fn)

    @staticmethod
    def _check_nan_hook(m, *tensors):
        _utils.check_nan(*tensors)

    def _init(self):
        pass

    @abc.abstractmethod
    def _forward(self):
        pass


class NoMeta(nn.Module):
    r"""A wrapper module to allow regular modules to take AttrTensors as input.

    Performing a forward pass through this module, will perform the following:
    * If the module input is an AttrTensor, gets the data from it, and use as input.
    * Perform forward pass through contained module with the modified input.
    * Using metadata information from module input (if provided), wrap the result into an
        AttrTensor and return this instead.
    """

    def __init__(self, module):
        super(NoMeta, self).__init__()
        self.module = module

    def forward(self, *args):
        module_args = []
        for arg in args:
            module_args.append(arg.data if isinstance(arg, AttrTensor) else arg)

        result = self.module(*module_args)
        if isinstance(args[0], AttrTensor):
            return AttrTensor.from_old_metadata(result, args[0])
        return result


class ModuleMap(nn.Module):
    """Holds submodules in a map.

    Similar to :class:`torch.nn.ModuleList`, but for maps."""

    def __getitem__(self, name):
        return getattr(self, name)

    def __setitem__(self, name, module):
        setattr(self, name, module)

    def __delitem__(self, name):
        delattr(self, name)


class MultiSequential(nn.Sequential):

    def forward(self, *inputs):
        modules = list(self._modules.values())
        inputs = modules[0](*inputs)
        for module in modules[1:]:
            inputs = module(inputs)
        return inputs


class LazyModuleFn(LazyModule):
    """A Lazy Module which simply wraps the module returned by a specified function."""

    def _init(self, fn, *args, **kwargs):
        self.module = fn(*args, **kwargs)

    def _forward(self, *args, **kwargs):
        return self.module.forward(*args, **kwargs)


class RNN(LazyModule):
    """A multi layered RNN that supports dropout and residual / highway connections."""
    _supported_styles = ['rnn', 'gru', 'lstm']

    @classmethod
    def supports_style(cls, style):
        return style.lower() in cls._supported_styles

    def _init(self,
              unit_type='gru',
              hidden_size=None,
              layers=1,
              bidirectional=True,
              dropout=0,
              input_dropout=0,
              last_layer_dropout=0,
              bypass_network=None,
              connect_num_layers=1,
              disable_packing=True,
              input_size=None,
              **extra_config):
        hidden_size = input_size if hidden_size is None else hidden_size
        last_layer_dropout = dropout if last_layer_dropout is None else last_layer_dropout

        if bidirectional:
            hidden_size //= 2

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
                self.dropouts.append(nn.Dropout(dropout))
            else:
                self.dropouts.append(nn.Dropout(last_layer_dropout))
            self.bypass_networks.append(_bypass_module(bypass_network))

            if bidirectional:
                rnn_in_size = hidden_size * 2
            else:
                rnn_in_size = hidden_size

    def _forward(self, input_with_meta):
        output = self.input_dropout(input_with_meta.data)

        for rnn, dropout, bypass in zip(self.rnn_groups, self.dropouts,
                                        self.bypass_networks):
            new_output = dropout(rnn(output)[0])
            if bypass:
                new_output = bypass(new_output, output)
            output = new_output

        return AttrTensor.from_old_metadata(output, input_with_meta)

    def _get_rnn_module(self, unit_type, *args, **kwargs):
        return getattr(nn, unit_type.upper())(*args, **kwargs)


class AlignmentNetwork(LazyModule):
    """Neural network to compute alignment between two given sequences."""

    _supported_styles = ['dot', 'bilinear', 'decomposable']

    @classmethod
    def supports_style(cls, style):
        return style.lower() in cls._supported_styles

    def _init(self,
              style='decomposable',
              hidden_size=None,
              transform_network='2-layer-highway',
              input_size=None):
        if style in ['bilinear', 'decomposable']:
            if style == 'bilinear':
                assert hidden_size is None or hidden_size == input_size
            self.transform = _transform_module(transform_network, hidden_size)
        # elif style in ['concat', 'concat_dot']:
        #     self.input_transform = nn.ModuleList()
        #     self.input_transform.append(_transform_module(transform_network, hidden_size))
        #     if style == 'concat':
        #         self.input_transform.append(Transform('1-layer', output_size=1))
        #     self.context_transform = _transform_module(transform_network, hidden_size,
        #                                                output_size)
        #     if style == 'concat_dot':
        #         self.output_transform = Transform(
        #             '1-layer', non_linearity=None, output_size=1)
        elif style != 'dot':
            raise ValueError('Unknown AlignmentNetwork style')

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
        # elif self.style in ['concat', 'concat_dot']:
        #     # batch x len1 x 1 x output_size
        #     input_transformed = self.input_transform(input).unsqueeze(2)
        #
        #     # batch x 1 x len2 x output_size
        #     context_transformed = self.context_transform(context).unsqueeze(1)
        #
        #     # batch x len1 x len2 x output_size
        #     pairwise_transformed = input_transformed + context_transformed
        #
        #     if self.style == 'concat':
        #         # batch x len1 x len2
        #         return pairwise_transformed.squeeze(3)
        #
        #     # batch x len1 x len2
        #     return self.output_transform(pairwise_transformed).squeeze(3)


class Lambda(nn.Module):

    def __init__(self, lambd):
        super(Lambda, self).__init__()
        self.lambd = lambd

    def forward(self, *args):
        return self.lambd(*args)


class Pool(LazyModule):
    """Module that aggregates a given sequence of vectors to produce a single vector."""

    _supported_styles = [
        'avg', 'divsqrt', 'inv-freq-avg', 'sif', 'max', 'last', 'last-simple',
        'birnn-last', 'birnn-last-simple'
    ]

    @classmethod
    def supports_style(cls, style):
        return style.lower() in cls._supported_styles

    def _init(self, style, alpha=0.001):
        assert self.supports_style(style)
        self.style = style.lower()
        self.register_buffer('alpha', torch.Tensor([alpha]))

    def _forward(self, input_with_meta):
        input = input_with_meta.data

        if self.style == 'last':
            lengths = input_with_meta.lengths
            lasts = Variable(lengths.view(-1, 1, 1).repeat(1, 1, input.size(2))) - 1
            output = torch.gather(input, 1, lasts).squeeze(1).float()
        elif self.style == 'last-simple':
            output = input[:, input.size(1), :]
        elif self.style == 'birnn-last':
            hsize = input.size(2) // 2
            lengths = input_with_meta.lengths
            lasts = Variable(lengths.view(-1, 1, 1).repeat(1, 1, hsize)) - 1

            forward_outputs = input.narrow(2, 0, input.size(2) // 2)
            forward_last = forward_outputs.gather(1, lasts).squeeze(1)

            backward_last = input[:, 0, hsize:]
            output = torch.cat((forward_last, backward_last), 1)
        elif self.style == 'birnn-last-simple':
            forward_last = input[:, input.size(1), :hsize]
            backward_last = input[:, 0, hsize:]
            output = torch.cat((forward_last, backward_last), 1)
        elif self.style == 'max':
            if input_with_meta.lengths is not None:
                mask = _utils.sequence_mask(input_with_meta.lengths)
                mask = mask.unsqueeze(2)  # Make it broadcastable.
                input.data.masked_fill_(1 - mask, -float('inf'))
            output = input.max(dim=1)[0]
        else:
            if input_with_meta.lengths is not None:
                mask = _utils.sequence_mask(input_with_meta.lengths)
                mask = mask.unsqueeze(2)  # Make it broadcastable.
                input.data.masked_fill_(1 - mask, 0)

            lengths = Variable(input_with_meta.lengths.clamp(min=1).unsqueeze(1).float())
            if self.style == 'avg':
                output = input.sum(1) / lengths
            elif self.style == 'divsqrt':
                output = input.sum(1) / lengths.sqrt()
            elif self.style == 'inv-freq-avg':
                inv_probs = self.alpha / (input_with_meta.word_probs + self.alpha)
                weighted = input * Variable(inv_probs.unsqueeze(2))
                output = weighted.sum(1) / lengths.sqrt()
            elif self.style == 'sif':
                inv_probs = self.alpha / (input_with_meta.word_probs + self.alpha)
                weighted = input * Variable(inv_probs.unsqueeze(2))
                output = (weighted.sum(1) / lengths.sqrt()) - Variable(input_with_meta.pc)
            else:
                raise NotImplementedError(self.style + ' is not implemented.')

        return AttrTensor.from_old_metadata(output, input_with_meta)


class Merge(LazyModule):
    """Module that takes two or more vectors and merges them produce a single vector."""

    _style_map = {
        'concat': lambda *args: torch.cat(args, args[0].dim() - 1),
        'diff': lambda x, y: x - y,
        'abs-diff': lambda x, y: torch.abs(x - y),
        'concat-diff': lambda x, y: torch.cat((x, y, x - y), x.dim() - 1),
        'concat-abs-diff': lambda x, y: torch.cat((x, y, torch.abs(x - y)), x.dim() - 1),
        'mul': lambda x, y: torch.mul(x, y)
    }

    @classmethod
    def supports_style(cls, style):
        return style.lower() in cls._style_map

    def _init(self, style):
        assert self.supports_style(style)
        self.op = Merge._style_map[style.lower()]

    def _forward(self, *args):
        return self.op(*args)


class Bypass(LazyModule):
    """Module that helps bypass a given transformation of an input.

    Supports residual and highway styles of bypass."""

    _supported_styles = ['residual', 'highway']

    @classmethod
    def supports_style(cls, style):
        return style.lower() in cls._supported_styles

    def _init(self, style, residual_scale=True, highway_bias=-2, input_size=None):
        assert self.supports_style(style)
        self.style = style.lower()
        self.residual_scale = residual_scale
        self.highway_bias = highway_bias
        self.highway_gate = nn.Linear(input_size[1], input_size[0])

    def _forward(self, transformed, raw):
        assert transformed.shape[:-1] == raw.shape[:-1]

        tsize = transformed.shape[-1]
        rsize = raw.shape[-1]
        adjusted_raw = raw
        if tsize < rsize:
            assert rsize / tsize <= 50
            if rsize % tsize != 0:
                padded = F.pad(raw, (0, tsize - rsize % tsize))
            else:
                padded = raw
            adjusted_raw = padded.view(*raw.shape[:-1], -1, tsize).sum(-2) * math.sqrt(
                tsize / rsize)
        elif tsize > rsize:
            multiples = math.ceil(tsize / rsize)
            adjusted_raw = raw.repeat([1] * (raw.dim() - 1), multiples).narrow(
                -1, 0, tsize)
            pdb.set_trace()

        if self.style == 'residual':
            res = transformed + adjusted_raw
            if self.residual_scale:
                res *= math.sqrt(0.5)
            return res
        elif self.style == 'highway':
            transform_gate = F.sigmoid(self.highway_gate(raw) + self.highway_bias)
            carry_gate = 1 - transform_gate
            return transform_gate * transformed + carry_gate * adjusted_raw


class Transform(LazyModule):
    """A multi layered transformation module.

    Supports various non-linearities and bypass operations"""

    _supported_nonlinearities = [
        'sigmoid', 'tanh', 'relu', 'elu', 'selu', 'glu', 'leaky_relu'
    ]

    @classmethod
    def supports_nonlinearity(cls, nonlin):
        return nonlin.lower() in cls._supported_nonlinearities

    def _init(self,
              style,
              layers=1,
              bypass_network=None,
              non_linearity='leaky_relu',
              hidden_size=None,
              input_size=None,
              output_size=None,
              force_bypass=False):
        hidden_size = hidden_size or input_size
        output_size = output_size or hidden_size

        parts = style.split('-')

        if 'layer' in parts:
            layers = int(parts[parts.index('layer') - 1])

        for part in parts:
            if Bypass.supports_style(part):
                bypass_network = part
            if Transform.supports_nonlinearity(part):
                non_linearity = part

        self.transforms = nn.ModuleList()
        self.bypass_networks = nn.ModuleList()

        assert (non_linearity is None or self.supports_nonlinearity(non_linearity))
        self.non_linearity = non_linearity.lower() if non_linearity else None

        transform_in_size = input_size
        transform_out_size = hidden_size
        for layer in range(layers):
            if layer == layers - 1:
                transform_out_size = output_size
            self.transforms.append(nn.Linear(transform_in_size, transform_out_size))
            self.bypass_networks.append(_bypass_module(bypass_network))
            transform_in_size = transform_out_size

    def _forward(self, input):
        output = input

        for transform, bypass in zip(self.transforms, self.bypass_networks):
            new_output = transform(output)
            if self.non_linearity:
                new_output = getattr(F, self.non_linearity)(new_output)
            if bypass:
                new_output = bypass(new_output, output)
            output = new_output

        return output


class Identity(LazyModule):
    """Identity transform module."""

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
    output_size = output_size or hidden_size
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
