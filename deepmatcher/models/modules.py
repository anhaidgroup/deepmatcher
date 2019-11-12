from __future__ import division

import abc
import logging
import math

import six

import deepmatcher as dm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from . import _utils
from ..batch import AttrTensor

logger = logging.getLogger('deepmatcher.modules')


@six.add_metaclass(abc.ABCMeta)
class LazyModule(nn.Module):
    r"""A lazily initialized module. Base class for most DeepMatcher modules.

    This module is an extension of PyTorch :class:`~torch.nn.Module` with the following
    property: constructing an instance this module does not immediately initialize it.
    This means that if the module has parameters, they will not be instantiated
    immediately after construction. The module is initialized the first time `forward` is
    called. This has the following benefits:

    * Can be safely deep copied to create structural clones that do not share
      parameters. E.g. deep copying a :class:`LazyModule` consisting of a 2 layer Linear
      NN will produce another :class:`LazyModule` with 2 layer Linear NN that 1) do not
      share parameters and 2) have different weight initializations.
    * Allows automatic input size inference. Refer to description of `_init` for details.

    This module also implements some additional goodies:

    * Output shape verification: As part of initialization, this module verifies that
      all output tensors have correct output shapes, if the expected output shape is
      specified using :meth:`expect_signature`. This verification is done only once during
      initialization to avoid slowing down training.
    * NaN checks: All module outputs are cheked for the presence of NaN values that may be
      difficult to trace down otherwise.

    Subclasses of this module are expected to override the following two methods:

    * _init(): This is where the constructor of the module should be defined. During the
      first forward pass, this method will be called to initialize the module. Whatever
      you typically define in the __init__ function of a PyTorch module, you may define
      it here. This function may optionally take in an `input_size` parameter. If it does,
      :class:`LazyModule` will set it to the size of the last dimension of the input.
      E.g., if the input is of size `32 * 300`, the `input_size` will be set to 300.
      Subclasses may choose not to override this method.
    * _forward(): This is where the computation for the forward pass of the module must be
      defined. Whatever you typically define in the forward function of a PyTorch module,
      you may define it here. All subclasses must override this method.
    """

    def __init__(self, *args, **kwargs):
        """Construct a :class:`LazyModule`. DO NOT OVERRIDE this method.

        This does NOT initialize the module - construction simply saves the positional and
        keyword arguments for future initialization.

        Args:
            *args: Positional arguments to the constructor of the module defined in
                :meth:`_init`.
            **kwargs: Keyword arguments to the constructor of the module defined in
                :meth:`_init`.
        """
        super(LazyModule, self).__init__()
        self._init_args = args
        self._init_kwargs = kwargs
        self._initialized = False
        self._fns = []
        self.signature = None

    def forward(self, input, *args, **kwargs):
        """Perform a forward pass through the module. DO NOT OVERRIDE this method.

        If the module is not initialized yet, this method also performs initialization.
        Initialization involves the following:

        1. Calling the :meth:`_init` method. Tries calling with the `input_size` keyword
           parameter set, along with the positional and keyword args specified during
           construction). If this fails with a :exc:`TypeError` (i.e., the
           :meth:`_init` method does not have an `input_size` parameter), then retries
           initialization without setting `input_size`.
        2. Verifying the output shape, if :meth:`expect_signature` was called prior to
           the forward pass.
        3. Setting PyTorch :class:`~torch.nn.Module` forward and backward hooks to check
           for NaNs in module outputs and gradients.

        Args:
            *args: Positional arguments to the forward function of the module defined in
                :meth:`_forward`.
            **kwargs: Keyword arguments to the forward function of the module defined in
                :meth:`_forward`.
        """
        if not self._initialized:
            try:
                self._init(
                    *self._init_args,
                    input_size=self._get_input_size(input, *args, **kwargs),
                    **self._init_kwargs)
            except TypeError as e:
                logger.debug('Got exception when passing input size: ' + str(e))
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
        """Set the expected module input / output signature.

        Note that this feature is currently not fully functional. More details will be
        added after implementation.
        """
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
    r"""A wrapper module to allow regular modules to take
    :class:`~deepmatcher.batch.AttrTensor` s as input.

    A forward pass through this module, will perform the following:

    * If the module input is an :class:`~deepmatcher.batch.AttrTensor`, gets the data from
      it, and use as input.
    * Perform a forward pass through wrapped module with the modified input.
    * Using metadata information from the module input (if provided), wrap the result into
      an :class:`~deepmatcher.batch.AttrTensor` and return it.

    Args:
        module (:class:`~torch.nn.Module`): The module to wrap.
    """

    def __init__(self, module):
        super(NoMeta, self).__init__()
        self.module = module

    def forward(self, *args):
        module_args = []
        for arg in args:
            module_args.append(arg.data if isinstance(arg, AttrTensor) else arg)

        results = self.module(*module_args)

        if not isinstance(args[0], AttrTensor):
            return results
        else:
            if not isinstance(results, tuple):
                results = (results,)

            if len(results) != len(args) and len(results) != 1 and len(args) != 1:
                raise ValueError(
                    'Number of inputs must equal number of outputs, or '
                    'number of inputs must be 1 or number of outputs must be 1.')

            results_with_meta = []
            for i in range(len(results)):
                arg_i = min(i, len(args) - 1)
                results_with_meta.append(
                    AttrTensor.from_old_metadata(results[i], args[arg_i]))

            if len(results_with_meta) == 1:
                return results_with_meta[0]

            return tuple(results_with_meta)


class ModuleMap(nn.Module):
    """Holds submodules in a map.

    Similar to :class:`torch.nn.ModuleList`, but for maps.

    Example::

        import torch.nn as nn
        import deepmatcher as dm

        class MyModule(nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                linears = dm.ModuleMap()
                linears['type'] = nn.Linear(10, 10)
                linears['color'] = nn.Linear(10, 10)
                self.linears = linears

            def forward(self, x1, x2):
                y1, y2 = self.linears['type'], self.linears['color']
                return y1, y2

    """

    def __getitem__(self, name):
        return getattr(self, name)

    def __setitem__(self, name, module):
        setattr(self, name, module)

    def __delitem__(self, name):
        delattr(self, name)


class MultiSequential(nn.Sequential):
    """A sequential container that supports multiple module inputs and outputs.

    This is an extenstion of PyTorch's :class:`~torch.nn.Sequential` module that allows
    each module to have multiple inputs and / or outputs.
    """

    def forward(self, *inputs):
        modules = list(self._modules.values())
        inputs = modules[0](*inputs)
        for module in modules[1:]:
            if isinstance(inputs, tuple) and not isinstance(inputs, AttrTensor):
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class LazyModuleFn(LazyModule):
    """A Lazy Module which simply wraps the :class:`~torch.nn.Module` returned by a
    specified function.

    This provides a way to convert a PyTorch :class:`~torch.nn.Module` into a
    :class:`LazyModule`.

    Args:
        fn (callable):
            Function that returns a :class:`~torch.nn.Module`.

        *args:
            Positional arguments to the function `fn`.

        *kwargs:
            Keyword arguments to the function `fn`.
    """

    def _init(self, fn, *args, **kwargs):
        self.module = fn(*args, **kwargs)

    def _forward(self, *args, **kwargs):
        return self.module.forward(*args, **kwargs)


class RNN(LazyModule):
    r"""__init__(unit_type='gru', hidden_size=None, layers=1, bidirectional=True, dropout=0, input_dropout=0, last_layer_dropout=0, bypass_network=None, connect_num_layers=1, input_size=None, **kwargs)

    A multi layered RNN that supports dropout and residual / highway connections.

    Args:
        unit_type (string):
            One of the support RNN unit types:

            * '**gru**': Apply a gated recurrent unit (GRU) RNN. Uses PyTorch
              :class:`~torch.nn.GRU` under the hood.
            * '**lstm**': Apply a long short-term memory unit (LSTM) RNN. Uses PyTorch
              :class:`~torch.nn.LSTM` under the hood.
            * '**rnn**': Apply an Elman RNN. Uses PyTorch :class:`~torch.nn.RNN` under the
              hood.

        hidden_size (int):
            The hidden size of all RNN layers.

        layers (int):
            Number of RNN layers.

        bidirectional (bool):
            Whether to use bidirectional RNNs.

        dropout (float):
            If non-zero, applies dropout to the outputs of each RNN layer except the last
            layer. Dropout probability must be between 0 and 1.

        input_dropout (float):
            If non-zero, applies dropout to the input to this module. Dropout probability
            must be between 0 and 1.

        last_layer_dropout (float):
            If non-zero, applies dropout to the output of the last RNN layer. Dropout
            probability must be between 0 and 1.

        bypass_network (string or :class:`Bypass` or callable):
            The bypass network (e.g. residual or highway network) to apply every
            `connect_num_layers` layers. Argument must specify a :ref:`bypass-op`
            operation. If None, does not use a bypass network.

        connect_num_layers (int):
            The number of layers between each bypass operation. Note that the layers in
            which dropout is applied is also controlled by this. If `layers` is 6 and
            `connect_num_layers` is 2, then a bypass network is applied after the
            2nd, 4th and 6th layers. Further, if `dropout` is non-zero, it will only be
            applied after the 2nd and 4th layers.

        input_size (int):
            The number of features in the input to the module. This parameter will be
            automatically specified by :class:`LazyModule`.

        **kwargs (dict):
            Additional keyword arguments are passed to the underlying PyTorch RNN module.

    Input: One 3d tensor of shape `(batch, seq_len, input_size)`.
        The tensor should be wrapped within an :class:`~deepmatcher.batch.AttrTensor`
        which contains metadata about the batch.
    Output: One 3d tensor of shape `(batch, seq_len, output_size)`.
        This will be wrapped within an :class:`~deepmatcher.batch.AttrTensor` (with
        metadata information unchanged). `output_size` need not be the same as
        `input_size`.
    """
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
              input_size=None,
              **kwargs):
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
        assert not any([a in kwargs for a in bad_args])

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
                    **kwargs))

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
    r"""__init__(style='decomposable', hidden_size=None, transform_network='2-layer-highway', input_size=None)

    Neural network to compute alignment between two vector sequences.

    Takes two sequences of vectors, aligns the words in them, and returns
    the corresponding alignment matrix.

    Args:
        style (string): One of the following strings:

            * '**decomposable**': Use decomposable attention. Alignment score between the
              :math:`i^{th}` vector in the first sequence :math:`a_i` , and the
              :math:`j^{th}` vector in the second sequence :math:`b_j` is computed as
              follows:

              .. math::

                  score(a_i, b_j) = F(a_i)^T F(b_j)

              where :math:`F` is a :ref:`transform-op` operation. Refer the
              `decomposable attention paper <https://arxiv.org/abs/1606.01933>`__ for more
              details.

            * '**general**': Use general attention. Alignment score between the
              :math:`i^{th}` vector in the first sequence :math:`a_i` , and the
              :math:`j^{th}` vector in the second sequence :math:`b_j` is computed as
              follows:

              .. math::

                  score(a_i, b_j) = a_i^T F(b_j)

              where :math:`F` is a :ref:`transform-op` operation. Refer the `Luong attention
              paper <https://arxiv.org/abs/1508.04025>`__ for more details.

            * '**dot**': Use dot product attention. Alignment score between the
              :math:`i^{th}` vector in the first sequence :math:`a_i` , and the
              :math:`j^{th}` vector in the second sequence :math:`b_j` is computed as
              follows:

            .. math::

                score(a_i, b_j) = a_i^T b_j

        hidden_size (int):
            The hidden size to use for the :ref:`transform-op` operation, if applicable
            for the specified `style`.

        transform_network (string or :class:`~deepmatcher.modules.Transform` or callable):
            The neural network to transform the input vectors, if applicable for the
            specified `style`. Argument must specify a :ref:`transform-op` operation.

        input_size (int):
            The number of features in the input to the module. This parameter will be
            automatically specified by :class:`LazyModule`.

    Input: Two 3d tensors.
        Two 3d tensors of shape `(batch, seq1_len, input_size)` and
        `(batch, seq2_len, input_size)`.

    Output: One 3d tensor of shape `(batch, seq1_len, seq2_len)`.
        The output represents the alignment matrix and contains unnormalized scores.
        `output_size` need not be the same as `input_size`, but all other dimensions will
        remain unchanged.
    """

    _supported_styles = ['dot', 'general', 'decomposable']

    @classmethod
    def supports_style(cls, style):
        return style.lower() in cls._supported_styles

    def _init(self,
              style='decomposable',
              hidden_size=None,
              transform_network='2-layer-highway',
              input_size=None):
        if style in ['general', 'decomposable']:
            if style == 'general':
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
        elif self.style == 'general':
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
    r"""Wrapper to convert a function to a module.

    Args:
        lambd (callable): The function to convert into a module. It must take in one or
            more Pytorch :class:`~torch.Tensor` s and return one or more
            :class:`~torch.Tensor` s.
    """

    def __init__(self, lambd):
        super(Lambda, self).__init__()
        self.lambd = lambd

    def forward(self, *args):
        return self.lambd(*args)


class Pool(LazyModule):
    r"""__init__(style, alpha=0.001)

    Module that aggregates a given sequence of vectors to produce a single vector.

    Args:
        style (string): One of the following strings:

            * '**avg**': Take the average of the input vectors. Given a sequence of
              vectors :math:`x_{1:N}` :

              .. math::

                  Pool(x_{1:N}) = \frac{1}{N} \sum_1^N x_i

            * '**divsqrt**': Take the sum of the input vectors :math:`x_{1:N}` and divide
              by :math:`\sqrt{N}` :

              .. math::

                  Pool(x_{1:N}) = \frac{1}{\sqrt{N}} \sum_1^N x_i

            * '**inv-freq-avg**': Take the smooth inverse frequency weighted sum of the
              :math:`N` input vectors and divide by :math:`\sqrt{N}`. This is similar to
              the 'sif' style but does not perform principal component removal. Given a
              sequence of vectors :math:`x_{1:N}` corresponding to words :math:`w_{1:N}`:

              .. math::

                  Pool(x_{1:N}) = \frac{1}{\sqrt{N}}
                      \sum_1^N \frac{\alpha}{\alpha + P(w)} x_i

              where :math:`P(w)` is the unigram probability of word :math:`w` (computed
              over all values of this attribute over the entire training dataset) and
              :math:`\alpha` is a scalar (specified by the `alpha` parameter).
              :math:`P(w)` is computed in :class:`~deepmatcher.data.MatchingDataset`, in
              the :meth:`~deepmatcher.data.MatchingDataset.compute_metadata` method.

            * '**sif**': Compute the
              `SIF encoding <https://openreview.net/pdf?id=SyK00v5xx>`__ of the input
              vectors. Takes the smooth inverse frequency weighted sum of the :math:`N`
              input vectors and divides it by :math:`\sqrt{N}`. Also removes the
              projection of the resulting vector along the first principal component of
              all word embeddings (corresponding to words in this attribute in the
              training set). Given a sequence of vectors :math:`x_{1:N}` corresponding to
              words :math:`w_{1:N}`:

              .. math::

                  v_x = \frac{1}{\sqrt{N}} \sum_1^N \frac{\alpha}{\alpha + P(w)} x_i

                  Pool(x_{1:N}) = v_x - u^T u v_x

              where :math:`u` is the first principal component as described earlier,
              :math:`P(w)` is the unigram probability of word :math:`w` (computed over all
              values of this attribute over the entire training dataset) and
              :math:`\alpha` is a scalar (specified by the `alpha` parameter). :math:`u`
              and :math:`P(w)` are computed in :class:`~deepmatcher.data.MatchingDataset`,
              in the :meth:`~deepmatcher.data.MatchingDataset.compute_metadata` method.

            * '**max**': Take the max of the input vector sequence along each input
              feature. If length metadata for each item in the input batch is available,
              ignores the padding vectors beyond the sequence length of each item when
              computing the max.

            * '**last**': Take the last vector in the input vector sequence. If length
              metadata for each item in the input batch is available, ignores the padding
              vectors beyond the sequence length of each item when taking the last vector.

            * '**last-simple**': Take the last vector in the input vector sequence. Does
              NOT take length metadata into account - simply takes the last vector for
              each input sequence in the batch.

            * '**birnn-last**': Treats the input sequence as the output from a
              bidirectional RNN and takes the last outputs from the forward and backward
              RNNs. The first half of each vector is assumed to be from the forward RNN
              and the second half is assumed to be from the bakward RNN. The output thus
              is the concatenation of first half of the last vector in the input sequence
              and the last half of the first vector in the sequence. If length
              metadata for each item in the input batch is available, ignores the padding
              vectors beyond the sequence length of each item when taking the last vectors
              for the forward RNN.

            * '**birnn-last-simple**': Treats the input sequence as the output from a
              bidirectional RNN and takes the last outputs from the forward and backward
              RNNs. Same as the 'birnn-last' style but does not consider length metadata
              even if available.

        alpha (float): The value used to smooth the inverse word frequencies. Used for
            'inv-freq-avg' and 'sif' `styles`.


    Input: A 3d tensor of shape `(batch, seq_len, input_size)`.
        The tensor should be wrapped within an :class:`~deepmatcher.batch.AttrTensor`
        which contains metadata about the batch.

    Output: A 2d tensor of shape `(batch, output_size)`.
        This will be wrapped within an :class:`~deepmatcher.batch.AttrTensor` (with
        metadata information unchanged). `output_size` need not be the same as
        `input_size`.
    """

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
                input.data.masked_fill_(~mask, -float('inf'))
            output = input.max(dim=1)[0]
        else:
            if input_with_meta.lengths is not None:
                mask = _utils.sequence_mask(input_with_meta.lengths)
                mask = mask.unsqueeze(2)  # Make it broadcastable.
                input.data.masked_fill_(~mask, 0)

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
                v = (weighted.sum(1) / lengths.sqrt())
                pc = Variable(input_with_meta.pc).unsqueeze(0).repeat(v.shape[0], 1)
                proj_v_on_pc = torch.bmm(v.unsqueeze(1), pc.unsqueeze(2)).squeeze(2) * pc
                output = v - proj_v_on_pc
            else:
                raise NotImplementedError(self.style + ' is not implemented.')

        return AttrTensor.from_old_metadata(output, input_with_meta)


class Merge(LazyModule):
    r"""__init__(style)

    Module that takes two or more vectors and merges them produce a single vector.

    Args:
        style (string): One of the following strings:

            * '**concat**': Concatenate all the input vectors along the last dimension
              (-1).
            * '**diff**': Take the difference between two input vectors.
            * '**abs-diff**': Take the absolute difference between two input vectors.
            * '**concat-diff**': Concatenate the two input vectors, take the difference
              between the two vectors, and concatenate these two resulting vectors.
            * '**concat-abs-diff**': Concatenate the two input vectors, take the absolute
              difference between the two vectors, and concatenate these two resulting
              vectors.
            * '**mul**': Take the element-wise multiplication between the two input
              vectors.

    Input: N K-d tensors of shape `(D1, D2, ..., input_size)`.
        N and K are both 2 or more.

    Output: One K-d tensor of shape `(D1, D2, ..., output_size)`.
        `output_size` need not be the same as `input_size`, but all other dimensions will
        remain unchanged.
    """

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
    r"""__init__(style)

    Module that helps bypass a given transformation of an input.

    Supports residual and highway styles of bypass.

    Args:
        style (string): One of the following strings:

            * '**residual**': Uses a `residual network <https://arxiv.org/abs/1512.03385>`__.
            * '**highway**': Uses a `highway network <https://arxiv.org/abs/1505.00387>`__.

    Input: Two N-d tensors.
        Two N-d tensors of shape `(D1, D2, ..., transformed_size)` and
        `(D1, D2, ..., input_size)`. The first tensor should corresponds to the
        transformed version of the second input.

    Output: One N-d tensor of shape `(D1, D2, ..., transformed_size)`.
        Note that the shape of the output will match the shape of the first input tensor.
    """

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
            adjusted_raw = raw.repeat(*([1] * (raw.dim() - 1)), multiples).narrow(
                -1, 0, tsize)

        if self.style == 'residual':
            res = transformed + adjusted_raw
            if self.residual_scale:
                res *= math.sqrt(0.5)
            return res
        elif self.style == 'highway':
            transform_gate = torch.sigmoid(self.highway_gate(raw) + self.highway_bias)
            carry_gate = 1 - transform_gate
            return transform_gate * transformed + carry_gate * adjusted_raw


class Transform(LazyModule):
    r"""__init__(style, layers=1, bypass_network=None, non_linearity='leaky_relu', hidden_size=None, output_size=None, input_size=None)

    A multi layered transformation module.

    Supports various non-linearities and bypass operations.

    Args:
        style (string):
            A string containing one or more of the following 3 parts, separated by dashes
            (-):

            * '**<N>-layer**': Specifies the number of layers. <N> sets the `layers`
              parameter. E.g.: '**2-layer**-highway'.

            * '**<nonlinearity>**': Specifies the non-linearity used after each layer.
              Sets the `non_linearity` parameter, refer that for details.

            * '**<bypass>**' Specifies the :ref:`bypass-op` operation to use.
              Sets the `bypass_network` parameter. <bypass> is one of:

              * 'residual': Use :class:`Bypass` with 'residual' `style`.
              * 'highway': Use :class:`Bypass` with 'highway' `style`.

            If any of the 3 parts are missing, the default value for the corresponding
            parameter is used.

            Examples: Sample `styles`
                '3-layer-relu-highway', 'tanh-residual-2-layer', 'tanh', 'highway',
                '4-layer'.

        layers (int):
            Number of linear transformation layers to use.

        bypass_network (string or :class:`Bypass` or callable):
            The bypass network (e.g. residual or highway network) to apply every layer.
            The input to each linear layer is considered as the raw input to the bypass
            network and the output of the non-linearity operation is considered as the
            transformed input. Argument must specify a :ref:`bypass-op` operation. If
            None, does not use a bypass network.

        non_linearity (string):
            The non-linearity to use after each linear layer. One of:

            * '**leaky_relu**': Use PyTorch :class:`~torch.nn.LeakyReLU`.
            * '**relu**': Use PyTorch :class:`~torch.nn.ReLU`.
            * '**elu**': Use PyTorch :class:`~torch.nn.ELU`.
            * '**selu**': Use PyTorch :class:`~torch.nn.SELU`.
            * '**glu**': Use PyTorch :func:`~torch.nn.functional.glu`.
            * '**tanh**': Use PyTorch :class:`~torch.nn.Tanh`.
            * '**sigmoid**': Use PyTorch :class:`~torch.nn.Sigmoid`.

        hidden_size (int):
            The hidden size of the linear transformation layers. If None, will be set
            to be equal to `input_size`.

        output_size (int):
            The hidden size of the last linear transformation layer. Will determine the
            number of features in the output of the module. If None, will be set to be
            equal to the `hidden_size`.

        input_size (int):
            The number of features in the input to the module. This parameter will be
            automatically specified by :class:`LazyModule`.

    Input: An N-d tensor of shape `(D1, D2, ..., input_size)`.
        N is 2 or more.

    Output: An N-d tensor of shape `(D1, D2, ..., output_size)`.
        `output_size` need not be the same as `input_size`, but all other dimensions will
        remain unchanged.
    """

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
              output_size=None,
              input_size=None):
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
