.. role:: hidden
    :class: hidden-section

deepmatcher.modules
=========================

.. automodule:: deepmatcher.modules
.. currentmodule:: deepmatcher.modules

Standard Operations
-------------------------------

Many components in DeepMatcher, e.g. AttentionWithRNN word aggregator, allow users to
customize the behavior of operations performed by the component, such as alignment, vector
transformation, pooling, etc., by setting a parameter that specifies the operation. Here
we describe operations commonly used across the package, and show how to specify them.

.. _transform-op:

Transform
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The transform operation takes a single vector and performs transforms it to produce
another vector as output. The transformation may be non-linear. A transform operation can
be specified by using one of the following:

* A string: One of the `styles` supported by the :class:`Transform` module.
* An instance of :class:`Transform`.
* A :attr:`callable`: A function that returns a PyTorch :class:`~torch.nn.Module`. This
  module must have the same input and output shape signature as the :class:`Transform`
  module.

This operation is implemented by the :class:`Transform` module:

.. autoclass:: Transform
    :members:

.. _pool-op:

Pool
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Pool operation takes a sequence of vectors and aggregates this sequence to produce a
single vector as output. A Pool operation can be specified using one of the
following:

* A string: One of the `styles` supported by the :class:`Pool` module.
* An instance of :class:`Pool`.
* A :attr:`callable`: A function that returns a PyTorch :class:`~torch.nn.Module`. This
  module must have the same input and output shape signature as the :class:`Pool` module.

This operation is implemented by the :class:`Pool` module:

.. autoclass:: Pool
    :members:

.. _merge-op:

Merge
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Merge operation takes two or more vectors and aggregates the information in them to
produce a single vector as output. Unlike the case of :ref:`pool-op` operation, the input
vectors here are not considered to be sequential in nature. A Merge operation can be
specified using one of the following:

* A string: One of the `styles` supported by the :class:`Merge` module. Note that some
  styles only support two input vectors to be merged, while others allow multiple inputs.
* An instance of :class:`Merge`.
* A :attr:`callable`: A function that returns a PyTorch :class:`~torch.nn.Module`. This
  module must have the same input and output shape signature as the
  :class:`Merge` module.

This operation is implemented by the :class:`Merge` module:

.. autoclass:: Merge
    :members:

.. _align-op:

Align
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Align operation takes two sequences of vectors, aligns the words in them, and returns
the corresponding alignment score matrix. For each word in the first sequence, the
alignment matrix contains unnormalized scores indicating the degree to which each word in
the second sequence aligns with it. For an example of one way to do this, take a look `at
this paper <https://arxiv.org/abs/1606.01933>`__. An Align operation can be specified
using one of the following:

* A string: One of the `styles` supported by the :class:`AlignmentNetwork` module.
* An instance of :class:`AlignmentNetwork`.
* A :attr:`callable`: A function that returns a PyTorch :class:`~torch.nn.Module`. This
  module must have the same input and output shape signature as the
  :class:`AlignmentNetwork` module.

This operation is implemented by the :class:`AlignmentNetwork` module:

.. autoclass:: AlignmentNetwork
    :members:

.. _bypass-op:

Bypass
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Bypass operation takes two tensors, one corresponding to an input tensor and the other
corresponding to a transformed version of the first tensor, applies a bypass network
and returns one tensor of the same size as the transformed tensor. Examples of bypass
networks include `residual networks <https://arxiv.org/abs/1512.03385>`__ and
`highway networks <https://arxiv.org/abs/1505.00387>`__.

* A string: One of the `styles` supported by the :class:`Bypass` module.
* An instance of :class:`Bypass`.
* A :attr:`callable`: A function that returns a PyTorch :class:`~torch.nn.Module`. This
  module must have the same input and output shape signature as the
  :class:`Bypass` module.

This operation is implemented by the :class:`Bypass` module:

.. autoclass:: Bypass
    :members:

.. _rnn-op:

RNN
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This operation takes a sequence of vectors and produces a context-aware transformation of
the input sequence as output. For an intro to RNNs, take a look at
`this article <http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/>`__.
An RNN operation can be specified using one of the following:

* A string: One of the `unit_types` supported by the :class:`RNN` module.
* An instance of :class:`RNN`.
* A :attr:`callable`: A function that returns a PyTorch :class:`~torch.nn.Module`. This
  module must have the same input and output shape signature as the
  :class:`RNN` module.

This operation is implemented by the :class:`RNN` module:

.. autoclass:: RNN
    :members:

Utility Modules
-------------------

Apart from standard operations, DeepMatcher also contains several utility modules to help
glue together various components. These are listed below.

:hidden:`Lambda`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: Lambda
    :members:

:hidden:`MultiSequential`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: MultiSequential
    :members:

:hidden:`NoMeta`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: NoMeta
    :members:

:hidden:`ModuleMap`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ModuleMap
    :members:

:hidden:`LazyModule`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: LazyModule
    :members: __init__, forward, expect_signature

:hidden:`LazyModuleFn`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: LazyModuleFn
    :members:
