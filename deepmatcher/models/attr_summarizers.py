"""Defines built-in attribute summarizers."""

import deepmatcher as dm


class SIF(dm.AttrSummarizer):
    """__init__(word_contextualizer=None, word_comparator=None, word_aggregator=None, \
        hidden_size=None)

    The attribute summarizer for the SIF (Smooth Inverse Frequency) model.

    Args:
        word_contextualizer (string or :class:`~deepmatcher.WordContextualizer` or callable): The
            word contextualizer module (refer to :class:`~deepmatcher.WordContextualizer` for
            details) to use for attribute summarization. The SIF model does not take word
            context information into account, hence this defaults to None.
        word_comparator (string or :class:`~deepmatcher.WordComparator` or callable): The word
            comparator module (refer to :class:`~deepmatcher.WordComparator` for details) to use
            for attribute summarization. The SIF model does not perform word by word
            comparisons, hence this defaults to None.
        word_aggregator (string or :class:`~deepmatcher.WordAggregator` or callable): The word
            aggregator module (refer to :class:`~deepmatcher.WordAggregator` for details) to use
            for attribute summarization. This model uses SIF-based weighted average
            aggregation over the  word embeddings of an input sequence, hence this
            defaults to 'sif-pool'.
        hidden_size (int): The hidden size to use for all 3 attribute summarization
            sub-modules (i.e., word contextualizer, word comparator, and word aggregator),
            if they are customized. By default, the SIF model does not use this parameter.
    """

    def _init(self,
              word_contextualizer=None,
              word_comparator=None,
              word_aggregator=None,
              hidden_size=None):
        word_aggregator = word_aggregator or 'sif-pool'
        super(SIF, self)._init(
            word_contextualizer=word_contextualizer,
            word_comparator=word_comparator,
            word_aggregator=word_aggregator,
            hidden_size=hidden_size)


class RNN(dm.AttrSummarizer):
    r"""The attribute summarizer for the RNN model.

    Args:
        word_contextualizer (string or :class:`~deepmatcher.WordContextualizer` or callable): The
            word contextualizer module (refer to :class:`~deepmatcher.WordContextualizer` for
            details) to use for attribute summarization. This model uses RNN to take into
            account the context information, and the default value is 'gru' (i.e., uses
            the bidirectional GRU model as the specific RNN instantiation.) Other options
            are 'rnn' (the vanilla bi-RNN) and 'lstm' (the bi-LSTM model).
        word_comparator (string or :class:`~deepmatcher.WordComparator` or callable): The word
            comparator module (refer to :class:`~deepmatcher.WordComparator` for details) to use
            for attribute summarization. The RNN model does not perform word by word
            comparisons, hence this defaults to None.
        word_aggregator (string or :class:`~deepmatcher.WordAggregator` or callable): The word
            aggregator module (refer to :class:`~deepmatcher.WordAggregator` for details) to use
            for attribute summarization. The RNN model uses bi-directional RNN and
            concatenates the last ouputs of the forward and backward RNNs, hence the
            default value is 'birnn-last-pool'.
        hidden_size (int): The hidden size to use for the word contextualizer. This value
            will also be used as the hidden size for the other 2 attribute summarization
            sub-modules (i.e., word comparator, and word aggregator), if they are
            customized. If not specified, the hidden size for each component will be set
            to be the same as its input size. E.g. if the word embedding dimension is 300
            and hidden_size is None, the word contextualizer's hidden size will be 300.
    """

    def _init(self,
              word_contextualizer=None,
              word_comparator=None,
              word_aggregator=None,
              hidden_size=None):
        word_contextualizer = word_contextualizer or 'gru'
        word_aggregator = word_aggregator or 'birnn-last-pool'
        super(RNN, self)._init(
            word_contextualizer=word_contextualizer,
            word_comparator=word_comparator,
            word_aggregator=word_aggregator,
            hidden_size=hidden_size)


class Attention(dm.AttrSummarizer):
    r"""The attribute summarizer for the attention-based model.

    Args:
        word_contextualizer (string or :class:`~deepmatcher.WordContextualizer` or callable): The
            word contextualizer module (refer to :class:`~deepmatcher.WordContextualizer` for
            details) to use for attribute summarization. The attention model does not take
            word context information into account, hence this defaults to None.
        word_comparator (string or :class:`~deepmatcher.WordComparator` or callable): The word
            comparator module (refer to :class:`~deepmatcher.WordComparator` for details) to use
            for attribute summarization. The attention model performs word by word
            comparison with the decomposable attention mechanism, hence this defaults to
            'decomposable-attention'.
        word_aggregator (string or :class:`~deepmatcher.WordAggregator` or callable): The word
            aggregator module (refer to :class:`~deepmatcher.WordAggregator` for details) to use
            for attribute summarization. The Attention model performs the aggregation by
            summing over the comparison results from the word comparator, divided by the
            length of the input sequence (to get constant variance through the network
            flow). Hence this defaults to 'divsqrt-pool'.
        hidden_size (int): The hidden size to use for the word comparator. This value
            will also be used as the hidden size for the other 2 attribute summarization
            sub-modules (i.e., word contextualizer, and word aggregator), if they are
            customized. If not specified, the hidden size for each component will be set
            to be the same as its input size. E.g. if the word embedding dimension is 300
            and hidden_size is None, the word contextualizer's hidden size will be 300.
    """

    def _init(self,
              word_contextualizer=None,
              word_comparator=None,
              word_aggregator=None,
              hidden_size=None):
        word_comparator = word_comparator or 'decomposable-attention'
        word_aggregator = word_aggregator or 'divsqrt-pool'
        super(Attention, self)._init(
            word_contextualizer=word_contextualizer,
            word_comparator=word_comparator,
            word_aggregator=word_aggregator,
            hidden_size=hidden_size)


class Hybrid(dm.AttrSummarizer):
    r"""The attribute summarizer for the hybrid model.

    Args:
        word_contextualizer (string or :class:`~deepmatcher.WordContextualizer` or callable): The
            word contextualizer module (refer to :class:`~deepmatcher.WordContextualizer` for
            details) to use for attribute summarization. The hybrid model uses
            bidirectional GRU(a specific type of RNN) to take into account the context
            information. The default value is 'gru'.
        word_comparator (string or :class:`~deepmatcher.WordComparator` or callable): The word
            comparator module (refer to :class:`~deepmatcher.WordComparator` for details) to use
            for attribute summarization. The hybrid model performs word by word comparison
            over the raw input word embeddings (rather than the RNN hiddens states), hence
            this defaults to an Attention object with 'decomposable' as the attention
            mechanism on the raw input embeddings.
        word_aggregator (string or :class:`~deepmatcher.WordAggregator` or callable): The word
            aggregator module (refer to :class:`~deepmatcher.WordAggregator` for details) to use
            for attribute summarization. A second layer of attention has been used for the
            aggregation. Please consult the paper for more information. The default value
            is 'concat-attention-with-rnn'.
        hidden_size (int): The hidden size to use for all 3 attribute summarization
            sub-modules (i.e., word contextualizer, word comparator, and word aggregator),
            if they are customized.
    """

    def _init(self,
              word_contextualizer=None,
              word_comparator=None,
              word_aggregator=None,
              hidden_size=None):
        word_contextualizer = word_contextualizer or 'gru'
        word_comparator = word_comparator or dm.word_comparators.Attention(
            alignment_network='decomposable', raw_alignment=True)
        word_aggregator = word_aggregator or 'attention-with-rnn'
        super(Hybrid, self)._init(
            word_contextualizer=word_contextualizer,
            word_comparator=word_comparator,
            word_aggregator=word_aggregator,
            hidden_size=hidden_size)
