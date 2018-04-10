import deepmatcher as dm


class SIF(dm.AttrSummarizer):

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
