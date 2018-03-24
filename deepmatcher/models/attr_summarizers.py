import deepmatcher as dm


class SIF(dm.AttrSummarizer):
    pass


class RNN(dm.AttrSummarizer):

    def _init(self, hidden_size=None):
        super(RNN, self)._init(
            word_contextualizer='gru',
            word_comparator=None,
            word_aggregator='birnnlast-pool')


class Attention(dm.AttrSummarizer):

    def _init(self, hidden_size=None):
        super(Attention, self)._init(
            word_contextualizer=None,
            word_comparator='decomposable-attention',
            word_aggregator='divsqrt')


class Hybrid(dm.AttrSummarizer):

    def _init(self, hidden_size=None):
        super(Hybrid, self)._init(
            word_contextualizer='gru',
            word_comparator=dm.word_comparators.Attention(
                alignment_network='decomposable', raw_alignment=True),
            word_aggregator='concat-attention-with-rnn')
