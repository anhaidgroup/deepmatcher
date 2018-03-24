class MatchingModel:
    def __init__(attr_summarizer,
                 attr_comparator,
                 classifier):
        """
        Create a Hybrid Entity Matching Model (see arxiv.org/...)

        Args:
            attr_summarizer (str or dm.AttrSummarizer):
                The neural network to summarize a sequence of words into a
                vector. One of 'sif', 'rnn', 'attention', 'hybrid', or a
                dm.AttrSummarizer. Defaults to 'hybrid'.
            attr_comparator (str or dm.AttrComparator):
                The neural network to compare two attribute summary vectors.
                One of 'euclidean', 'cosine', 'concat', 'diff', 'concat_diff',
                or a dm.AttrComparator. Default is selected based on
                `attr_summarizer` choice.
            classifier (dm.Classifier):
                The neural network to perform match / mismatch classification
                based on attribute similarity representations.
        """

class AttrSummarizer:
    def __init__(word_contextualizer,
                 word_comparator,
                 word_aggregator):
        """
        Create an Attribute Summarization Module.

        Args:
            word_contextualizer (str or dm.WordContextualizer):
                The neural network to process an input word sequence to
                consider word sequence into account. One of 'cnn', 'rnn',
                'self-alignment', or a dm.WordContextualizer. Defaults to
                None.
            word_comparator (str or dm.WordComparator):
                The neural network to compare each word in one word sequence
                to words in another word sequence. One of
                'dot_product_attention', 'additive_attention', or a
                dm.WordComparator. Defaults to None.
            word_aggregator (str or dm.WordAggregator):
                The neural network to aggregate a sequence of word
                context / comparison vectors. One of 'pool', 'cnn', 'rnn',
                'attention_rnn', 'last', or dm.WordAggregator.
                Defaults to 'pool'.
        """

# Module dm.attr_summarizers:
class SIF(dm.AttrSummarizer):
    def __init__(self):
        super(SIF, self).__init__(
            word_aggregator=dm.word_aggregators.Pool(type='freq_weighted_sum'))

class RNN(dm.AttrSummarizer):
    def __init__(self):
        super(RNN, self).__init__(
            word_contextualizer='rnn',
            word_aggregator=dm.word_aggregators.Pool(type='freq_weighted_sum'))
    ...
class Attention(dm.AttrSummarizer):
    ...
class Hybrid(dm.AttrSummarizer):
    ...

## Module dm.word_contextualizers:
class CNN(dm.WordContextualizer):
    ...
class RNN(dm.WordContextualizer):
    ...
class SelfAttention(dm.WordContextualizer):
    ...

## Module dm.word_comparators:
class DotProductAttention(dm.WordComparator):
    ...
class AdditiveAttention(dm.WordComparator):
    ...

## Module dm.word_aggregators:
class Pool(dm.WordAggregator):
    ...
class CNN(dm.WordAggregator):
    ...
class RNN(dm.WordAggregator):
    ...
class AttentionRNN(dm.WordAggregator):
    ...
