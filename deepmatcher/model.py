## Module dm:
class MatchingModel:
  def __init__(attr_summarizer,
               attr_comparator,
               classifier):
    """
    Create a Hybrid Entity Matching Model (see arxiv.org/...)

    Args:
      attr_summarizer (dm.AttrSummarizer):
        The neural network to summarize a sequence of words into a
        vector. Defaults to `dm.attr_summarizers.Hybrid()`.
      attr_comparator (dm.AttrComparator):
        The neural network to compare two attribute summary vectors.
        Default is selected based on `attr_summarizer` choice.
      classifier (dm.Classifier):
        The neural network to perform match / mismatch classification
        based on attribute similarity representations.
        Defaults to `dm.Classifier()`.
    """

class AttrSummarizer:
  def __init__(word_contextualizer,
               word_comparator,
               word_aggregator):
    """
    Create an Attribute Summarization Module.

    Args:
      word_contextualizer (dm.WordContextualizer or list or list of lists):
        The neural network to process an input word sequence to
        consider word sequence into account. If the input is a list of
        `dm.WordContextualizer`s, each of these are applied to the word
        sequences and the result is concatenated along the last dimension.
        If the input is a list of lists of `dm.WordContextualizer`s,
        each of the word_contextualizers in the inner lists are stacked, i.e.,
        run one after the other, and the output of each list of
        word_contextualizers is then concatenated along the last
        dimension. Defaults to None.
      word_comparator (dm.WordComparator or list):
        The neural network to compare each word in one word sequence
        to words in another word sequence. If a list of
        `dm.WordComparator`s is specified, each of these are applied to
        the word sequences and the result is concatenated along the last
        dimension. Defaults to None.
      word_aggregator (dm.WordAggregator or list):
        The neural network to aggregate a sequence of word
        context / comparison vectors. If a list of
        `dm.WordAggregator`s is specified, each of these are applied to
        the word sequences and the result is concatenated along the last
        dimension.
        Defaults to `dm.WordAggregator.Pool(type='average')`.
    """

## Module dm.word_contextualizers:
class CNN(dm.WordContextualizer):
  ...
class RNN(dm.WordContextualizer):
  ...
class SelfAttention(dm.WordContextualizer):
  ...

## Module dm.word_comparators:
class MultiplicativeAttention(dm.WordComparator):
  ...
class AdditiveAttention(dm.WordComparator):
  ...

## Future Additions to dm.word_comparators:
class MultiPerspectiveMatching(dm.WordComparator):
  ...
class CafeAttention(dm.WordComparator):
  ...

## Module dm.word_aggregators:
class Pool(dm.WordAggregator):
  ...
class CNN(dm.WordAggregator):
  ...
class RNN(dm.WordAggregator):
  ...
class AttentionWithRNN(dm.WordAggregator):
  ...

## Module dm.attr_summarizers:
class SIF(dm.AttrSummarizer):
def __init__(self):
super().__init__(
  word_aggregator=dm.word_aggregators.Pool(type='sum_freq_weighted'))

class RNN(dm.AttrSummarizer):
def __init__(self):
super().__init__(
  word_contextualizer=dm.word_contextualizers.RNN(),
  word_aggregator=dm.word_aggregators.Last())

class Attention(dm.AttrSummarizer):
def __init__(self):
super().__init__(
  word_comparator=dm.word_comparators.MultiplicativeAttention(),
  word_aggregator=dm.word_aggregators.Pool(type='sum_length_normalized'))

class Hybrid(dm.AttrSummarizer):
def __init__(self):
super().__init__(
  word_contextualizer=dm.word_contextualizers.RNN(),
  word_comparator=dm.word_comparators.MultiplicativeAttention(
      raw_word_alignment=True),
  word_aggregator=dm.word_aggregators.AttentionWithRNN())

## Sample Usage:
contextualizer1 = dm.word_contextualizers.RNN(layers=2,
    hidden_sizes=[512, 512], dropout_before=0.2, dropout_between_layers=0.2,
    residual=True)
contextualizer2 = dm.word_contextualizers.CNN(layers=2,
    filters=[1024, 512], kernel_sizes=[3, 3])
comparator1 = dm.word_comparators.MultiplicativeAttention()
comparator2 = dm.word_comparators.AdditiveAttention(layers=2,
    hidden_sizes=[1024, 512])
aggregator = dm.word_aggregators.RNN()

my_fancy_attr_summarizer = dm.AttrSummarizer(
    word_contextualizer=[contextualizer1, contextualizer2],
    word_comparator=[comparator1, comparator2],
    word_aggregator=aggregator)

    my_fancy_attr_summarizer = dm.AttrSummarizer(
        word_contextualizer=dm.f('rnn', layers=2),
        word_comparator=[comparator1, comparator2],
        word_aggregator=aggregator)
