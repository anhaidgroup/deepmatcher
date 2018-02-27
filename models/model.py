import torch


## Module dm:
class MatchingModel:

    def __init__(attr_summarizer, attr_comparator, classifier):
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
