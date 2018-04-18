r"""
The deepmatcher package contains high level modules used in the construction of deep
learning modules for entity matching. It also contains data processing utilities.
"""

from .data import process
from .models import modules
from .models.core import (MatchingModel, AttrSummarizer, AttrComparator,
                          WordContextualizer, WordComparator, WordAggregator, Classifier)
from .models import (attr_summarizers, word_aggregators, word_comparators,
                     word_contextualizers)

__all__ = [
    attr_summarizers, word_aggregators, word_comparators, word_contextualizers, process,
    MatchingModel, AttrSummarizer, AttrComparator, WordContextualizer,
    WordComparator, WordAggregator, Classifier, modules
]

_check_nan = True


def disable_nan_checks():
    _check_nan = False


def enable_nan_checks():
    _check_nan = True
