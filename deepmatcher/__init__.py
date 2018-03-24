from .data import process
from .optim import Optimizer
from .runner import Statistics
from .models import modules
from .models.core import (MatchingModel, AttrSummarizer, AttrComparator,
                          WordContextualizer, WordComparator, WordAggregator)
from .models import (attr_summarizers, word_aggregators, word_comparators,
                     word_contextualizers)

__all__ = [
    attr_summarizers, word_aggregators, word_comparators,
    word_contextualizers, process, Optimizer, Statistics, MatchingModel, AttrSummarizer,
    AttrComparator, WordContextualizer, WordComparator, WordAggregator, modules
]

_check_nan = True


def disable_nan_checks():
    _check_nan = False


def enable_nan_checks():
    _check_nan = True
