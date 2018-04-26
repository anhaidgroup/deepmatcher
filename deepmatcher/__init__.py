r"""
The deepmatcher package contains high level modules used in the construction of deep
learning modules for entity matching.
"""

import logging
import warnings

from .data import process as data_process
from .models import modules
from .models.core import (MatchingModel, AttrSummarizer, AttrComparator,
                          WordContextualizer, WordComparator, WordAggregator, Classifier)
from .models import (attr_summarizers, word_aggregators, word_comparators,
                     word_contextualizers)

warnings.filterwarnings('always', module='deepmatcher')

logging.basicConfig()
logging.getLogger('deepmatcher.data.field').setLevel(logging.INFO)


def process(*args, **kwargs):
    warnings.warn('"deepmatcher.process" is deprecated and will be removed in a later '
                  'release, please use "deepmatcher.data.process" instead',
                  DeprecationWarning)
    return data_process(*args, **kwargs)


__version__ = '0.0.1a0'

__all__ = [
    attr_summarizers, word_aggregators, word_comparators, word_contextualizers, process,
    MatchingModel, AttrSummarizer, AttrComparator, WordContextualizer, WordComparator,
    WordAggregator, Classifier, modules
]

_check_nan = True


def disable_nan_checks():
    _check_nan = False


def enable_nan_checks():
    _check_nan = True
