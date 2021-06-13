r"""
The deepmatcher package contains high level modules used in the construction of deep
learning modules for entity matching.
"""

import logging
import warnings
import sys

from .data import process as data_process
from .models import modules
from .models.core import (MatchingModel, AttrSummarizer, WordContextualizer,
                          WordComparator, WordAggregator, Classifier)
from .models import (attr_summarizers, word_aggregators, word_comparators,
                     word_contextualizers)

# Register these as submodules of deepmatcher. This helps organize files better while
# permitting an easier way to access these modules.
sys.modules['deepmatcher.attr_summarizers'] = attr_summarizers
sys.modules['deepmatcher.word_contextualizers'] = word_contextualizers
sys.modules['deepmatcher.word_comparators'] = word_comparators
sys.modules['deepmatcher.word_aggregators'] = word_aggregators
sys.modules['deepmatcher.modules'] = modules

warnings.filterwarnings('always', module='deepmatcher')

logging.basicConfig()
logging.getLogger('deepmatcher.data.field').setLevel(logging.INFO)


def process(*args, **kwargs):
    warnings.warn('"deepmatcher.process" is deprecated and will be removed in a later '
                  'release, please use "deepmatcher.data.process" instead',
                  DeprecationWarning)
    return data_process(*args, **kwargs)


__version__ = '0.1.2.post2'
__author__ = 'Sidharth Mudgal, Han Li'

__all__ = [
    'attr_summarizers', 'word_aggregators', 'word_comparators', 'word_contextualizers',
    'process', 'MatchingModel', 'AttrSummarizer', 'WordContextualizer', 'WordComparator',
    'WordAggregator', 'Classifier', 'modules'
]

_check_nan = True


def disable_nan_checks():
    _check_nan = False


def enable_nan_checks():
    _check_nan = True
