"""The deepmatcher package contains high level modules used in the construction of deep learning modules for entity matching."""

import logging
import warnings

from deepmatcher.data import process as data_process
from deepmatcher.models import (
    attr_summarizers,
    modules,
    word_aggregators,
    word_comparators,
    word_contextualizers,
)
from deepmatcher.models.core import (
    AttrSummarizer,
    Classifier,
    MatchingModel,
    WordAggregator,
    WordComparator,
    WordContextualizer,
)

warnings.filterwarnings("always", module="deepmatcher")

logging.basicConfig()
logging.getLogger("deepmatcher.data.field").setLevel(logging.INFO)


def process(*args, **kwargs):
    warnings.warn(
        '"deepmatcher.process" is deprecated and will be removed in a later '
        'release, please use "deepmatcher.data.process" instead',
        DeprecationWarning,
    )
    return data_process(*args, **kwargs)


__version__ = "0.1.0.post1"
__author__ = "Sidharth Mudgal, Han Li"

__all__ = [
    "attr_summarizers",
    "word_aggregators",
    "word_comparators",
    "word_contextualizers",
    "process",
    "MatchingModel",
    "AttrSummarizer",
    "WordContextualizer",
    "WordComparator",
    "WordAggregator",
    "Classifier",
    "modules",
]

_check_nan = True
