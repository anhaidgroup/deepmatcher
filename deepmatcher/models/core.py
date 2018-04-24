import copy
import pdb
from collections import Iterable, Mapping

import six

import deepmatcher as dm
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import _utils
from ..batch import AttrTensor
from ..data import MatchingDataset, MatchingIterator
from ..runner import Runner
from ..utils import Bunch


class MatchingModel(nn.Module):
    r"""A neural network model for entity matching.

    This network consists of the following components:

    #. Attribute Summarizers

    #. Attribute Comparators

    #. A Classifier

    Creating a MatchingModel instance does not immediately construct the neural network.
    The network will be constructed just before training based on metadata from the
    training set:

    #. For each attribute (e.g., Product Name, Address, etc.), an Attribute Summarizer is
       constructed using specified `attr_summarizer` template.
    #. For each attribute, an Attribute Comparator is constructed using specified
       `attr_summarizer` template.
    #. A Classifier is constructed based on the `classifier` template.

    Args:
        attr_summarizer (:class:`AttrSummarizer`):
            The neural network to summarize an attribute value of a tuple (a sequence of
            words) into a vector. Defaults to :obj:`~deepmatcher.attr_summarizers.Hybrid`,
            which is the hybrid model. Please consult the paper for more information.
        attr_comparator (:class:`AttrComparator`):
            The neural network to compare two attribute summary vectors.
            Default is selected based on `attr_summarizer` choice.
        attr_condense_factor (string or int):
            The attribute condensing factor that controls the condensing ratio from the
            attribute summarization vector to the corresponding classifier input. The
            purpose of condensing the attribute summarization vector is to reduce the size
            of input vector to the classifier to reduce the number of parameters. The
            default value is 'auto'. If the default value 'auto' is used, the actual
            condens factor is the minimal value of the number attributes (excluding id and
            label columns) or 6.
        attr_merge (string):
            The way to merge the attribute summarization vectors to generate the input
            vector of the classifier. The default value is 'concat', which is to
            concatenate the condensed vector of each attribute summarization.
        classifier (:class:`Classifier`):
            The neural network to perform match / mismatch classification
            based on attribute similarity representations.
            Defaults to :class:`Classifier`.
        hidden_size (int):
            TODO(Sid)
    """

    def __init__(self,
                 attr_summarizer='hybrid',
                 attr_condense_factor='auto',
                 attr_comparator=None,
                 attr_merge='concat',
                 classifier='2-layer-highway',
                 hidden_size=300):

        super(MatchingModel, self).__init__()

        self.attr_summarizer = attr_summarizer
        self.attr_condense_factor = attr_condense_factor
        self.attr_comparator = attr_comparator
        self.attr_merge = attr_merge
        self.classifier = classifier

        self.hidden_size = hidden_size
        self._train_buffers = set()
        self._initialized = False

    def initialize(self, train_dataset, init_batch=None):
        r"""
        Initialize (not lazily) the matching model given the actual training data.

        Args:
            train_dataset (:class:`data.MatchingDataset`): the training dataset.
        """

        if self._initialized:
            return

        self.meta = train_dataset

        # Copy over training info from train set for persistent state. But remove actual
        # data examples.
        self.register_train_buffer('state_meta', Bunch(**train_dataset.__dict__))
        del self.state_meta.metadata  # we only need `self.meta.orig_metadata`
        if hasattr(self.state_meta, 'fields'):
            del self.state_meta.fields
            del self.state_meta.examples

        self.attr_summarizers = dm.modules.ModuleMap()
        if isinstance(self.attr_summarizer, Mapping):
            for name, summarizer in self.attr_summarizer.items():
                self.attr_summarizers[name] = AttrSummarizer.create(
                    summarizer, hidden_size=self.hidden_size)
            assert len(
                set(self.attr_summarizers.keys()) ^ set(self.meta.canonical_text_fields)
            ) == 0
        else:
            self.attr_summarizer = AttrSummarizer.create(
                self.attr_summarizer, hidden_size=self.hidden_size)
            for name in self.meta.canonical_text_fields:
                self.attr_summarizers[name] = copy.deepcopy(self.attr_summarizer)

        if self.attr_condense_factor == 'auto':
            self.attr_condense_factor = min(len(self.meta.canonical_text_fields), 6)
            if self.attr_condense_factor == 1:
                self.attr_condense_factor = None

        if not self.attr_condense_factor:
            self.attr_condensors = None
        else:
            self.attr_condensors = dm.modules.ModuleMap()
            for name in self.meta.canonical_text_fields:
                self.attr_condensors[name] = dm.modules.Transform(
                    '1-layer-highway',
                    non_linearity=None,
                    output_size=self.hidden_size // self.attr_condense_factor)

        self.attr_comparators = dm.modules.ModuleMap()
        if isinstance(self.attr_comparator, Mapping):
            for name, comparator in self.attr_comparator.items():
                self.attr_comparators[name] = AttrComparator.create(comparator)
            assert len(
                set(self.attr_comparators.keys()) ^ set(self.meta.canonical_text_fields)
            ) == 0
        else:
            if isinstance(self.attr_summarizer, AttrSummarizer):
                self.attr_comparator = self.get_attr_comparator(
                    self.attr_comparator, self.attr_summarizer)
            self.attr_comparator = AttrComparator.create(self.attr_comparator)
            for name in self.meta.canonical_text_fields:
                self.attr_comparators[name] = copy.deepcopy(self.attr_comparator)

        self.attr_merge = dm.modules._merge_module(self.attr_merge)
        self.classifier = _utils.get_module(
            Classifier, self.classifier, hidden_size=self.hidden_size)

        self.reset_embeddings(train_dataset.vocabs)

        # Instantiate all components using a small batch from training set.
        if not init_batch:
            run_iter = MatchingIterator(
                train_dataset,
                train_dataset,
                train=False,
                batch_size=4,
                device=-1,
                sort_in_buckets=False)
            init_batch = next(run_iter.__iter__())
        self.forward(init_batch)

        # Keep this init_batch for future initializations.
        self.state_meta.init_batch = init_batch

        self._initialized = True

    def reset_embeddings(self, vocabs):
        self.embed = dm.modules.ModuleMap()
        field_vectors = {}
        for name in self.meta.all_text_fields:
            vectors = vocabs[name].vectors
            if vectors not in field_vectors:
                vectors_size = vectors.shape
                embed = nn.Embedding(vectors_size[0], vectors_size[1])
                embed.weight.data.copy_(vectors)
                embed.weight.requires_grad = False
                field_vectors[vectors] = dm.modules.NoMeta(embed)
            self.embed[name] = field_vectors[vectors]

    def get_attr_comparator(self, arg, attr_summarizer):
        r"""
        Get the attribute comparator.
        Args:
        arg (string):
            The attribute comparator to use. Can be one of the supported style arguments
            for :class:`~modules.Merge`, specifically, 'abs-diff', 'diff', 'concat',
            'concat-diff', 'concat-abs-diff', or 'mul'. If not specified uses 'abs-diff'
            for SIF and RNN, 'concat' for Attention, and 'concat-diff' for Hybrid".
        attr_summarizer (:obj:`AttrSummarizer`):
            The attribute summarizer object in :module:`attr_summarizers` (i.e., it
            should be one of "SIF", "RNN", "Attention", and "Hybrid" defined in
            :module:`attr_summarizers`).

        Returns:
            string: Return the type of attribute comparator.
        """
        if arg is not None:
            return arg
        if isinstance(attr_summarizer, dm.attr_summarizers.SIF):
            return 'abs-diff'
        elif isinstance(attr_summarizer, dm.attr_summarizers.RNN):
            return 'abs-diff'
        elif isinstance(attr_summarizer, dm.attr_summarizers.Attention):
            return 'concat'
        elif isinstance(attr_summarizer, dm.attr_summarizers.Hybrid):
            return 'concat-abs-diff'
        raise ValueError('Cannot infer attr comparator, please specify.')

    def forward(self, input):
        r"""
        The forward function of the matching model.

        Args:
        input ():
            TODO(Sid)
        """
        embeddings = {}
        for name in self.meta.all_text_fields:
            attr_input = getattr(input, name)
            embeddings[name] = self.embed[name](attr_input)

        attr_comparisons = []
        for name in self.meta.canonical_text_fields:
            left, right = self.meta.text_fields[name]
            left_summary, right_summary = self.attr_summarizers[name](embeddings[left],
                                                                      embeddings[right])
            left_summary, right_summary = left_summary.data, right_summary.data
            if self.attr_condensors:
                left_summary = self.attr_condensors[name](left_summary)
                right_summary = self.attr_condensors[name](right_summary)
            attr_comparisons.append(self.attr_comparators[name](left_summary,
                                                                right_summary))

        entity_comparison = self.attr_merge(*attr_comparisons)
        return self.classifier(entity_comparison)

    def register_train_buffer(self, name, value):
        self._train_buffers.add(name)
        setattr(self, name, value)

    def save_state(self, path, include_meta=True):
        r"""
        Save the model state to a certain path.

        Args:
        path (string): The path to save the model state.
        """
        state = {'model': self.state_dict()}
        for k in self._train_buffers:
            if include_meta or k != 'state_meta':
                state[k] = getattr(self, k)
        torch.save(state, path)

    def load_state(self, path):
        r"""
        Load the model state from a file in a certain path.

        Args:
        path (string): The path to load the model state.
        """
        state = torch.load(path)
        for k, v in six.iteritems(state):
            if k != 'model':
                self._train_buffers.add(k)
                setattr(self, k, v)

        if hasattr(self, 'state_meta'):
            train_info = copy.copy(self.state_meta)

            # Handle metadata manually.
            # TODO (Sid): Make this cleaner.
            train_info.metadata = train_info.orig_metadata
            MatchingDataset.finalize_metadata(train_info)

            self.initialize(train_info, self.state_meta.init_batch)

        self.load_state_dict(state['model'])

    def run_train(self, *args, **kwargs):
        return Runner.train(self, *args, **kwargs)

    def run_eval(self, *args, **kwargs):
        return Runner.eval(self, *args, **kwargs)

    def run_prediction(self, *args, **kwargs):
        return Runner.predict(self, *args, **kwargs)

    # def train_mode(self):
    #     super(MatchingModel, self).train()
    #
    # def eval_mode(self):
    #     super(MatchingModel, self).eval()


class AttrSummarizer(dm.modules.LazyModule):
    r"""
    The attribute summarizer that will summarize the input sequence of an attribute value.

    Args:
    word_contextualizer (string or :module:`word_contextualizers` or callable):
        The word contextualizer to process an input word sequence to consider word
        sequence into account. For details please refer to :class:`WordContextualizer`.
    word_comparator (string or :module:`word_comparators` or callable):
        The word comparator that will be used in the attention step to compare the a word
        with the corresponding alignment in the other sequence. For details please refer
        to :class:`WordComparator`.
    word_aggregator (string or :module:`word_aggregators` or callable):
        The word aggregator to aggregate a sequence of word context / comparison vectors.
        For details please refer to :class:`WordAggregator`.
    hidden_size (int):
        The hidden size used for the three sub-modules (i.e., word_contextualizer,
        word_comparator, and word_aggregator).
    """

    def _init(self,
              word_contextualizer,
              word_comparator,
              word_aggregator,
              hidden_size=None):
        self.word_contextualizer = WordContextualizer.create(
            word_contextualizer, hidden_size=hidden_size)
        self.word_comparator = WordComparator.create(
            word_comparator, hidden_size=hidden_size)
        self.word_aggregator = WordAggregator.create(
            word_aggregator, hidden_size=hidden_size)

    def _forward(self, left_input, right_input):
        r"""
        The forward function of attribute summarizer.

        Args:
        left_input ():
            TODO(Sid)
        right_input ():
            TODO(Sid)
        """
        left_contextualized, right_contextualized = left_input, right_input
        if self.word_contextualizer:
            left_contextualized = self.word_contextualizer(left_input)
            right_contextualized = self.word_contextualizer(right_input)

        left_compared, right_compared = left_contextualized, right_contextualized
        if self.word_comparator:
            left_compared = self.word_comparator(
                left_contextualized, right_contextualized, left_input, right_input)
            right_compared = self.word_comparator(
                right_contextualized, left_contextualized, right_input, left_input)

        left_aggregator_context = right_input
        right_aggregator_context = left_input

        left_aggregated = self.word_aggregator(left_compared, left_aggregator_context)
        right_aggregated = self.word_aggregator(right_compared, right_aggregator_context)
        return left_aggregated, right_aggregated

    @classmethod
    def create(cls, arg, **kwargs):
        r"""
        Create an attribute summ object.

        Args:
        arg (string or :module:`attr_summarizers` or callable):
            The argument for creating the attribute summarizer. It can be one of:
            * a string specifying the attribute summarizer to use ("sif", "rnn",
                "attention", "hybrid").
            * a :class:`~dm.AttrSummarizer` object.
            * a callable that returns a :class:`nn.Module`.
        **kwargs:
            Keyword arguments to the constructor of the AttrSummarizer sub-class.
            For details on what these can be, please refer to the documentation of the
            sub-classes in :module:`attr_summarizers`.
        """
        assert arg is not None
        if isinstance(arg, six.string_types):
            type_map = {
                'sif': dm.attr_summarizers.SIF,
                'rnn': dm.attr_summarizers.RNN,
                'attention': dm.attr_summarizers.Attention,
                'hybrid': dm.attr_summarizers.Hybrid
            }
            if arg in type_map:
                asr = type_map[arg](**kwargs)
            else:
                raise ValueError('Unknown Attribute Summarizer name.')
        else:
            asr = _utils.get_module(AttrSummarizer, arg)

        asr.expect_signature('[AxBxC, AxDxC] -> [AxE, AxE]')
        return asr


class AttrComparator(dm.modules.LazyModule):
    r"""The attribute comparator that will compare the two summarizations of the same
    attribute in a tuple pair and generate a hidden representation of the comparison
    result.
    """

    @classmethod
    def create(cls, arg):
        r"""Create an attribute comparator object.

        Args:
        arg (string):
            The argument for creating an attribute comparator object. It can be one of the
            following strings: "concat", "diff", "abs-diff", "concat-diff",
            "concat-abs-diff", "mul".
        """
        assert arg is not None
        module = dm.modules._merge_module(arg)
        module.expect_signature('[AxB, AxB] -> [AxC]')
        return module


class WordContextualizer(dm.modules.LazyModule):
    r"""The neural network to process an input word sequence to consider word
    sequence into account.
    """

    @classmethod
    def create(cls, arg, **kwargs):
        r"""Create a word contextualizer object.

        Args:
        arg (string or :module:`word_contextualizers` or callable):
            The argument for creating a word contextualizer object. It can be one of:
            * a string specifying the word contextualizer to use. It can be one of:
                * RNN-based contextualizer ("rnn", "gru", "lstm").
                * Self-attention-based contextualizer ("selfattention").
            * a :class:`~dm.WordContextualizer` object.
            * a callable that returns a :class:`nn.Module`.
        **kwargs:
            Keyword arguments to the constructor of the WordContextualizer sub-class.
            For details on what these can be, please refer to the documentation of the
            sub-classes in :module:`word_contextualizers`.
        """
        if isinstance(arg, six.string_types):
            if dm.word_contextualizers.RNN.supports_style(arg):
                wc = dm.word_contextualizers.RNN(arg, **kwargs)
            elif arg == 'selfattention':
                wc = dm.word_contextualizers.SelfAttention(**kwargs)
            else:
                raise ValueError('Unknown Word Contextualizer name.')
        else:
            wc = _utils.get_module(WordContextualizer, arg)

        if wc is not None:
            wc.expect_signature('[AxBxC] -> [AxBxD]')

        return wc


class WordComparator(dm.modules.LazyModule):
    r"""The neural network that will be used in the attention step to compare the a word
    with the corresponding alignment in the other sequence.
    """

    @classmethod
    def create(cls, arg, **kwargs):
        r"""Create a word comparator object.

        Args:
        arg (string or :module:`word_comparators` or callable):
            The argument for creating a word comparator object. It can be one of:
            * a string specifying the word comparator to use. For now we support
              different types of attention-based comparison ("dot-attention",
              "bilinear-attention", "decomposable-attention", "concat-attention",
              "concat_dot-attention").
            * a :class:`~dm.WordComparator` object.
            * a callable that returns a :class:`nn.Module`.
        **kwargs:
            Keyword arguments to the constructor of the WordComparator sub-class.
            For details on what these can be, please refer to the documentation of the
            sub-classes in :module:`word_comparators`.
        """
        if isinstance(arg, six.string_types):
            parts = arg.split('-')
            if (parts[1] == 'attention' and
                    dm.modules.AlignmentNetwork.supports_style(parts[0])):
                wc = dm.word_comparators.Attention(alignment_network=parts[0], **kwargs)
            else:
                raise ValueError('Unknown Word Comparator name.')
        else:
            wc = _utils.get_module(WordComparator, arg)

        if wc is not None:
            wc.expect_signature('[AxBxC, AxDxC, AxBxE, AxDxE] -> [AxBxF]')

        return wc


class WordAggregator(dm.modules.LazyModule):
    r"""
    The neural network to aggregate a sequence of word context / comparison vectors.
    """

    @classmethod
    def create(cls, arg, **kwargs):
        r"""
        Create a word aggregator object.

        Args:
        arg (string or :module:`word_aggregators` or callable):
            The argument for creating a word aggregator object. It can be one of:
            * a string specifying the word aggregator to use. It can be one of the following:
                * pool aggregators ("avg-pool", "divsqrt-pool", "inv-freq-avg-pool",
                  "sif-pool", "max-pool", "last-pool", "last-simple-pool",
                  "birnn-last-pool", "birnn-last-simple-pool").
                * attention with rnn ("attention-with-rnn").
            * a :class:`~dm.WordAggregator` object.
            * a callable that returns a :class:`nn.Module`. This module should take in a
              tensor of shape (batch,  seq_len, input_size) and return a tensor of shape
              (batch, input_size).
            * a callable that returns a :class:`nn.Module`. This module should take two
              arguments, a primary input tensor of shape (batch, primary_len, input_size)
              and a context input tensor of shape (batch, context_len, input_size). It
              should aggregate the word sequence in the primary input and return a tensor
              of shape (batch, input_size). Your module can ignore the context input while
              aggregating the primary input, if you think the context is not helpful for
              the primary input aggregation.
        **kwargs:
            Keyword arguments to the constructor of the WordAggregator sub-class.
            For details on what these can be, please refer to the documentation of the
            sub-classes in :module:`word_aggregators`.
        """
        assert arg is not None
        if isinstance(arg, six.string_types):
            parts = arg.split('-')
            if (parts[-1] == 'pool' and
                    dm.word_aggregators.Pool.supports_style('-'.join(parts[:-1]))):
                seq = []
                seq.append(dm.modules.Lambda(lambda x1, x2: x1))  # Ignore the context.
                seq.append(dm.word_aggregators.Pool(style='-'.join(parts[:-1])))

                # Make lazy module.
                wa = dm.modules.LazyModuleFn(lambda: dm.modules.MultiSequential(*seq))
            elif arg == 'attention-with-rnn':
                wa = dm.word_aggregators.AttentionWithRNN(**kwargs)
            else:
                raise ValueError('Unknown Word Aggregator name.')
        else:
            wa = _utils.get_module(WordAggregator, arg)

        wa.expect_signature('[AxBxC] -> [AxD]')
        return wa


class Classifier(nn.Sequential):
    r"""
    The classifier network to predict a tuple pair match or not given the attribute
    summarizations.

    Args:
    transform_network (string or :class:`~modules.Transform` or callable):
        The transformation network to transform the input vector of the classifier to a
        hidden representation with the size of `hidden_size`.
    hidden_size (int):
        The size of the hidden representation generated by the transformation network.
    """

    def __init__(self, transform_network, hidden_size=None):
        super(Classifier, self).__init__()
        if transform_network:
            self.add_module('transform',
                            dm.modules._transform_module(transform_network, hidden_size))
        self.add_module('softmax_transform',
                        dm.modules.Transform(
                            '1-layer', non_linearity=None, output_size=2))
        self.add_module('softmax', nn.LogSoftmax(dim=1))
