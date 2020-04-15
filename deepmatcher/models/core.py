import copy
import logging
from collections import Mapping

import dill
import six

import deepmatcher as dm
import torch
import torch.nn as nn

from . import _utils
from ..data import MatchingDataset, MatchingIterator
from ..runner import Runner
from ..utils import Bunch, tally_parameters

logger = logging.getLogger('deepmatcher.core')


class MatchingModel(nn.Module):
    r"""A neural network model for entity matching.

    Refer to the
    `Matching Models tutorial <https://nbviewer.jupyter.org/github/sidharthms/deepmatcher/blob/master/examples/matching_models.ipynb>`_
    for details on how to customize a `MatchingModel`. A brief intro is below:

    This network consists of the following components:

    #. Attribute Summarizers

    #. Attribute Comparators

    #. A Classifier

    Creating a MatchingModel instance does not immediately construct the neural network.
    The network will be constructed just before training based on metadata from the
    training set:

    #. For each attribute (e.g., Product Name, Address, etc.), an Attribute Summarizer is
       constructed using the specified `attr_summarizer` template.
    #. For each attribute, an Attribute Comparator is constructed using the specified
       `attr_summarizer` template.
    #. A Classifier is constructed based on the specified `classifier` template.

    Args:
        attr_summarizer (string or :class:`AttrSummarizer` or callable):
            The attribute summarizer. Takes in two word embedding sequences and summarizes
            the information in them to produce two summary vectors as output.
            Options `listed here <https://nbviewer.jupyter.org/github/sidharthms/deepmatcher/blob/master/examples/matching_models.ipynb#attr_summarizer-can-be-set-to-one-of-the-following:>`__.
            Defaults to 'hybrid', i.e., the :class:`~deepmatcher.attr_summarizers.Hybrid`
            model.
        attr_comparator (string or :class:`~deepmatcher.modules.Merge` or callable):
            The attribute comparator. Takes as input the two summary vectors and applies a
            comparison function over those summaries to obtain the final similarity
            representation of the two attribute values. Argument must specify a
            :ref:`merge-op` operation. Default is selected based on `attr_summarizer`
            choice.
        attr_condense_factor (string or int):
            The factor by which to condense each attribute similarity vector. E.g. if
            `attr_condense_factor` is set to 3 and the attribute similarity vector size is
            300, then each attribute similarity vector is transformed to a 100 dimensional
            vector using a linear transformation. The purpose of condensing is to reduce
            the number of parameters in the classifier module. This parameter can be set
            to a number or 'auto'. If 'auto', then the condensing factor is set to be
            equal to the number attributes, but if there are more than 6 attributes, then
            the condensing factor is set to 6. Defaults to 'auto'.
        attr_merge (string or :class:`~deepmatcher.modules.Merge` or callable):
            The operation used to merge the (optionally condensed) attribute similarity
            vectors to obtain the input to the classifier. Argument must specify a
            :ref:`merge-op` operation. Defaults to 'concat', i.e., concatenate all
            attribute similarity vectors to form the classifier input.
        classifier (string or :class:`Classifier` or callable):
            The neural network to perform match / mismatch classification
            based on attribute similarity representations.
            Options `listed here <https://nbviewer.jupyter.org/github/sidharthms/deepmatcher/blob/master/examples/matching_models.ipynb#classifier-can-be-set-to-one-of-the-following:>`__.
            Defaults to '2-layer-highway', i.e., use a two layer highway network followed
            by a softmax layer for classification.
        hidden_size (int):
            The hidden size to use for the `attr_summarizer` and the `classifier` modules,
            if they are string arguments. If a module or :attr:`callable` input is specified
            for a component, this argument is ignored for that component.
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

    def run_train(self, *args, **kwargs):
        """run_train(train_dataset, validation_dataset, best_save_path,epochs=30, \
            criterion=None, optimizer=None, pos_neg_ratio=None, pos_weight=None, \
            label_smoothing=0.05, save_every_prefix=None, save_every_freq=None, \
            batch_size=32, device=None, progress_style='bar', log_freq=5, \
            sort_in_buckets=None)

        Train the model using the specified training set.

        Args:
            train_dataset (:class:`~deepmatcher.data.MatchingDataset`):
                The training dataset obtained using :func:`deepmatcher.data.process`.
            validation_dataset (:class:`~deepmatcher.data.MatchingDataset`):
                The validation dataset obtained using :func:`deepmatcher.data.process`.
                This is used for `early stopping
                <https://en.wikipedia.org/wiki/Early_stopping>`_.
            best_save_path (string):
                The path to save the best model to. At the end of each epoch, if the new
                model accuracy (F1) is better than all previous epochs, then it is saved
                to this location.
            epochs (int):
                Number of training epochs, i.e., number of times to cycle through the
                entire training set. Defaults to 50.
            criterion (:class:`torch.nn.Module`):
                The loss function to use. Refer to the `losses section of the PyTorch API
                <https://pytorch.org/docs/master/nn.html#loss-functions>`_ for options. By
                default, `deepmatcher` will output a 2d tensor of shape (N, C) where N is
                the batch size and C is 2 - the number of classes. Keep this in mind when
                picking the loss. Defaults to :class:`~deepmatcher.optim.SoftNLLLoss` with
                label smoothing.
            optimizer (:class:`~deepmatcher.optim.Optimizer`):
                The optimizer to use for updating the trainable parameters of the
                :class:`~deepmatcher.MatchingModel` neural network after each iteration.
                If not specified an :class:`~deepmatcher.optim.Optimizer` with Adam
                optimizer will be constructed.
            pos_neg_ratio (int):
                The weight of the positive class (match) wrt the negative class
                (non-match). This parameter must be specified if there is a significant
                class imbalance in the dataset.
            label_smoothing (float):
                The `label_smoothing` parameter to constructor of
                :class:`~deepmatcher.optim.SoftNLLLoss` criterion. Only used when
                `criterion` param is None. Defaults to 0.05.
            save_every_prefix (string):
                Prefix of the path to save model to, after end of
                every N epochs, where N is determined by `save_every_freq` param.
                E.g. setting this to "/path/to/saved/model" will save models to
                "/path/to/saved/model_ep1.pth", "/path/to/saved/model_ep2.pth", etc.
                Models will not be saved periodically if this is None. Defaults to
                None.
            save_every_freq (int):
                Determines the frequency (number of epochs) for saving models periodically
                to the path specified by the `save_every_prefix` param (has no effect if
                that param is not set). Defaults to 1.
            batch_size (int):
                Mini-batch size for SGD. For details on what this is
                `see this video <https://www.coursera.org/learn/machine-learning/lecture/9zJUs/mini-batch-gradient-descent>`__.
                Defaults to 32. This is a keyword only param.
            device (string):
                The device on which to train the model ('cpu' or 'cuda'). If None, will use
                first available GPU, or use CPU if no GPUs are available. Defaults to None.
                This is a keyword only param.
            progress_style (string):
                Sets the progress update style. One of 'bar' or 'log'. If 'bar', uses a
                progress bar, updated every N batches. If 'log', prints training stats
                every N batches. N is determined by the `log_freq` param.
                This is a keyword only param.
            log_freq (int):
                Number of batches between progress updates. Defaults to 5.
                This is a keyword only param.
            sort_in_buckets (bool):
                Whether to batch examples of similar lengths together. If True, minimizes
                amount of padding needed while producing freshly shuffled batches for each
                new epoch. Implemented using :func:`torchtext.data.pool`.
                Defaults to True. This is a keyword only param.

        Returns:
            float: The best F1 score obtained by the model on the validation dataset.
        """
        return Runner.train(self, *args, **kwargs)

    def run_eval(self, *args, **kwargs):
        """run_eval(dataset, batch_size=32, device=None, progress_style='bar', \
            log_freq=5, sort_in_buckets=None)

        Evaluate the model on the specified dataset.

        Args:
            dataset (:class:`~deepmatcher.data.MatchingDataset`): The evaluation dataset
                obtained using :func:`deepmatcher.data.process`.
            batch_size (int):
                Mini-batch size for SGD. For details on what this is
                `see this video <https://www.coursera.org/learn/machine-learning/lecture/9zJUs/mini-batch-gradient-descent>`__.
                Defaults to 32. This is a keyword only param.
            device (string):
                The device on which to train the model ('cpu' or 'cuda'). If None, will use
                first available GPU, or use CPU if no GPUs are available. Defaults to None.
                This is a keyword only param.
            progress_style (string):
                Sets the progress update style. One of 'bar' or 'log'. If 'bar', uses a
                progress bar, updated every N batches. If 'log', prints training stats
                every N batches. N is determined by the `log_freq` param.
                This is a keyword only param.
            log_freq (int):
                Number of batches between progress updates. Defaults to 5.
                This is a keyword only param.
            sort_in_buckets (bool):
                Whether to batch examples of similar lengths together. If True, minimizes
                amount of padding needed while producing freshly shuffled batches for each
                new epoch. Implemented using :func:`torchtext.data.pool`.
                Defaults to True. This is a keyword only param.

        Returns:
            float: The F1 score obtained by the model on the dataset.
        """
        return Runner.eval(self, *args, **kwargs)

    def run_prediction(self, *args, **kwargs):
        """run_prediction(dataset, output_attributes=False, batch_size=32, device=None, \
            progress_style='bar', log_freq=5, sort_in_buckets=None)

        Use the model to obtain predictions, i.e., match scores on the specified dataset.

        Args:
            dataset (:class:`~deepmatcher.data.MatchingDataset`): The dataset (labeled or
                not) obtained using :func:`deepmatcher.data.process` or
                :func:`deepmatcher.data.process_unlabeled`.
            output_attributes (bool): Whether to include all attributes in the original
                CSV file of the dataset in the returned pandas table.
            batch_size (int):
                Mini-batch size for SGD. For details on what this is
                `see this video <https://www.coursera.org/learn/machine-learning/lecture/9zJUs/mini-batch-gradient-descent>`__.
                Defaults to 32. This is a keyword only param.
            device (string):
                The device on which to train the model ('cpu' or 'cuda'). If None, will use
                first available GPU, or use CPU if no GPUs are available. Defaults to None.
                This is a keyword only param.
            progress_style (string):
                Sets the progress update style. One of 'bar' or 'log'. If 'bar', uses a
                progress bar, updated every N batches. If 'log', prints training stats
                every N batches. N is determined by the `log_freq` param.
                This is a keyword only param.
            log_freq (int):
                Number of batches between progress updates. Defaults to 5.
                This is a keyword only param.
            sort_in_buckets (bool):
                Whether to batch examples of similar lengths together. If True, minimizes
                amount of padding needed while producing freshly shuffled batches for each
                new epoch. Implemented using :func:`torchtext.data.pool`.
                Defaults to True. This is a keyword only param.


        Returns:
            pandas.DataFrame: A pandas DataFrame containing tuple pair IDs (in the "id"
                column) and the corresponding match score predictions (in the
                "match_score" column). Will also include all attributes in the original
                CSV file of the dataset if `output_attributes` is True.
        """
        return Runner.predict(self, *args, **kwargs)

    def initialize(self, train_dataset, init_batch=None):
        r"""Initialize (not lazily) the matching model given the actual training data.

        Instantiates all sub-components and their trainable parameters.

        Args:
            train_dataset (:class:`~deepmatcher.data.MatchingDataset`):
                The training dataset obtained using :func:`deepmatcher.data.process`.
            init_batch (:class:`~deepmatcher.batch.MatchingBatch`):
                A batch of data to forward propagate through the model. If None, a batch
                is drawn from the training dataset.
        """

        if self._initialized:
            return

        # Copy over training info from train set for persistent state. But remove actual
        # data examples.
        self.meta = Bunch(**train_dataset.__dict__)
        if hasattr(self.meta, 'fields'):
            del self.meta.fields
            del self.meta.examples

        self._register_train_buffer('state_meta', Bunch(**self.meta.__dict__))
        del self.state_meta.metadata  # we only need `self.meta.orig_metadata` for state.

        self.attr_summarizers = dm.modules.ModuleMap()
        if isinstance(self.attr_summarizer, Mapping):
            for name, summarizer in self.attr_summarizer.items():
                self.attr_summarizers[name] = AttrSummarizer._create(
                    summarizer, hidden_size=self.hidden_size)
            assert len(
                set(self.attr_summarizers.keys()) ^ set(self.meta.canonical_text_fields)
            ) == 0
        else:
            self.attr_summarizer = AttrSummarizer._create(
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
                self.attr_comparators[name] = _create_attr_comparator(comparator)
            assert len(
                set(self.attr_comparators.keys()) ^ set(self.meta.canonical_text_fields)
            ) == 0
        else:
            if isinstance(self.attr_summarizer, AttrSummarizer):
                self.attr_comparator = self._get_attr_comparator(
                    self.attr_comparator, self.attr_summarizer)
            else:
                if self.attr_comparator is None:
                    raise ValueError('"attr_comparator" must be specified if '
                                     '"attr_summarizer" is custom.')

            self.attr_comparator = _create_attr_comparator(self.attr_comparator)
            for name in self.meta.canonical_text_fields:
                self.attr_comparators[name] = copy.deepcopy(self.attr_comparator)

        self.attr_merge = dm.modules._merge_module(self.attr_merge)
        self.classifier = _utils.get_module(
            Classifier, self.classifier, hidden_size=self.hidden_size)

        self._reset_embeddings(train_dataset.vocabs)

        # Instantiate all components using a small batch from training set.
        if not init_batch:
            run_iter = MatchingIterator(
                train_dataset,
                train_dataset,
                train=False,
                batch_size=4,
                device='cpu',
                sort_in_buckets=False)
            init_batch = next(run_iter.__iter__())
        self.forward(init_batch)

        # Keep this init_batch for future initializations.
        self.state_meta.init_batch = init_batch

        self._initialized = True
        logger.info('Successfully initialized MatchingModel with {:d} trainable '
                    'parameters.'.format(tally_parameters(self)))

    def _reset_embeddings(self, vocabs):
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

    def _get_attr_comparator(self, arg, attr_summarizer):
        r"""Get the attribute comparator.

        Args:
            arg (string):
                The attribute comparator to use. Can be one of the supported style
                arguments for :class:`~modules.Merge`, specifically, 'abs-diff', 'diff',
                'concat', 'concat-diff', 'concat-abs-diff', or 'mul'. If not specified
                uses 'abs-diff' for SIF and RNN, 'concat' for Attention, and 'concat-diff'
                for Hybrid".
            attr_summarizer (:class:`deepmatcher.AttrSummarizer`):
                The attribute summarizer object in :mod:`attr_summarizers` (i.e., it
                should be one of "SIF", "RNN", "Attention", and "Hybrid" defined in
                :mod:`attr_summarizers`).

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
        r"""Performs a forward pass through the model.

        Overrides :meth:`torch.nn.Module.forward`.

        Args:
            input (:class:`~deepmatcher.batch.MatchingBatch`): A batch of tuple pairs
                processed into tensors.
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

            # Remove metadata information at this point.
            left_summary, right_summary = left_summary.data, right_summary.data

            if self.attr_condensors:
                left_summary = self.attr_condensors[name](left_summary)
                right_summary = self.attr_condensors[name](right_summary)
            attr_comparisons.append(self.attr_comparators[name](left_summary,
                                                                right_summary))

        entity_comparison = self.attr_merge(*attr_comparisons)
        return self.classifier(entity_comparison)

    def _register_train_buffer(self, name, value):
        r"""Adds a persistent buffer containing training metadata to the module.
        """
        self._train_buffers.add(name)
        setattr(self, name, value)

    def save_state(self, path, include_meta=True):
        r"""Save the model state to a certain path.

        Args:
            path (string): The path to save the model state to.
            include_meta (bool): Whether to include training dataset metadata along with
                the model parameters when saving. If False, the model will not be
                automatically initialized upon loading - you will need to initialize
                manually using :meth:`initialize`.
        """
        state = {'model': self.state_dict()}
        for k in self._train_buffers:
            if include_meta or k != 'state_meta':
                state[k] = getattr(self, k)
        torch.save(state, path, pickle_module=dill)

    def load_state(self, path, map_location=None):
        r"""Load the model state from a file in a certain path.

        Args:
            path (string): The path to load the model state from.
        """
        state = torch.load(path, pickle_module=dill, map_location=map_location)
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


class AttrSummarizer(dm.modules.LazyModule):
    r"""__init__(word_contextualizer, word_comparator, word_aggregator, hidden_size=None)

    The Attribute Summarizer.

    Summarizes the two word embedding sequences of an attribute.
    `Refer this tutorial <https://nbviewer.jupyter.org/github/sidharthms/deepmatcher/blob/master/examples/matching_models.ipynb#2.1.-Attribute-Summarization>`__
    for details. Sub-classes that implement various built-in options for this module are
    defined in :mod:`deepmatcher.attr_summarizers`.

    Args:
        word_contextualizer (string or :class:`WordContextualizer` or callable):
            Module that takes a word embedding sequence and produces a context-aware
            word embedding sequence as output.
            Options `listed here <https://nbviewer.jupyter.org/github/sidharthms/deepmatcher/blob/master/examples/matching_models.ipynb#word_contextualizer-can-be-set-to-one-of-the-following:>`__.
        word_comparator (string or :class:`WordComparator` or callable):
            Module that takes two word embedding sequences, aligns words in the two
            sequences, and performs a word by word comparison.
            Options `listed here <https://nbviewer.jupyter.org/github/sidharthms/deepmatcher/blob/master/examples/matching_models.ipynb#word_comparator-can-be-set-to-one-of-the-following:>`__.
        word_aggregator (string or :class:`WordAggregator` or callable):
            Module that takes a sequence of vectors and aggregates it into a single
            vector.
            Options `listed here <https://nbviewer.jupyter.org/github/sidharthms/deepmatcher/blob/master/examples/matching_models.ipynb#word_aggregator-can-be-set-to-one-of-the-following:>`__.
        hidden_size (int):
            The hidden size used for the three sub-modules (i.e., word_contextualizer,
            word_comparator, and word_aggregator). If None, uses the input size, i.e., the
            size of the last dimension of the input to this module as the hidden size.
            Defaults to None.
    """

    def _init(self,
              word_contextualizer,
              word_comparator,
              word_aggregator,
              hidden_size=None):
        self.word_contextualizer = WordContextualizer._create(
            word_contextualizer, hidden_size=hidden_size)
        self.word_comparator = WordComparator._create(
            word_comparator, hidden_size=hidden_size)
        self.word_aggregator = WordAggregator._create(
            word_aggregator, hidden_size=hidden_size)

    def _forward(self, left_input, right_input):
        r"""The forward function of attribute summarizer.
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
    def _create(cls, arg, **kwargs):
        r"""Create an attribute summarization object.

        Args:
            arg (string or :mod:`deepmatcher.attr_summarizers` or callable):
                Same as the `attr_summarizer` argument to the constructor of
                :class:`MatchingModel`.
            **kwargs:
                Keyword arguments to the constructor of the AttrSummarizer sub-class.
                For details on what these can be, please refer to the documentation of the
                sub-classes in :mod:`deepmatcher.attr_summarizers`.
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


def _create_attr_comparator(arg):
    r"""Create an attribute comparator object.

    Args:
        arg (string):
            Same as the `attr_comparator` argument to the constructor of
            :class:`MatchingModel`.
    """
    assert arg is not None
    module = dm.modules._merge_module(arg)
    module.expect_signature('[AxB, AxB] -> [AxC]')
    return module


class WordContextualizer(dm.modules.LazyModule):
    r"""__init__()

    The Word Contextualizer.

    Takes a word embedding sequence and produces a context-aware word embedding sequence.
    `Refer this tutorial <https://nbviewer.jupyter.org/github/sidharthms/deepmatcher/blob/master/examples/matching_models.ipynb#2.1.1.-Word-Contextualizer>`__
    for details. Sub-classes that implement various options for this module are defined
    in :mod:`deepmatcher.word_contextualizers`.
    """

    @classmethod
    def _create(cls, arg, **kwargs):
        r"""Create a word contextualizer object.

        Args:
            arg (string or :mod:`deepmatcher.word_contextualizers` or callable):
                Same as the `word_contextualizer` argument to the constructor of
                :class:`AttrSummarizer`.
            **kwargs:
                Keyword arguments to the constructor of the WordContextualizer sub-class.
                For details on what these can be, please refer to the documentation of the
                sub-classes in :mod:`deepmatcher.word_contextualizers`.
        """
        if isinstance(arg, six.string_types):
            if dm.word_contextualizers.RNN.supports_style(arg):
                wc = dm.word_contextualizers.RNN(arg, **kwargs)
            elif arg == 'self-attention':
                wc = dm.word_contextualizers.SelfAttention(**kwargs)
            else:
                raise ValueError('Unknown Word Contextualizer name.')
        else:
            wc = _utils.get_module(WordContextualizer, arg)

        if wc is not None:
            wc.expect_signature('[AxBxC] -> [AxBxD]')

        return wc


class WordComparator(dm.modules.LazyModule):
    r"""__init__()

    The Word Comparator.

    Takes two word embedding sequences, aligns words in the two sequences, and performs a
    word by word comparison.
    `Refer this tutorial <https://nbviewer.jupyter.org/github/sidharthms/deepmatcher/blob/master/examples/matching_models.ipynb#2.1.2.-Word-Comparator>`__
    for details. Sub-classes that implement various options for this module are defined
    in :mod:`deepmatcher.word_comparators`.
    """

    @classmethod
    def _create(cls, arg, **kwargs):
        r"""Create a word comparator object.

        Args:
            arg (string or :mod:`deepmatcher.word_comparators` or callable):
                Same as the `word_comparator` argument to the constructor of
                :class:`AttrSummarizer`.
            **kwargs:
                Keyword arguments to the constructor of the WordComparator sub-class.
                For details on what these can be, please refer to the documentation of the
                sub-classes in :mod:`deepmatcher.word_comparators`.
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
    r"""__init__()

    The Word Aggregator.

    Takes a sequence of vectors and aggregates it into a single vector.
    `Refer this tutorial <https://nbviewer.jupyter.org/github/sidharthms/deepmatcher/blob/master/examples/matching_models.ipynb#2.1.3.-Word-Aggregator>`__
    for details. Sub-classes that implement various options for this module are defined in
    :mod:`deepmatcher.word_aggregators`.
    """

    @classmethod
    def _create(cls, arg, **kwargs):
        r"""
        Create a word aggregator object.

        Args:
            arg (string or :mod:`deepmatcher.word_aggregators` or callable):
                Same as the `word_aggregator` argument to the constructor of
                :class:`AttrSummarizer`.
        **kwargs:
            Keyword arguments to the constructor of the WordAggregator sub-class.
            For details on what these can be, please refer to the documentation of the
            sub-classes in :mod:`deepmatcher.word_aggregators`.
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
    r"""The Classifier Network.

    Predicts whether a tuple pair matches or not given a representation of all the
    attribute summarizations.
    `Refer this tutorial <https://nbviewer.jupyter.org/github/sidharthms/deepmatcher/blob/master/examples/matching_models.ipynb#3.-Classifier>`__
    for details.

    Args:
        transform_network (string or :class:`~deepmatcher.modules.Transform` or callable):
            The neural network to transform the input vector of the classifier to a
            hidden representation of size `hidden_size`. Argument must specify a
            :ref:`transform-op` operation.

        hidden_size (int):
            The size of the hidden representation generated by the transformation network.
            If None, uses the size of the input vector to this module as the hidden size.
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
