from __future__ import division

import copy
import logging
import os
import pdb
from collections import Counter, defaultdict
from timeit import default_timer as timer

import pandas as pd
import pyprind
import six
from sklearn.decomposition import TruncatedSVD

import torch
import torch.nn as nn
from torchtext.legacy import data
from torchtext.legacy.data.example import Example
from torchtext.utils import unicode_csv_reader

from ..models.modules import NoMeta, Pool
from .field import MatchingField
from .iterator import MatchingIterator

logger = logging.getLogger(__name__)


def split(table,
          path,
          train_prefix,
          validation_prefix,
          test_prefix,
          split_ratio=[0.6, 0.2, 0.2],
          stratified=False,
          strata_field='label'):
    """Split a pandas dataframe or CSV file into train / validation / test data sets.

    Args:
        table (pandas.Dataframe or string): The pandas dataframe or CSV file to split.
        path (string): The directory to save the train, validation and test CSV files to.
        train: Suffix to add to `path` to get the training set save path.
        validation: Suffix to add to `path` to get the validation set save path.
        test: Suffix to add to `path` to get the test set save path.
        split_ratio (List of floats): a list of 3 numbers denoting the relative sizes of
            train, test and valid splits respectively. Default is [0.6, 0.2, 0.2].
        stratified (bool): whether the sampling should be stratified.
            Default is False.
        strata_field (str): name of the examples Field stratified over.
            Default is 'label' for the conventional label field.
    """
    assert len(split_ratio) == 3

    if not isinstance(table, pd.DataFrame):
        table = pd.read_csv(table)
    if table.index.name is not None:
        table = table.reset_index()

    examples = list(table.itertuples(index=False))
    fields = [(col, None) for col in list(table)]
    dataset = data.Dataset(examples, fields)
    train, valid, test = dataset.split(split_ratio, stratified, strata_field)

    tables = (pd.DataFrame(train.examples), pd.DataFrame(valid.examples),
              pd.DataFrame(test.examples))
    prefixes = (train_prefix, validation_prefix, test_prefix)

    for i in range(len(tables)):
        tables[i].columns = table.columns
        tables[i].to_csv(os.path.join(path, prefixes[i]), index=False)


class MatchingDataset(data.Dataset):
    r"""Represents dataset with associated metadata.

    Holds all information about one split of a dataset (e.g. training set).

    Attributes:
        fields (dict): A mapping from attribute names (e.g. "left_address") to
            corresponding :class:`~data.MatchingField` objects that specify how to process
            the field.
        examples (list): A list containing all the examples (labeled tuple pairs) in this
            dataset.
        metadata (dict): Metadata about the dataset (e.g. word probabilities).
            See :meth:`~data.MatchingDataset.compute_metadata` for details.
        corresponding_field (dict): A mapping from left table attribute names
            (e.g. "left_address") to corresponding right table attribute names
            (e.g. "right_address") and vice versa.
        text_fields (dict): A mapping from canonical attribute names (e.g. "address") to
            tuples of the corresponding left and right attribute names
            (e.g. ("left_address", "right_address")).
        all_left_fields (list): A list of all left table attribute names.
        all_right_fields (list): A list of all right table attribute names.
        canonical_text_fields (list): A list of all canonical attribute names.
        label_field (str): Name of the column containing labels.
        id_field (str): Name of the column containing tuple pair ids.
    """

    class CacheStaleException(Exception):
        r"""Raised when the dataset cache is stale and no fallback behavior is specified.
        """
        pass

    def __init__(self,
                 fields,
                 column_naming,
                 path=None,
                 format='csv',
                 examples=None,
                 metadata=None,
                 **kwargs):
        r"""Creates a MatchingDataset.

        Creates a MatchingDataset by performing the following, if `examples` parameter is
        not specified:

        #. Read each example (tuple pair) in specified CSV file.
        #. Preprocess example. Involves lowercasing and tokenization (unless disabled).
        #. Compute metadata. See :meth:`~data.MatchingDataset.compute_metadata` for
            details.

        If `examples` is specified, initializes MatchingDataset from given `examples`
        and `metadata` arguments.

        Arguments:
            fields (list(tuple(str, MatchingField))): A list of tuples containing column
                name (e.g. "left_address") and corresponding :class:`~data.MatchingField`
                pairs, in the same order that the columns occur in the CSV file. Tuples of
                (name, None) represent columns that will be ignored.
            column_naming (dict): A `dict` containing the following keys:
                * ``id``: The name of the tuple pair ID column.
                * ``label``: The name of the tuple pair match label column.
                * ``left``: The prefix for attribute names belonging to the left table.
                * ``right``: The prefix for attribute names belonging to the right table.
            path (str): Path to the data file. Must be specified if `examples` is None.
            format (str): The format of the data file. One of "CSV" or "TSV".
            examples (list): A list containing all the examples (labeled tuple pairs) in
                this dataset. Must be specified if `path` is None.
            metadata (dict): Metadata about the dataset (e.g. word probabilities).
                See :meth:`~data.MatchingDataset.compute_metadata` for details.
            filter_pred (callable or None): Use only examples for which
                filter_pred(example) is True, or use all examples if None.
                Default is None. This is a keyword-only parameter.
        """
        if examples is None:
            make_example = {
                'json': Example.fromJSON, 'dict': Example.fromdict,
                'tsv': Example.fromCSV, 'csv': Example.fromCSV}[format.lower()]

            lines = 0
            with open(os.path.expanduser(path), encoding="utf8") as f:
                for line in f:
                    lines += 1

            with open(os.path.expanduser(path), encoding="utf8") as f:
                if format == 'csv':
                    reader = unicode_csv_reader(f)
                elif format == 'tsv':
                    reader = unicode_csv_reader(f, delimiter='\t')
                else:
                    reader = f

                next(reader)
                examples = [make_example(line, fields) for line in
                    pyprind.prog_bar(reader, iterations=lines,
                        title='\nReading and processing data from "' + path + '"')]

            super(MatchingDataset, self).__init__(examples, fields, **kwargs)
        else:
            self.fields = dict(fields)
            self.examples = examples
            self.metadata = metadata

        self.path = path
        self.column_naming = column_naming
        self._set_attributes()

    def _set_attributes(self):
        """Sets attributes by inferring mapping between left and right table attributes.
        """
        self.corresponding_field = {}
        self.text_fields = {}

        self.all_left_fields = []
        for name, field in six.iteritems(self.fields):
            if name.startswith(self.column_naming['left']) and field is not None:
                self.all_left_fields.append(name)

        self.all_right_fields = []
        for name, field in six.iteritems(self.fields):
            if name.startswith(self.column_naming['right']) and field is not None:
                self.all_right_fields.append(name)

        self.canonical_text_fields = []
        for left_name in self.all_left_fields:
            canonical_name = left_name[len(self.column_naming['left']):]
            right_name = self.column_naming['right'] + canonical_name
            self.corresponding_field[left_name] = right_name
            self.corresponding_field[right_name] = left_name
            self.text_fields[canonical_name] = left_name, right_name
            self.canonical_text_fields.append(canonical_name)

        self.all_text_fields = self.all_left_fields + self.all_right_fields
        self.label_field = self.column_naming['label']
        self.id_field = self.column_naming['id']

    def compute_metadata(self, pca=False):
        r"""Computes metadata about the dataset.

        Computes the following metadata about the dataset:

        * ``word_probs``: For each attribute in the dataset, a mapping from words to
          word (token) probabilities.
        * ``totals``: For each attribute in the dataset, a count of the total number of
          words present in all attribute examples.
        * ``pc``: For each attribute in the dataset, the first principal component of the
          sequence embeddings for all values of that attribute. The sequence embedding of
          an attribute value is computed by taking the weighted average of its word
          embeddings, where the weight is the soft inverse word probability. Refer
          `Arora et al. (2017) <https://openreview.net/pdf?id=SyK00v5xx>`__ for details.

        Arguments:
            pca (bool): Whether to compute the ``pc`` metadata.
        """
        self.metadata = {}

        # Create an iterator over the entire dataset.
        train_iter = MatchingIterator(
            self, self, train=False, batch_size=1024, device='cpu', sort_in_buckets=False)
        counter = defaultdict(Counter)

        # For each attribute, find the number of times each word id occurs in the dataset.
        # Note that word ids here also include ``UNK`` tokens, padding tokens, etc.
        for batch in pyprind.prog_bar(train_iter, title='\nBuilding vocabulary'):
            for name in self.all_text_fields:
                attr_input = getattr(batch, name)
                counter[name].update(attr_input.data.data.view(-1).tolist())

        word_probs = {}
        totals = {}
        for name in self.all_text_fields:
            attr_counter = counter[name]
            total = sum(attr_counter.values())
            totals[name] = total

            field_word_probs = {}
            for word, freq in attr_counter.items():
                field_word_probs[word] = freq / total
            word_probs[name] = field_word_probs
        self.metadata['word_probs'] = word_probs
        self.metadata['totals'] = totals

        if not pca:
            return

        # To compute principal components, we need to compute weighted sequence embeddings
        # for each attribute. To do so, for each attribute, we first construct a neural
        # network to compute word embeddings and take their weighted average.
        field_embed = {}
        embed = {}
        inv_freq_pool = Pool('inv-freq-avg')
        for name in self.all_text_fields:
            field = self.fields[name]
            if field not in field_embed:
                vectors_size = field.vocab.vectors.shape
                embed_layer = nn.Embedding(vectors_size[0], vectors_size[1])
                embed_layer.weight.data.copy_(field.vocab.vectors)
                embed_layer.weight.requires_grad = False
                field_embed[field] = NoMeta(embed_layer)
            embed[name] = field_embed[field]

        # Create an iterator over the entire dataset.
        train_iter = MatchingIterator(
            self, self, train=False, batch_size=1024, device='cpu', sort_in_buckets=False)
        attr_embeddings = defaultdict(list)

        # Run the constructed neural network to compute weighted sequence embeddings
        # for each attribute of each example in the dataset.
        for batch in pyprind.prog_bar(train_iter,
            title='\nComputing principal components'):
            for name in self.all_text_fields:
                attr_input = getattr(batch, name)
                embeddings = inv_freq_pool(embed[name](attr_input))
                attr_embeddings[name].append(embeddings.data.data)

        # Compute the first principal component of weighted sequence embeddings for each
        # attribute.
        pc = {}
        for name in self.all_text_fields:
            concatenated = torch.cat(attr_embeddings[name])
            svd = TruncatedSVD(n_components=1, n_iter=7)
            svd.fit(concatenated.numpy())
            pc[name] = svd.components_[0]
        self.metadata['pc'] = pc

    def finalize_metadata(self):
        r"""Perform final touches to dataset metadata.

        This allows performing modifications to metadata that cannot be serialized into
        the cache.
        """

        self.orig_metadata = copy.deepcopy(self.metadata)
        for name in self.all_text_fields:
            self.metadata['word_probs'][name] = defaultdict(
                lambda: 1 / self.metadata['totals'][name],
                self.metadata['word_probs'][name])

    def get_raw_table(self):
        r"""Create a raw pandas table containing all examples (tuple pairs) in the dataset.

        To resurrect tokenized attributes, this method currently naively joins the tokens
        using the whitespace delimiter.
        """
        rows = []
        columns = list(name for name, field in six.iteritems(self.fields) if field)
        for ex in self.examples:
            row = []
            for attr in columns:
                if self.fields[attr]:
                    val = getattr(ex, attr)
                    if self.fields[attr].sequential:
                        val = ' '.join(val)
                    row.append(val)
            rows.append(row)

        return pd.DataFrame(rows, columns=columns)

    def sort_key(self, ex):
        r"""Sort key for dataset examples.

        A key to use for sorting dataset examples for batching together examples with
        similar lengths to minimize padding."""

        return interleave_keys([len(getattr(ex, attr)) for attr in self.all_text_fields])

    @staticmethod
    def save_cache(datasets, fields, datafiles, cachefile, column_naming, state_args):
        r"""Save datasets and corresponding metadata to cache.

        This method also saves as many data loading arguments as possible to help ensure
        that the cache contents are still relevant for future data loading calls. Refer
        to :meth:`~data.Dataset.load_cache` for more details.

        Arguments:
            datasets (list): List of datasets to cache.
            fields (dict): Mapping from attribute names (e.g. "left_address") to
                corresponding :class:`~data.MatchingField` objects that specify how to
                process the field.
            datafiles (list): A list of the data files.
            cachefile (str): The cache file path.
            column_naming (dict): A `dict` containing column naming conventions. See
                `__init__` for details.
            state_args (dict): A `dict` containing other information about the state under
                which the cache was created.
        """
        examples = [dataset.examples for dataset in datasets]
        train_metadata = datasets[0].metadata
        datafiles_modified = [os.path.getmtime(datafile) for datafile in datafiles]
        vocabs = {}
        field_args = {}
        reverse_fields = {}
        for name, field in six.iteritems(fields):
            reverse_fields[field] = name

        for field, name in six.iteritems(reverse_fields):
            if field is not None and hasattr(field, 'vocab'):
                vocabs[name] = field.vocab
        for name, field in six.iteritems(fields):
            field_args[name] = None
            if field is not None:
                field_args[name] = field.preprocess_args()

        data = {
            'examples': examples,
            'train_metadata': train_metadata,
            'vocabs': vocabs,
            'datafiles': datafiles,
            'datafiles_modified': datafiles_modified,
            'field_args': field_args,
            'state_args': state_args,
            'column_naming': column_naming
        }
        torch.save(data, cachefile)

    @staticmethod
    def load_cache(fields, datafiles, cachefile, column_naming, state_args):
        r"""Load datasets and corresponding metadata from cache.

        This method also checks whether any of the data loading arguments have changes
        that make the cache contents invalid. The following kinds of changes are currently
        detected automatically:

        * Data filename changes (e.g. different train filename)
        * Data file modifications (e.g. train data modified)
        * Column changes (e.g. using a different subset of columns in CSV file)
        * Column specification changes (e.g. changing lowercasing behavior)
        * Column naming convention changes (e.g. different labeled data column)

        Arguments:
            fields (dict): Mapping from attribute names (e.g. "left_address") to
                corresponding :class:`~data.MatchingField` objects that specify how to
                process the field.
            datafiles (list): A list of the data files.
            cachefile (str): The cache file path.
            column_naming (dict): A `dict` containing column naming conventions. See
                `__init__` for details.
            state_args (dict): A `dict` containing other information about the state under
                which the cache was created.

        Returns:
            Tuple containing unprocessed cache data dict and a list of cache staleness
            causes, if any.

        .. warning::
            Note that if a column specification, i.e., arguments to
            :class:`~data.MatchingField` include callable arguments (e.g. lambdas or
            functions) these arguments cannot be serialized and hence will not be checked
            for modifications.
        """
        cached_data = torch.load(cachefile)
        cache_stale_cause = set()

        if datafiles != cached_data['datafiles']:
            cache_stale_cause.add('Data file list has changed.')

        datafiles_modified = [os.path.getmtime(datafile) for datafile in datafiles]
        if datafiles_modified != cached_data['datafiles_modified']:
            cache_stale_cause.add('One or more data files have been modified.')

        if set(fields.keys()) != set(cached_data['field_args'].keys()):
            cache_stale_cause.add('Fields have changed.')

        for name, field in six.iteritems(fields):
            none_mismatch = (field is None) != (cached_data['field_args'][name] is None)
            args_mismatch = False
            if field is not None and cached_data['field_args'][name] is not None:
                args_mismatch = field.preprocess_args() != cached_data['field_args'][name]
            if none_mismatch or args_mismatch:
                cache_stale_cause.add('Field arguments have changed.')
            if field is not None and not isinstance(field, MatchingField):
                cache_stale_cause.add('Cache update required.')

        if column_naming != cached_data['column_naming']:
            cache_stale_cause.add('Other arguments have changed.')

        cache_stale_cause.update(
            MatchingDataset.state_args_compatibility(state_args,
                                                     cached_data['state_args']))

        return cached_data, cache_stale_cause

    @staticmethod
    def state_args_compatibility(cur_state, old_state):
        errors = []
        if not old_state['train_pca'] and cur_state['train_pca']:
            errors.append('PCA computation necessary.')
        return errors

    @staticmethod
    def restore_data(fields, cached_data):
        r"""Recreate datasets and related data from cache.

        This restores all datasets, metadata and attribute information (including the
        vocabulary and word embeddings for all tokens in each attribute).
        """
        datasets = []
        for d in range(len(cached_data['datafiles'])):
            metadata = None
            if d == 0:
                metadata = cached_data['train_metadata']
            dataset = MatchingDataset(
                path=cached_data['datafiles'][d],
                fields=fields,
                examples=cached_data['examples'][d],
                metadata=metadata,
                column_naming=cached_data['column_naming'])
            datasets.append(dataset)

        for name, field in fields:
            if name in cached_data['vocabs']:
                field.vocab = cached_data['vocabs'][name]

        return datasets

    @classmethod
    def splits(cls,
               path,
               train=None,
               validation=None,
               test=None,
               fields=None,
               embeddings=None,
               embeddings_cache=None,
               column_naming=None,
               cache=None,
               check_cached_data=True,
               auto_rebuild_cache=False,
               train_pca=False,
               **kwargs):
        r"""Create Dataset objects for multiple splits of a dataset.

        Args:
            path (str): Common prefix of the splits' file paths.
            train (str): Suffix to add to path for the train set.
            validation (str): Suffix to add to path for the validation set, or None
                for no validation set. Default is None.
            test (str): Suffix to add to path for the test set, or None for no test
                set. Default is None.
            fields (list(tuple(str, MatchingField))): A list of tuples containing column
                name (e.g. "left_address") and corresponding :class:`~data.MatchingField`
                pairs, in the same order that the columns occur in the CSV file. Tuples of
                (name, None) represent columns that will be ignored.
            embeddings (str or list): Same as `embeddings` parameter of
                :func:`~data.process`.
            embeddings_cache (str): Directory to store dowloaded word vector data.
            column_naming (dict): Same as `column_naming` paramter of `__init__`.
            cache (str): Suffix to add to path for cache file. If `None` disables caching.
            check_cached_data (bool): Verify that data files haven't changes since the
                cache was constructed and that relevant field options haven't changed.
            auto_rebuild_cache (bool): Automatically rebuild the cache if the data files
                are modified or if the field options change. Defaults to False.
            train_pca (bool): Whether to compute PCA for each attribute as part of
                dataset metadata compuatation. Defaults to False.
            filter_pred (callable or None): Use only examples for which
                filter_pred(example) is True, or use all examples if None.
                Default is None. This is a keyword-only parameter.

        Returns:
            Tuple[MatchingDataset]: Datasets for (train, validation, and test) splits in
                that order, if provided.
        """

        fields_dict = dict(fields)
        state_args = {'train_pca': train_pca}

        datasets = None
        if cache:
            datafiles = list(f for f in (train, validation, test) if f is not None)
            datafiles = [os.path.expanduser(os.path.join(path, d)) for d in datafiles]
            cachefile = os.path.expanduser(os.path.join(path, cache))
            try:
                cached_data, cache_stale_cause = MatchingDataset.load_cache(
                    fields_dict, datafiles, cachefile, column_naming, state_args)

                if check_cached_data and cache_stale_cause:
                    if not auto_rebuild_cache:
                        raise MatchingDataset.CacheStaleException(cache_stale_cause)
                    else:
                        logger.warning('Rebuilding data cache because: %s', list(cache_stale_cause))

                if not check_cached_data or not cache_stale_cause:
                    datasets = MatchingDataset.restore_data(fields, cached_data)

            except IOError:
                pass

        if not datasets:
            begin = timer()
            dataset_args = {'fields': fields, 'column_naming': column_naming, **kwargs}
            train_data = None if train is None else cls(
                path=os.path.join(path, train), **dataset_args)
            val_data = None if validation is None else cls(
                path=os.path.join(path, validation), **dataset_args)
            test_data = None if test is None else cls(
                path=os.path.join(path, test), **dataset_args)
            datasets = tuple(
                d for d in (train_data, val_data, test_data) if d is not None)

            after_load = timer()
            logger.info('Data load took: {}s'.format(after_load - begin))

            fields_set = set(fields_dict.values())
            for field in fields_set:
                if field is not None and field.use_vocab:
                    field.build_vocab(
                        *datasets, vectors=embeddings, cache=embeddings_cache)
            after_vocab = timer()
            logger.info('Vocab construction time: {}s'.format(after_vocab - after_load))

            if train:
                datasets[0].compute_metadata(train_pca)
            after_metadata = timer()
            logger.info(
                'Metadata computation time: {}s'.format(after_metadata - after_vocab))

            if cache:
                MatchingDataset.save_cache(datasets, fields_dict, datafiles, cachefile,
                                           column_naming, state_args)
                after_cache = timer()
                logger.info('Cache save time: {}s'.format(after_cache - after_vocab))

        if train:
            datasets[0].finalize_metadata()

            # Save additional information to train dataset.
            datasets[0].embeddings = embeddings
            datasets[0].embeddings_cache = embeddings_cache
            datasets[0].train_pca = train_pca

        # Set vocabs.
        for dataset in datasets:
            dataset.vocabs = {
                name: datasets[0].fields[name].vocab
                for name in datasets[0].all_text_fields
            }

        if len(datasets) == 1:
            return datasets[0]
        return tuple(datasets)


def interleave_keys(keys):
    r"""Interleave bits from two sort keys to form a joint sort key.

    Examples that are similar in both of the provided keys will have similar
    values for the key defined by this function. Useful for tasks with two
    text fields like machine translation or natural language inference.
    """

    def interleave(args):
        return ''.join([x for t in zip(*args) for x in t])

    return int(''.join(interleave(format(x, '016b') for x in keys)), base=2)
