import copy
import io
import logging
import os
from timeit import default_timer as timer

import six

from torchtext.utils import unicode_csv_reader

from .dataset import MatchingDataset
from .field import MatchingField

logger = logging.getLogger(__name__)


def _check_header(header, id_attr, left_prefix, right_prefix, label_attr, ignore_columns):
    r"""Verify CSV file header.

    Checks that:
    * There is a label column
    * There is an ID column
    * All columns except the label and ID columns, and ignored columns start with either
        the left table attribute prefix or the right table attribute prefix.
    * The number of left and right table attributes are the same.
    """
    # assert id_attr in header
    if label_attr:
        assert label_attr in header

    for attr in header:
        if attr not in (id_attr, label_attr) and attr not in ignore_columns:
            if not attr.startswith(left_prefix) and not attr.startswith(right_prefix):
                raise ValueError('Attribute ' + attr + ' is not a left or a right table '
                                 'column, not a label or id and is not ignored. Not sure '
                                 'what it is...')

    num_left = sum(attr.startswith(left_prefix) for attr in header)
    num_right = sum(attr.startswith(right_prefix) for attr in header)
    assert num_left == num_right


def _make_fields(header, id_attr, label_attr, ignore_columns, lower, tokenize,
                 include_lengths):
    r"""Create field metadata, i.e., attribute processing specification for each
    attribute.

    This includes fields for label and ID columns.

    Returns:
        list(tuple(str, MatchingField)): A list of tuples containing column name
            (e.g. "left_address") and corresponding :class:`~data.MatchingField` pairs,
            in the same order that the columns occur in the CSV file.
    """

    text_field = MatchingField(
        lower=lower,
        tokenize=tokenize,
        init_token='<<<',
        eos_token='>>>',
        batch_first=True,
        include_lengths=include_lengths)
    numeric_field = MatchingField(
        sequential=False, preprocessing=lambda x: int(x), use_vocab=False)
    id_field = MatchingField(sequential=False, use_vocab=False, id=True)

    fields = []
    for attr in header:
        if attr == id_attr:
            fields.append((attr, id_field))
        elif attr == label_attr:
            fields.append((attr, numeric_field))
        elif attr in ignore_columns:
            fields.append((attr, None))
        else:
            fields.append((attr, text_field))
    return fields


def _maybe_download_nltk_data():
    import nltk
    nltk.download('perluniprops', quiet=True)
    nltk.download('nonbreaking_prefixes', quiet=True)
    nltk.download('punkt', quiet=True)


def process(path,
            train=None,
            validation=None,
            test=None,
            unlabeled=None,
            cache='cacheddata.pth',
            check_cached_data=True,
            auto_rebuild_cache=True,
            tokenize='nltk',
            lowercase=True,
            embeddings='fasttext.en.bin',
            embeddings_cache_path='~/.vector_cache',
            ignore_columns=(),
            include_lengths=True,
            id_attr='id',
            label_attr='label',
            left_prefix='left_',
            right_prefix='right_',
            use_magellan_convention=False,
            pca=True):
    """Creates dataset objects for multiple splits of a dataset.

    This involves the following steps (if data cannot be retrieved from the cache):
    #. Read CSV header of a data file and verify header is sane.
    #. Create fields, i.e., column processing specifications (e.g. tokenization, label
        conversion to integers etc.)
    #. Load each data file:
        #. Read each example (tuple pair) in specified CSV file.
        #. Preprocess example. Involves lowercasing and tokenization (unless disabled).
        #. Compute metadata if training data file. \
            See :meth:`MatchingDataset.compute_metadata` for details.
    #. Create vocabulary consisting of all tokens in all attributes in all datasets.
    #. Download word embedding data if necessary.
    #. Create mapping from each word in vocabulary to its word embedding.
    #. Compute metadata
    #. Write to cache

    Arguments:
        path (str): Common prefix of the splits' file paths.
        train (str): Suffix to add to path for the train set.
        validation (str): Suffix to add to path for the validation set, or None
            for no validation set. Default is None.
        test (str): Suffix to add to path for the test set, or None for no test
            set. Default is None.
        cache (str): Suffix to add to path for cache file. If `None` disables caching.
        check_cached_data (bool): Verify that data files haven't changes since the
            cache was constructed and that relevant field options haven't changed.
        auto_rebuild_cache (bool): Automatically rebuild the cache if the data files
            are modified or if the field options change. Defaults to False.
        lowercase (bool): Whether to lowercase all words in all attributes.
        embeddings (str or list): One or more of the following strings:

            * `fasttext.{lang}.bin`:
                This uses sub-word level word embeddings based on binary models from "wiki
                word vectors" released by FastText. {lang} is 'en' or any other 2 letter
                ISO 639-1 Language Code, or 3 letter ISO 639-2 Code, if the language does
                not have a 2 letter code. 300d vectors.
                ``fasttext.en.bin`` is the default.
            * `fasttext.wiki.vec`:
                Uses wiki news word vectors released as part of "Advances in Pre-Training
                Distributed Word Representations" by Mikolov et al. (2018). 300d vectors.
            * `fasttext.crawl.vec`:
                Uses Common Crawl word vectors released as part of "Advances in
                Pre-Training Distributed Word Representations" by Mikolov et al. (2018).
                300d vectors.
            * `glove.6B.{dims}`:
                Uses uncased Glove trained on Wiki + Gigaword. {dims} is one of (50d,
                100d, 200d, or 300d).
            * `glove.42B.300d`:
                Uses uncased Glove trained on Common Crawl. 300d vectors.
            * `glove.840B.300d`:
                Uses cased Glove trained on Common Crawl. 300d vectors.
            * `glove.twitter.27B.{dims}`:
                Uses cased Glove trained on Twitter. {dims} is one of (25d, 50d, 100d, or
                200d).
        embeddings_cache_path (str): Directory to store dowloaded word vector data.
        ignore_columns (list): A list of columns to ignore in the CSV files.
        include_lengths (bool): Whether to provide the model with the lengths of
            each attribute sequence in each batch. If True, length information can be
            used by the neural network, e.g. when picking the last RNN output of each
            attribute sequence.
        id_attr (str): The name of the tuple pair ID column in the CSV file.
        label_attr (str): The name of the tuple pair match label column in the CSV file.
        left_prefix (str): The prefix for attribute names belonging to the left table.
        right_prefix (str): The prefix for attribute names belonging to the right table.
        use_magellan_convention (bool): Set `id_attr`, `left_prefix`, and `right_prefix`
            according to Magellan (py_entitymatching Python package) naming conventions.
            Specifically, set them to be '_id', 'ltable_', and 'rtable_' respectively.
        pca (bool): Whether to compute PCA for each attribute (needed for SIF model).
            Defaults to False.

    Returns:
        Tuple[MatchingDataset]: Datasets for (train, validation, and test) splits in that
            order, if provided, or dataset for unlabeled, if provided.
    """
    if unlabeled is not None:
        raise ValueError('Parameter "unlabeled" has been deprecated, use '
                         '"deepmatcher.data.process_unlabeled" instead.')

    if use_magellan_convention:
        id_attr = '_id'
        left_prefix = 'ltable_'
        right_prefix = 'rtable_'

    # TODO(Sid): check for all datasets to make sure the files exist and have the same schema
    a_dataset = train or validation or test
    with io.open(os.path.expanduser(os.path.join(path, a_dataset)), encoding="utf8") as f:
        header = next(unicode_csv_reader(f))

    _maybe_download_nltk_data()
    _check_header(header, id_attr, left_prefix, right_prefix, label_attr, ignore_columns)
    fields = _make_fields(header, id_attr, label_attr, ignore_columns, lowercase,
                          tokenize, include_lengths)

    column_naming = {
        'id': id_attr,
        'left': left_prefix,
        'right': right_prefix,
        'label': label_attr
    }

    datasets = MatchingDataset.splits(
        path,
        train,
        validation,
        test,
        fields,
        embeddings,
        embeddings_cache_path,
        column_naming,
        cache,
        check_cached_data,
        auto_rebuild_cache,
        train_pca=pca)

    # Save additional information to train dataset.
    datasets[0].ignore_columns = ignore_columns
    datasets[0].tokenize = tokenize
    datasets[0].lowercase = lowercase
    datasets[0].include_lengths = include_lengths

    return datasets


def process_unlabeled(path, trained_model, ignore_columns=None):
    """Creates a dataset object for an unlabeled dataset.

    Args:
        path (string):
            The full path to the unlabeled data file (not just the directory).
        trained_model (:class:`~deepmatcher.MatchingModel`):
            The trained model. The model is aware of the configuration of the training
            data on which it was trained, and so this method reuses the same
            configuration for the unlabeled data.
        ignore_columns (list):
            A list of columns to ignore in the unlabeled CSV file.
    """
    with io.open(path, encoding="utf8") as f:
        header = next(unicode_csv_reader(f))

    train_info = trained_model.meta
    if ignore_columns is None:
        ignore_columns = train_info.ignore_columns
    column_naming = dict(train_info.column_naming)
    column_naming['label'] = None

    fields = _make_fields(header, column_naming['id'], column_naming['label'],
                          ignore_columns, train_info.lowercase, train_info.tokenize,
                          train_info.include_lengths)

    begin = timer()
    dataset_args = {'fields': fields, 'column_naming': column_naming}
    dataset = MatchingDataset(path=path, **dataset_args)

    # Make sure we have the same attributes.
    assert set(dataset.all_text_fields) == set(train_info.all_text_fields)

    after_load = timer()
    logger.info('Data load time: {}s'.format(after_load - begin))

    reverse_fields_dict = dict((pair[1], pair[0]) for pair in fields)
    for field, name in reverse_fields_dict.items():
        if field is not None and field.use_vocab:
            # Copy over vocab from original train data.
            field.vocab = copy.deepcopy(train_info.vocabs[name])
            # Then extend the vocab.
            field.extend_vocab(
                dataset, vectors=train_info.embeddings, cache=train_info.embeddings_cache)

    dataset.vocabs = {
        name: dataset.fields[name].vocab
        for name in train_info.all_text_fields
    }

    after_vocab = timer()
    logger.info('Vocab update time: {}s'.format(after_vocab - after_load))

    return dataset
