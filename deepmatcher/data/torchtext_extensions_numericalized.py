import logging
import os
import pdb

import six

import torch
from torchtext import data

logger = logging.getLogger(__name__)


class MatchingField(data.Field):

    def __init__(self, tokenize='spacy', *args, **kwargs):
        self.tokenizer_arg = tokenize
        super(MatchingField, self).__init__(*args, **kwargs)

    def numericalize_args(self):
        attrs = [
            'use_vocab', 'sequential', 'init_token', 'eos_token', 'unk_token',
            'tensor_type', 'preprocessing', 'lower', 'tokenizer_arg'
        ]
        return {attr: getattr(self, attr) for attr in attrs}

    def numericalize_example(self, ex):
        """Turn an example that use this field into a Tensor.

        Arguments:
            ex (...): ...
        """
        if self.use_vocab:
            if self.sequential:
                ex = [self.vocab.stoi[x] for x in ex]
            else:
                ex = self.vocab.stoi[ex]
        else:
            if self.tensor_type not in self.tensor_types:
                raise ValueError(
                    "Specified Field tensor_type {} can not be used with "
                    "use_vocab=False because we do not know how to numericalize it. "
                    "Please raise an issue at "
                    "https://github.com/pytorch/text/issues".format(self.tensor_type))
            numericalization_func = self.tensor_types[self.tensor_type]
            # It doesn't make sense to explictly coerce to a numeric type if
            # the data is sequential, since it's unclear how to coerce padding tokens
            # to a numeric type.
            if not self.sequential:
                ex = numericalization_func(ex) if isinstance(ex, six.string_types) else ex

        ex = self.tensor_type(ex)
        print('ex_type:', ex.type())
        print('ex_size:', ex.size())
        pdb.set_trace()
        return ex


class MatchingDataset(data.TabularDataset):

    class CacheStaleException(Exception):
        pass

    def __init__(self, path, format, fields, examples=None, **kwargs):
        if examples is None:
            super(MatchingDataset, self).__init__(path, format, fields, **kwargs)
        else:
            self.fields = MatchingDataset._make_fields_dict(fields)
            self.examples = examples

    @staticmethod
    def _make_fields_dict(fields):
        if isinstance(fields, dict):
            fields, field_dict = [], fields
            for field in field_dict.values():
                if isinstance(field, list):
                    fields.extend(field)
                else:
                    fields.append(field)
        return dict(fields)

    @staticmethod
    def save_cache(datasets, fields, datafiles, cachefile):
        examples = [dataset.examples for dataset in datasets]
        datafiles_modified = [os.path.getmtime(datafiles[d]) for d in datafiles]
        vocabs = {}
        field_args = {}
        for name, field in six.iteritems(fields):
            field_args[name] = None
            if field is not None:
                vocabs[name] = field.vocab
                numericalize_args = field.numericalize_args()
                for param, arg in six.iteritems(numericalize_args):
                    assert not six.callable(arg), (
                        'Cannot perform cache data checks since argument %s of field %s '
                        'is callable. Set "check_cached_data" to False or make this '
                        'argument non-callable.' % (param, field))
                field_args[name] = numericalize_args

        data = {
            'examples': examples,
            'vocabs': vocabs,
            'datafiles_modified': datafiles_modified,
            'field_args': field_args
        }
        torch.save(data, cachefile)

    @staticmethod
    def load_cache(fields, datafiles, cachefile):
        cached_data = torch.load(cachefile)
        cache_stale_cause = None

        if datafiles != cached_data['datafiles']:
            cache_stale_cause = 'Data file list has changed.'

        datafiles_modified = [os.path.getmtime(datafiles[d]) for d in datafiles]
        if datafiles_modified != cached_data['datafiles_modified']:
            cache_stale_cause = 'One or more data files have been modified.'

        if set(fields.keys()) != set(cached_data['field_args'].keys()):
            cache_stale_cause = 'Fields have changed.'

        for name, field in six.iteritems(fields):
            if field.numericalize_args() != cached_data['field_args'][name]:
                cache_stale_cause = 'Field arguments have changed.'

        return cached_data, cache_stale_cause

    @staticmethod
    def restore_data(fields, cached_data):
        datasets = [
            MatchingDataset(examples=cached_data['examples'][d])
            for d in range(len(cached_data['datafiles']))
        ]

        for name, field in six.iteritems(fields):
            if field is not None:
                field.vocab = cached_data['vocabs'][name]

        return datasets

    @classmethod
    def splits(cls,
               path,
               train,
               validation=None,
               test=None,
               fields=None,
               cache='cacheddata.pth',
               check_cached_data=True,
               auto_rebuild_cache=False,
               **kwargs):
        """Create Dataset objects for multiple splits of a dataset.

        Arguments:
            path (str): Common prefix of the splits' file paths.
            train (str): Suffix to add to path for the train set.
            validation (str): Suffix to add to path for the validation set, or None
                for no validation set. Default is None.
            test (str): Suffix to add to path for the test set, or None for no test
                set. Default is None.
            fields (list(tuple(str, Field)) or dict[str: tuple(str, Field)]:
                If using a list, the format must be CSV or TSV, and the values of the list
                should be tuples of (name, field).
                The fields should be in the same order as the columns in the CSV or TSV
                file, while tuples of (name, None) represent columns that will be ignored.

                If using a dict, the keys should be a subset of the JSON keys or CSV/TSV
                columns, and the values should be tuples of (name, field).
                Keys not present in the input dictionary are ignored.
                This allows the user to rename columns from their JSON/CSV/TSV key names
                and also enables selecting a subset of columns to load.
            check_cached_data (bool): Verify that data files haven't changes since the
                cache was constructed and that relevant field options haven't changed.
            auto_rebuild_cache (bool): Automatically rebuild the cache if the data files
                are modified or if the field options change. Defaults to False.
        Returns:
            Tuple[Dataset]: Datasets for train, validation, and
                test splits in that order, if provided.
        """

        fields_dict = MatchingDataset._make_fields_dict(fields)
        if cache:
            datafiles = list(f for f in (train, validation, test) if f is not None)
            cachefile = os.path.join(path, cache)
            try:
                cached_data, cache_stale_cause = MatchingDataset.load_cache(
                    fields_dict, datafiles, cachefile)

                if check_cached_data and cache_stale_cause and not auto_rebuild_cache:
                    raise MatchingDataset.CacheStaleException(cache_stale_cause)

                if not check_cached_data or not cache_stale_cause:
                    return MatchingDataset.restore_data(fields_dict, cached_data)

            except IOError:
                pass

        datasets = list(
            super(MatchingDataset, cls).splits(
                path, train, validation, test, fields=fields, **kwargs))

        for name, field in fields_dict.items():
            field.build_vocab(*datasets)

        for d in range(len(datasets)):
            datasets[d].numericalize()

        if cache:
            MatchingDataset.save_cache(datasets, fields_dict, datafiles, cachefile)

    def numericalize(self):
        numericalized_exs = []

        for ex in self.examples:
            numericalized_ex = []

            for name, field in self.fields.items():
                if field is not None:
                    field_value = getattr(ex, name)
                    numericalized_ex.append((name,
                                             field.numericalize_example(field_value)))

            numericalized_exs.append(numericalized_ex)

        self.examples = numericalized_exs

class MatchingIterator(data.BucketIterator):
