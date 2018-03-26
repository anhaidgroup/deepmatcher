import copy
import logging
import os
import pdb
import random
import tarfile
import zipfile
from collections import namedtuple
from itertools import islice
from timeit import default_timer as timer

import six

import fastText
import torch
import torch.nn.functional as F
from six.moves.urllib.request import urlretrieve
from torchtext import data, vocab
from torchtext.utils import reporthook
from tqdm import tqdm

logger = logging.getLogger(__name__)


class FastText(vocab.Vectors):

    url_base = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/'

    def __init__(self, suffix='wiki-news-300d-1M.vec.zip', **kwargs):
        url = self.url_base + suffix
        base, ext = os.path.splitext(suffix)
        name = suffix if ext == '.vec' else base
        super(FastText, self).__init__(name, url=url, **kwargs)


class FastTextBinary(vocab.Vectors):

    url_base = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.{}.zip'
    name_base = 'wiki.{}.bin'

    def __init__(self, language='en', cache=None):
        """
        Arguments:
           language: Language of fastText pre-trained embedding model
           cache: directory for cached model
         """
        cache = os.path.expanduser(cache)
        url = FastTextBinary.url_base.format(language)
        name = FastTextBinary.name_base.format(language)

        self.cache(name, cache, url=url)

    def __getitem__(self, token):
        return torch.Tensor(self.model.get_word_vector(token))

    def cache(self, name, cache, url=None):
        path = os.path.join(cache, name)
        if not os.path.isfile(path) and url:
            logger.info('Downloading vectors from {}'.format(url))
            if not os.path.exists(cache):
                os.makedirs(cache)
            dest = os.path.join(cache, os.path.basename(url))
            if not os.path.isfile(dest):
                with tqdm(unit='B', unit_scale=True, miniters=1, desc=dest) as t:
                    urlretrieve(url, dest, reporthook=reporthook(t))
            logger.info('Extracting vectors into {}'.format(cache))
            ext = os.path.splitext(dest)[1][1:]
            if ext == 'zip':
                with zipfile.ZipFile(dest, "r") as zf:
                    zf.extractall(cache)
            elif ext == 'gz':
                with tarfile.open(dest, 'r:gz') as tar:
                    tar.extractall(path=cache)
        if not os.path.isfile(path):
            raise RuntimeError('no vectors found at {}'.format(path))

        self.model = fastText.load_model(path)
        self.dim = len(self['a'])


# class MatchingVocab(data.vocab.Vocab):
#
#     def extend_vectors(self, tokens, vectors):
#         tot_dim = sum(v.dim for v in vectors)
#         prev_len = len(self.itos)
#
#         new_tokens = []
#         for token in tokens:
#             if token not in self.stoi:
#                 self.itos.append(token)
#                 self.stoi[token] = len(self.itos) - 1
#                 new_tokens.append(token)
#         self.vectors.resize_(len(self.itos), tot_dim)
#
#         for i in range(prev_len, prev_len + len(new_tokens)):
#             token = self.itos[i]
#             assert token == new_tokens[i - prev_len]
#
#             start_dim = 0
#             for v in vectors:
#                 end_dim = start_dim + v.dim
#                 self.vectors[i][start_dim:end_dim] = v[token.strip()]
#                 start_dim = end_dim
#             assert (start_dim == tot_dim)


class MatchingField(data.Field):
    # vocab_cls = MatchingVocab
    _cached_vec_data = {}

    def __init__(self, tokenize='moses', **kwargs):
        self.tokenizer_arg = tokenize
        super(MatchingField, self).__init__(tokenize=tokenize, **kwargs)

    def preprocess_args(self):
        attrs = [
            'sequential', 'init_token', 'eos_token', 'unk_token', 'preprocessing',
            'lower', 'tokenizer_arg'
        ]
        args_dict = {attr: getattr(self, attr) for attr in attrs}
        for param, arg in list(six.iteritems(args_dict)):
            if six.callable(arg):
                del args_dict[param]
                # logger.warning(
                #     'Cannot perform cache data checks for argument %s of field '
                #     '%s since it is callable.' % (param, arg))
        return args_dict

    # def numericalize_example(self, ex):
    #     """Turn an example that use this field into a Tensor.
    #
    #     Arguments:
    #         ex (...): ...
    #     """
    #     if self.use_vocab:
    #         if self.sequential:
    #             ex = [self.vocab.stoi[x] for x in ex]
    #         else:
    #             ex = self.vocab.stoi[ex]
    #     else:
    #         if self.tensor_type not in self.tensor_types:
    #             raise ValueError(
    #                 "Specified Field tensor_type {} can not be used with "
    #                 "use_vocab=False because we do not know how to numericalize it. "
    #                 "Please raise an issue at "
    #                 "https://github.com/pytorch/text/issues".format(self.tensor_type))
    #         numericalization_func = self.tensor_types[self.tensor_type]
    #         # It doesn't make sense to explictly coerce to a numeric type if
    #         # the data is sequential, since it's unclear how to coerce padding tokens
    #         # to a numeric type.
    #         if not self.sequential:
    #             ex = numericalization_func(ex) if isinstance(ex, six.string_types) else ex
    #
    #     ex = self.tensor_type(ex)
    #     print('ex_type:', ex.type())
    #     print('ex_size:', ex.size())
    #     pdb.set_trace()
    #     return ex

    @classmethod
    def _get_vector_data(cls, vecs, cache):
        if not isinstance(vecs, list):
            vecs = [vecs]

        vec_datas = []
        for vec in vecs:
            if not isinstance(vec, vocab.Vectors):
                vec_name = vec
                vec_data = cls._cached_vec_data.get(vec_name)
                if vec_data is None:
                    parts = vec_name.split('.')
                    if parts[0] == 'fasttext':
                        if parts[2] == 'bin':
                            vec_data = FastTextBinary(language=parts[1], cache=cache)
                        elif parts[2] == 'vec' and parts[1] == 'wiki':
                            vec_data = FastText(
                                suffix='wiki-news-300d-1M.vec.zip', cache=cache)
                        elif parts[2] == 'vec' and parts[1] == 'crawl':
                            vec_data = FastText(
                                suffix='crawl-300d-2M.vec.zip', cache=cache)
                if vec_data is None:
                    vec_data = _pretrained_aliases[vec_name](cache=cache)
                cls._cached_vec_data[vec_name] = vec_data
                vec_datas.append(vec_data)
            else:
                vec_datas.append(vec)

        return vec_datas

    def build_vocab(self, *args, vectors=None, cache=None, **kwargs):
        if vectors is not None:
            vectors = MatchingField._get_vector_data(vectors, cache)
        if cache is not None:
            cache = os.path.expanduser(cache)
        super(MatchingField, self).build_vocab(*args, vectors=vectors, **kwargs)

    # def embed(self, tensor):
    #     return F.embedding(tensor, self.vocab.vectors)

    # def process(self, *args, **kwargs):
    #     processed = super(MatchingField, self).process(*args, **kwargs)
    #     tensor = processed[0] if self.include_lengths else processed
    #     if self.output_vectors:
    #         tensor = self.embed(tensor)
    #     if self.include_lengths:
    #         return tensor, processed[1:]
    #     return tensor

    # def extend_vocab(self, *args):
    #     sources = []
    #     for arg in args:
    #         if isinstance(arg, data.Dataset):
    #             sources += [
    #                 getattr(arg, name)
    #                 for name, field in arg.fields.items()
    #                 if field is self
    #             ]
    #         else:
    #             sources.append(arg)
    #
    #     tokens = set()
    #     for source in sources:
    #         for x in source:
    #             if not self.sequential:
    #                 tokens.add(x)
    #             else:
    #                 tokens.update(x)
    #     self.vocab.extend_vectors(tokens, self.vectors)


def interleave_keys(keys):
    """Interleave bits from two sort keys to form a joint sort key.

    Examples that are similar in both of the provided keys will have similar
    values for the key defined by this function. Useful for tasks with two
    text fields like machine translation or natural language inference.
    """

    def interleave(args):
        return ''.join([x for t in zip(*args) for x in t])

    return int(''.join(interleave(format(x, '016b') for x in keys)), base=2)


class MatchingDataset(data.TabularDataset):

    class CacheStaleException(Exception):
        pass

    def __init__(self,
                 path=None,
                 fields=None,
                 column_naming=None,
                 format='csv',
                 examples=None,
                 metadata=None,
                 **kwargs):
        if examples is None:
            super(MatchingDataset, self).__init__(
                path, format, fields, skip_header=True, **kwargs)
            self._compute_metadata()
        else:
            self.fields = MatchingDataset._make_fields_dict(fields)
            self.examples = examples
            self.metadata = metadata

        self.column_naming = column_naming
        self._infer_attr_mapping()

    def _infer_attr_mapping(self):
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

    def _compute_metadata(self):
        self.metadata = {}

    def corresponding_field(self, name):
        if name.startswith(self.column_naming['left']):
            canonical_name = name[len(self.column_naming['left']):]
            return self.column_naming['right'] + canonical_name

        if name.startswith(self.column_naming['right']):
            canonical_name = name[len(self.column_naming['right']):]
            return self.column_naming['left'] + canonical_name

        raise ValueError('Not a left or right field')

    def text_fields(self, cname):
        return self.column_naming['left'] + cname, self.column_naming['right'] + cname

    def sort_key(self, ex):
        return interleave_keys([len(getattr(ex, attr)) for attr in self.all_text_fields])

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
    def save_cache(datasets, fields, datafiles, cachefile, column_naming):
        examples = [dataset.examples for dataset in datasets]
        metadata = [dataset.metadata for dataset in datasets]
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
            'metadata': metadata,
            'vocabs': vocabs,
            'datafiles': datafiles,
            'datafiles_modified': datafiles_modified,
            'field_args': field_args,
            'column_naming': column_naming
        }
        torch.save(data, cachefile)

    @staticmethod
    def load_cache(fields, datafiles, cachefile, column_naming):
        cached_data = torch.load(cachefile)
        cache_stale_cause = []

        if datafiles != cached_data['datafiles']:
            cache_stale_cause.append('Data file list has changed.')

        datafiles_modified = [os.path.getmtime(datafile) for datafile in datafiles]
        if datafiles_modified != cached_data['datafiles_modified']:
            cache_stale_cause.append('One or more data files have been modified.')

        if set(fields.keys()) != set(cached_data['field_args'].keys()):
            cache_stale_cause.append('Fields have changed.')

        # for name, field_args in six.iteritems(cached_data['field_args']):
        #     if fields[name].preprocess_args() != field_args:
        #         cache_stale_cause = 'Field arguments have changed.'

        for name, field in six.iteritems(fields):
            none_mismatch = (field is None) != (cached_data['field_args'][name] is None)
            args_mismatch = False
            if field is not None and cached_data['field_args'][name] is not None:
                args_mismatch = field.preprocess_args() != cached_data['field_args'][name]
            if none_mismatch or args_mismatch:
                cache_stale_cause.append('Field arguments have changed.')

        if column_naming != cached_data['column_naming']:
            cache_stale_cause.append('Other arguments have changed.')

        return cached_data, cache_stale_cause

    @staticmethod
    def restore_data(fields, cached_data):
        datasets = []
        for d in range(len(cached_data['datafiles'])):
            dataset = MatchingDataset(
                fields=fields,
                examples=cached_data['examples'][d],
                metadata=cached_data['metadata'][d],
                column_naming=cached_data['column_naming'])
            datasets.append(dataset)

        for name, field in fields:
            if name in cached_data['vocabs']:
                field.vocab = cached_data['vocabs'][name]

        return datasets

    @classmethod
    def splits(cls,
               path,
               train,
               validation=None,
               test=None,
               unlabeled=None,
               fields=None,
               embeddings=None,
               embeddings_cache=None,
               column_naming=None,
               cache=None,
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
            unlabeled (str): Suffix to add to path for an unlabeled dataset (e.g. for
                prediction). Default is None.
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

        # names = 'Train', 'Validation', 'Test'
        # sets = (train, validation, test)
        # dataset_names = [pair[0] if pair[1] for pair in zip(names, sets)]

        datasets = None
        if cache and not unlabeled:
            datafiles = list(f for f in (train, validation, test) if f is not None)
            datafiles = [os.path.expanduser(os.path.join(path, d)) for d in datafiles]
            cachefile = os.path.expanduser(os.path.join(path, cache))
            try:
                cached_data, cache_stale_cause = MatchingDataset.load_cache(
                    fields_dict, datafiles, cachefile, column_naming)

                if check_cached_data and cache_stale_cause and not auto_rebuild_cache:
                    pdb.set_trace()
                    raise MatchingDataset.CacheStaleException(cache_stale_cause)

                if not check_cached_data or not cache_stale_cause:
                    datasets = MatchingDataset.restore_data(fields, cached_data)

            except IOError:
                pass

        if not datasets:
            begin = timer()
            if not unlabeled:
                datasets = super(MatchingDataset, cls).splits(
                    path,
                    train=train,
                    validation=validation,
                    test=test,
                    fields=fields,
                    column_naming=column_naming,
                    **kwargs)
            else:
                datasets = (MatchingDataset(
                    os.path.join(path, unlabeled),
                    fields=fields,
                    column_naming=column_naming,
                    **kwargs),)

            after_load = timer()
            print('Load time:', after_load - begin)

            fields_set = set(fields_dict.values())
            for field in fields_set:
                if field is not None and field.use_vocab:
                    field.build_vocab(
                        *datasets, vectors=embeddings, cache=embeddings_cache)
            after_vocab = timer()
            print('Vocab time:', after_vocab - after_load)

            # for d in range(len(datasets)):
            #     datasets[d].numericalize()
            # after_numericalize = timer()
            # print('Numeicalize time:', after_numericalize - after_vocab)

            if cache and not unlabeled:
                MatchingDataset.save_cache(datasets, fields_dict, datafiles, cachefile,
                                           column_naming)
                after_cache = timer()
                print('Cache time:', after_cache - after_vocab)

        # if unlabeled is not None:
        #     unlabeled_dataset = MatchingDataset(
        #         os.path.join(path, unlabeled),
        #         fields,
        #         left_prefix=left_prefix,
        #         right_prefix=right_prefix,
        #         **kwargs)
        #
        #     for name, field in fields_dict.items():
        #         field.extend_vocab(unlabeled_dataset)
        #
        #     datasets.append(unlabeled_dataset)

        if len(datasets) == 1:
            return datasets[0]
        return tuple(datasets)

    # def numericalize(self):
    #     numericalized_exs = []
    #
    #     for ex in self.examples:
    #         numericalized_ex = []
    #
    #         for name, field in self.fields.items():
    #             if field is not None:
    #                 field_value = getattr(ex, name)
    #                 numericalized_ex.append((name,
    #                                          field.numericalize_example(field_value)))
    #
    #         numericalized_exs.append(numericalized_ex)
    #
    #     self.examples = numericalized_exs


AttrTensor_ = namedtuple('AttrTensor', ['data', 'lengths', 'tok_freqs', 'pc'])


class AttrTensor(AttrTensor_):

    @staticmethod
    def __new__(cls, *args, **kwargs):
        if len(kwargs) == 0:
            return super(AttrTensor, cls).__new__(cls, *args)
        else:
            name = kwargs['name']
            attr = kwargs['attr']
            train_dataset = kwargs['train_dataset']
            if isinstance(attr, tuple):
                data = attr[0]
                lengths = attr[1]
            else:
                data = attr
                lengths = None
            tok_freqs = None
            if 'tok_freqs' in train_dataset.metadata:
                tok_freqs = train_dataset.metadata['tok_freqs'][name]
            pc = None
            if 'pc' in train_dataset.metadata:
                pc = train_dataset.metadata['pc'][name]
            return AttrTensor(data, lengths, tok_freqs, pc)

    @staticmethod
    def from_old_metadata(data, old_attrtensor):
        return AttrTensor(data, *old_attrtensor[1:])


class MatchingBatch(object):

    def __init__(self, input, train_dataset):
        copy_fields = train_dataset.all_text_fields
        for name in copy_fields:
            setattr(self, name, AttrTensor(name=name, attr=getattr(input, name),
                train_dataset=train_dataset))
        for name in [train_dataset.label_field, train_dataset.id_field]:
            setattr(self, name, getattr(input, name))


class MatchingIterator(data.BucketIterator):

    def __init__(self,
                 dataset,
                 train_dataset,
                 batch_size,
                 repeat=False,
                 sort_in_buckets=True,
                 **kwargs):
        self.train_dataset = train_dataset
        self.sort_in_buckets = sort_in_buckets
        train = dataset == train_dataset
        super(MatchingIterator, self).__init__(
            dataset, batch_size, train=train, repeat=repeat, **kwargs)

    @classmethod
    def splits(cls, datasets, batch_sizes=None, **kwargs):
        """Create Iterator objects for multiple splits of a dataset.

        Arguments:
            datasets: Tuple of Dataset objects corresponding to the splits. The
                first such object should be the train set.
            batch_sizes: Tuple of batch sizes to use for the different splits,
                or None to use the same batch_size for all splits.
            Remaining keyword arguments: Passed to the constructor of the
                iterator class being used.
        """
        if batch_sizes is None:
            batch_sizes = [kwargs.pop('batch_size')] * len(datasets)
        ret = []
        for i in range(len(datasets)):
            ret.append(
                cls(datasets[i],
                    train_dataset=datasets[0],
                    batch_size=batch_sizes[i],
                    **kwargs))
        return tuple(ret)

    def __iter__(self):
        for batch in super(MatchingIterator, self).__iter__():
            yield MatchingBatch(batch, self.train_dataset)

    def create_batches(self):
        if self.sort_in_buckets:
            return data.BucketIterator.create_batches(self)
        else:
            return data.Iterator.create_batches(self)


# class _BucketBatchSampler(object):
#
#     def __init__(self, dataset, batch_size, shuffle=True):
#         if shuffle:
#             indices = torch.randperm(len(self.dataset))
#         else:
#             indices = range(len(self.dataset))
#
#         self.indices = indices
#         self.batch_size = batch_size
#         self.sort_key = self.dataset.sort_key
#
#     def _grouper(iterable, n):
#         "Collect data into fixed-length chunks or blocks"
#         it = iter(iterable)
#         piece = list(islice(it, n))
#         while piece:
#             yield piece
#             piece = list(islice(it, n))
#
#     def __iter__(self):
#         for p in self._grouper(self.indices, 100):
#             p_batch = self._grouper(sorted(p, key=self.sort_key), self.batch_size)
#             shuffled_batches = random.shuffle(list(p_batch))
#             for b in shuffled_batches:
#                 yield b
#
#     def __len__(self):
#         return (len(self.dataset) + self.batch_size - 1) // self.batch_size

# class MatchingDataLoader(DataLoader):
#
#     def _collate_fn():
#         pass
#
#     def __init__(self,
#                  dataset,
#                  batch_size,
#                  embeddings='fasttext',
#                  num_workers=2,
#                  pin_memory=True):
#         super(MatchingDataLoader, self).__init__(
#             dataset,
#             batch_size,
#             batch_sampler=_BucketBatchSampler(dataset, batch_size),
#             num_workers=num_workers,
#             collate_fn= ...,
#             pin_memory=pin_memory)
"""Mapping from string name to factory function"""
_pretrained_aliases = {
    "charngram.100d":
    lambda **kwargs: vocab.CharNGram(**kwargs),
    "fasttext.en.300d":
    lambda **kwargs: vocab.FastText(language="en", **kwargs),
    "fasttext.simple.300d":
    lambda **kwargs: vocab.FastText(language="simple", **kwargs),
    "glove.42B.300d":
    lambda **kwargs: vocab.GloVe(name="42B", dim="300", **kwargs),
    "glove.840B.300d":
    lambda **kwargs: vocab.GloVe(name="840B", dim="300", **kwargs),
    "glove.twitter.27B.25d":
    lambda **kwargs: vocab.GloVe(name="twitter.27B", dim="25", **kwargs),
    "glove.twitter.27B.50d":
    lambda **kwargs: vocab.GloVe(name="twitter.27B", dim="50", **kwargs),
    "glove.twitter.27B.100d":
    lambda **kwargs: vocab.GloVe(name="twitter.27B", dim="100", **kwargs),
    "glove.twitter.27B.200d":
    lambda **kwargs: vocab.GloVe(name="twitter.27B", dim="200", **kwargs),
    "glove.6B.50d":
    lambda **kwargs: vocab.GloVe(name="6B", dim="50", **kwargs),
    "glove.6B.100d":
    lambda **kwargs: vocab.GloVe(name="6B", dim="100", **kwargs),
    "glove.6B.200d":
    lambda **kwargs: vocab.GloVe(name="6B", dim="200", **kwargs),
    "glove.6B.300d":
    lambda **kwargs: vocab.GloVe(name="6B", dim="300", **kwargs)
}
