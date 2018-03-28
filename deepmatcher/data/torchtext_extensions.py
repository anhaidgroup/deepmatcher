from __future__ import division

import logging
import os
import pdb
import tarfile
import zipfile
from collections import Counter, defaultdict, namedtuple
from timeit import default_timer as timer

import six
from sklearn.decomposition import TruncatedSVD

import fastText
import torch
import torch.nn as nn
from six.moves.urllib.request import urlretrieve
from torch.autograd import Variable
from torchtext import data, vocab
from torchtext.utils import reporthook
from tqdm import tqdm

from ..models.modules import Pool, NoMeta
from ..common import AttrTensor, MatchingBatch

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


class MatchingField(data.Field):
    _cached_vec_data = {}

    def __init__(self, tokenize='moses', id=False, **kwargs):
        self.tokenizer_arg = tokenize
        self.is_id = id
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
        return args_dict

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
                    vec_data = data.vocab.pretrained_aliases[vec_name](cache=cache)
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

    def numericalize(self, arr, *args, **kwargs):
        if not self.is_id:
            return super(MatchingField, self).numericalize(arr, *args, **kwargs)
        return arr


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

    def compute_metadata(self, pca=False):
        self.metadata = {}

        train_iter = MatchingIterator(
            self, self, batch_size=1024, device=-1, sort_in_buckets=False)
        counter = defaultdict(Counter)
        for batch in train_iter:
            for name in self.all_text_fields:
                attr_input = getattr(batch, name)
                counter[name].update(attr_input.data.data.view(-1))

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

        train_iter = MatchingIterator(
            self, self, batch_size=1024, device=-1, sort_in_buckets=False)
        attr_embeddings = defaultdict(list)
        for batch in train_iter:
            for name in self.all_text_fields:
                attr_input = getattr(batch, name)
                embeddings = inv_freq_pool(embed[name](attr_input))
                attr_embeddings[name].append(embeddings.data.data)

        pc = {}
        for name in self.all_text_fields:
            concatenated = torch.cat(attr_embeddings[name])
            svd = TruncatedSVD(n_components=1, n_iter=7)
            svd.fit(concatenated.numpy())
            pc[name] = svd.components_[0]
        self.metadata['pc'] = pc

    def finalize_metadata(self):

        for name in self.all_text_fields:
            self.metadata['word_probs'][name] = defaultdict(
                    lambda: 1 / self.metadata['totals'][name],
                    self.metadata['word_probs'][name])

    def get_raw_table(self):
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

        import pandas as pd
        return pd.DataFrame(rows, columns=columns)


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
    def save_cache(datasets, fields, datafiles, cachefile, column_naming, state_args):
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
        cached_data = torch.load(cachefile)
        cache_stale_cause = []

        if datafiles != cached_data['datafiles']:
            cache_stale_cause.append('Data file list has changed.')

        datafiles_modified = [os.path.getmtime(datafile) for datafile in datafiles]
        if datafiles_modified != cached_data['datafiles_modified']:
            cache_stale_cause.append('One or more data files have been modified.')

        if set(fields.keys()) != set(cached_data['field_args'].keys()):
            cache_stale_cause.append('Fields have changed.')

        for name, field in six.iteritems(fields):
            none_mismatch = (field is None) != (cached_data['field_args'][name] is None)
            args_mismatch = False
            if field is not None and cached_data['field_args'][name] is not None:
                args_mismatch = field.preprocess_args() != cached_data['field_args'][name]
            if none_mismatch or args_mismatch:
                cache_stale_cause.append('Field arguments have changed.')

        if column_naming != cached_data['column_naming']:
            cache_stale_cause.append('Other arguments have changed.')

        cache_stale_cause.extend(MatchingDataset.state_args_compatibility(state_args, cached_data['state_args']))

        return cached_data, cache_stale_cause

    @staticmethod
    def state_args_compatibility(cur_state, old_state):
        errors = []
        if not old_state['train_pca'] and cur_state['train_pca']:
            errors.append('PCA computation necessary.')
        return errors

    @staticmethod
    def restore_data(fields, cached_data):
        datasets = []
        for d in range(len(cached_data['datafiles'])):
            metadata = None
            if d == 0:
                metadata = cached_data['train_metadata']
            dataset = MatchingDataset(
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
               train_pca=False,
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
        state_args = {'train_pca': train_pca}

        datasets = None
        if cache and not unlabeled:
            datafiles = list(f for f in (train, validation, test) if f is not None)
            datafiles = [os.path.expanduser(os.path.join(path, d)) for d in datafiles]
            cachefile = os.path.expanduser(os.path.join(path, cache))
            try:
                cached_data, cache_stale_cause = MatchingDataset.load_cache(
                    fields_dict, datafiles, cachefile, column_naming, state_args)

                if check_cached_data and cache_stale_cause and not auto_rebuild_cache:
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

            if train:
                datasets[0].compute_metadata(train_pca)
            after_metadata = timer()
            print('Metadata time:', after_metadata - after_vocab)

            if cache and not unlabeled:
                MatchingDataset.save_cache(datasets, fields_dict, datafiles, cachefile,
                                           column_naming, state_args)
                after_cache = timer()
                print('Cache time:', after_cache - after_vocab)

        if train:
            datasets[0].finalize_metadata()

        if len(datasets) == 1:
            return datasets[0]
        return tuple(datasets)


class MatchingIterator(data.BucketIterator):

    def __init__(self, dataset, train_dataset, batch_size, sort_in_buckets=True,
                 **kwargs):
        self.train_dataset = train_dataset
        self.sort_in_buckets = sort_in_buckets
        train = dataset == train_dataset
        super(MatchingIterator, self).__init__(
            dataset, batch_size, train=train, repeat=False, **kwargs)

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
