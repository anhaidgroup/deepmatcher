import gzip
import logging
import os
import shutil
import tarfile
import zipfile

import nltk
import six

import fasttext
import torch
from torchtext.legacy import data, vocab
from torchtext.utils import download_from_url
from urllib.request import urlretrieve

logger = logging.getLogger(__name__)


class FastText(vocab.Vectors):

    def __init__(self,
                 suffix='wiki-news-300d-1M.vec.zip',
                 url_base='https://dl.fbaipublicfiles.com/fasttext/vectors-english/',
                 **kwargs):
        url = url_base + suffix
        base, ext = os.path.splitext(suffix)
        name = suffix if ext == '.vec' else base
        print('Fasttext url b', url_base)
        super(FastText, self).__init__(name, url=url, **kwargs)


class FastTextBinary(vocab.Vectors):

    name_base = 'wiki.{}.bin'
    _direct_en_url = 'https://drive.google.com/uc?export=download&id=1Vih8gAmgBnuYDxfblbT94P6WjB7s1ZSh'

    def __init__(self, language='en', url_base=None, cache=None):
        """
        Arguments:
           language: Language of fastText pre-trained embedding model
           cache: directory for cached model
         """
        cache = os.path.expanduser(cache)
        base = url_base or 'https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.{}.zip'
        url = base.format(language)
        backup_url = None
        self.destination = os.path.join(cache, 'wiki.{}.zip'.format(language))
        if language == 'en' and url_base is None:
            backup_url = url
            self.backup_destination = self.destination
            url = FastTextBinary._direct_en_url
            self.destination = os.path.join(cache, 'wiki.en.bin')
        name = FastTextBinary.name_base.format(language)

        self.cache(name, cache, url=url, backup_url=backup_url)

    def __getitem__(self, token):
        return torch.Tensor(self.model.get_word_vector(token))

    def cache(self, name, cache, url=None, backup_url=None):
        path = os.path.join(cache, name)
        if not os.path.isfile(path) and url:
            if not os.path.exists(self.destination):
                logger.info('Downloading vectors from {} to {}'.format(url, self.destination))
                if not os.path.exists(cache):
                    os.makedirs(cache)
                if not os.path.isfile(self.destination):
                    if 'drive.google.com' in url:
                        try:
                            download_from_url(url, self.destination)
                        except RuntimeError as err:
                            if 'quota was exceeded' in str(err):
                                logger.info('Unable to fetch cached English Word Embeddings'
                                        ' from {}\nDownloading embeddings from {} to {}'.format(
                                        url, backup_url, self.backup_destination))
                                if not os.path.exists(self.backup_destination):
                                    urlretrieve(backup_url, self.backup_destination)
                                self.destination = self.backup_destination
                    else:
                        urlretrieve(url, self.destination)
            logger.info('Extracting vectors into {}'.format(cache))
            ext = os.path.splitext(self.destination)[1][1:]
            if ext == 'zip':
                with zipfile.ZipFile(self.destination, "r") as zf:
                    zf.extractall(cache)
            elif ext == 'gz':
                try:
                    with tarfile.open(self.destination, 'r:gz') as tar:
                        tar.extractall(path=cache)
                except:
                    with gzip.open(self.destination, 'rb') as infile:
                        with open(path, 'wb') as outfile:
                            shutil.copyfileobj(infile, outfile)
        if not os.path.isfile(path):
            raise RuntimeError('no vectors found at {}'.format(path))

        self.model = fasttext.load_model(path)
        self.dim = len(self['a'])


class MatchingVocab(vocab.Vocab):

    def extend_vectors(self, tokens, vectors):
        tot_dim = sum(v.dim for v in vectors)
        prev_len = len(self.itos)

        new_tokens = []
        for token in tokens:
            if token not in self.stoi:
                self.itos.append(token)
                self.stoi[token] = len(self.itos) - 1
                new_tokens.append(token)
        self.vectors.resize_(len(self.itos), tot_dim)

        for i in range(prev_len, prev_len + len(new_tokens)):
            token = self.itos[i]
            assert token == new_tokens[i - prev_len]

            start_dim = 0
            for v in vectors:
                end_dim = start_dim + v.dim
                self.vectors[i][start_dim:end_dim] = v[token.strip()]
                start_dim = end_dim
            assert (start_dim == tot_dim)


class MatchingField(data.Field):
    vocab_cls = MatchingVocab

    _cached_vec_data = {}

    def __init__(self, tokenize='nltk', id=False, **kwargs):
        self.tokenizer_arg = tokenize
        self.is_id = id
        tokenize = MatchingField._get_tokenizer(tokenize)
        super(MatchingField, self).__init__(tokenize=tokenize, **kwargs)

    @staticmethod
    def _get_tokenizer(tokenizer):
        if tokenizer == 'nltk':
            return nltk.word_tokenize
        return tokenizer

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
                    vec_data = vocab.pretrained_aliases[vec_name](cache=cache)
                cls._cached_vec_data[vec_name] = vec_data
                vec_datas.append(vec_data)
            else:
                vec_datas.append(vec)

        return vec_datas

    def build_vocab(self, *args, vectors=None, cache=None, **kwargs):
        if cache is not None:
            cache = os.path.expanduser(cache)
        if vectors is not None:
            vectors = MatchingField._get_vector_data(vectors, cache)
        super(MatchingField, self).build_vocab(*args, vectors=vectors, **kwargs)

    def extend_vocab(self, *args, vectors=None, cache=None):
        sources = []
        for arg in args:
            if isinstance(arg, data.Dataset):
                sources += [
                    getattr(arg, name)
                    for name, field in arg.fields.items()
                    if field is self
                ]
            else:
                sources.append(arg)

        tokens = set()
        for source in sources:
            for x in source:
                if not self.sequential:
                    tokens.add(x)
                else:
                    tokens.update(x)

        if self.vocab.vectors is not None:
            vectors = MatchingField._get_vector_data(vectors, cache)
            self.vocab.extend_vectors(tokens, vectors)

    def numericalize(self, arr, *args, **kwargs):
        if not self.is_id:
            return super(MatchingField, self).numericalize(arr, *args, **kwargs)
        return arr


def reset_vector_cache():
    MatchingField._cached_vec_data = {}
