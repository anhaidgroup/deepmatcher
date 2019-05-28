import logging
import os
import tarfile
import zipfile

import nltk
import six

import fastText
import torch
from torchtext import data, vocab
from torchtext.utils import download_from_url

import os
import time
import shutil
from tqdm import tqdm
import requests

logger = logging.getLogger(__name__)


class FastText(vocab.Vectors):

    def __init__(self,
                 suffix='wiki-news-300d-1M.vec.zip',
                 url_base='https://dl.fbaipublicfiles.com/fasttext/vectors-english/',
                 **kwargs):
        url = url_base + suffix
        base, ext = os.path.splitext(suffix)
        name = suffix if ext == '.vec' else base
        super(FastText, self).__init__(name, url=url, **kwargs)


class FastTextBinary(vocab.Vectors):

    name_base = 'wiki.{}.bin'
    _direct_en_url = 'https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.zip'

    def __init__(self, language='en', url_base=None, cache=None):
        """
        Arguments:
           language: Language of fastText pre-trained embedding model
           cache: directory for cached model
         """
        cache = os.path.expanduser(cache)
        if language == 'en' and url_base is None:
            url = FastTextBinary._direct_en_url
            self.destination = os.path.join(cache, 'wiki.' + language + '.bin')
        else:
            if url_base is None:
                url_base = 'https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.{}.zip'
            url = url_base.format(language)
            self.destination = os.path.join(cache, 'wiki.' + language + '.zip')
        name = FastTextBinary.name_base.format(language)

        self.cache(name, cache, url=url)

    def __getitem__(self, token):
        return torch.Tensor(self.model.get_word_vector(token))
    
    def __download_with_resume(self, url, destination):
        # Check if the requested url is ok, i.e. 200 <= status_code < 400
        head = requests.head(url)
        if not head.ok:
            head.raise_for_status()

        # Since requests doesn't support local file reading
        # we check if protocol is file://
        if url.startswith('file://'):
            url_no_protocol = url.replace('file://', '', count=1)
            if os.path.exists(url_no_protocol):
                print('File already exists, no need to download')
                return
            else:
                raise Exception('File not found at %s' % url_no_protocol)
        
        # Don't download if the file exists
        if os.path.exists(os.path.expanduser(destination)):
            print('File already exists, no need to download')
            return

        tmp_file = destination + '.part'
        first_byte = os.path.getsize(tmp_file) if os.path.exists(tmp_file) else 0
        chunk_size = 1024 ** 2  # 1 MB
        file_mode = 'ab' if first_byte else 'wb'

        # Set headers to resume download from where we've left 
        headers = {"Range": "bytes=%s-" % first_byte}
        r = requests.get(url, headers=headers, stream=True)
        file_size = int(r.headers.get('Content-length', -1))
        if file_size >= 0:
            # Content-length set
            file_size += first_byte
            total = file_size
        else:
            # Content-length not set
            print('Cannot retrieve Content-length from server')
            total = None

        print('Download from ' + url)
        print('Starting download at %.1fMB' % (first_byte / (10 ** 6)))
        print('File size is %.1fMB' % (file_size / (10 ** 6)))

        with tqdm(initial=first_byte, total=total, unit_scale=True) as pbar:
            with open(tmp_file, file_mode) as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk: # filter out keep-alive new chunks
                        f.write(chunk)
                        pbar.update(len(chunk))

        # Rename the temp download file to the correct name if fully downloaded
        shutil.move(tmp_file, destination)
    
    def cache(self, name, cache, url=None):
        path = os.path.join(cache, name)
        if not os.path.isfile(path) and url:
            logger.info('Downloading vectors from {}'.format(url))
            if not os.path.exists(cache):
                os.makedirs(cache)
            if not os.path.isfile(self.destination):
                # self.__download_with_resume(url, self.destination)
                self.__download_with_resume(url, self.destination)
            logger.info('Extracting vectors into {}'.format(cache))
            ext = os.path.splitext(self.destination)[1][1:]
            if ext == 'zip':
                with zipfile.ZipFile(self.destination, "r") as zf:
                    zf.extractall(cache)
            elif ext == 'gz':
                with tarfile.open(self.destination, 'r:gz') as tar:
                    tar.extractall(path=cache)
        if not os.path.isfile(path):
            raise RuntimeError('no vectors found at {}'.format(path))

        self.model = fastText.load_model(path)
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
