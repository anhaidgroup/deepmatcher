import logging
import os

import nltk
from torchtext import data, vocab
from torchtext.vocab import FastText

logger = logging.getLogger(__name__)


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
            assert start_dim == tot_dim


class MatchingField(data.Field):
    vocab_cls = MatchingVocab

    _cached_vec_data = {}

    def __init__(self, tokenize="nltk", id=False, **kwargs):  # noqa: A002
        self.tokenizer_arg = tokenize
        self.is_id = id
        tokenize = MatchingField._get_tokenizer(tokenize)
        super(MatchingField, self).__init__(tokenize=tokenize, **kwargs)

    @staticmethod
    def _get_tokenizer(tokenizer):
        if tokenizer == "nltk":
            return nltk.word_tokenize
        return tokenizer

    def preprocess_args(self):
        attrs = [
            "sequential",
            "init_token",
            "eos_token",
            "unk_token",
            "preprocessing",
            "lower",
            "tokenizer_arg",
        ]
        args_dict = {attr: getattr(self, attr) for attr in attrs}
        for param, arg in list(args_dict.items()):
            if callable(arg):
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
                    parts = vec_name.split(".")
                    if parts[0] == "fasttext":
                        vec_data = FastText(language=parts[1])
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
