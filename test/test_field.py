import os
import shutil
import unittest
from collections import Counter
from test import test_dir_path

from nose.tools import *

import torch
from deepmatcher.data.dataset import MatchingDataset
from deepmatcher.data.field import (FastText, FastTextBinary, MatchingField,
                                    MatchingVocab, reset_vector_cache)
from torchtext.vocab import Vectors
from urllib.parse import urljoin
from urllib.request import pathname2url

# import nltk
# nltk.download('perluniprops')
# nltk.download('nonbreaking_prefixes')


class ClassFastTextTestCases(unittest.TestCase):

    def test_init_1(self):
        vectors_cache_dir = '.cache'
        if os.path.exists(vectors_cache_dir):
            shutil.rmtree(vectors_cache_dir)

        pathdir = os.path.abspath(os.path.join(test_dir_path, 'test_datasets'))
        filename = 'fasttext_sample.vec.zip'
        url_base = urljoin('file:', pathname2url(pathdir)) + os.path.sep
        mft = FastText(filename, url_base=url_base, cache=vectors_cache_dir)
        self.assertEqual(mft.dim, 300)
        self.assertEqual(mft.vectors.size(), torch.Size([100, 300]))

        if os.path.exists(vectors_cache_dir):
            shutil.rmtree(vectors_cache_dir)


class ClassFastTextBinaryTestCases(unittest.TestCase):

    @raises(RuntimeError)
    def test_init_1(self):
        vectors_cache_dir = '.cache'
        if os.path.exists(vectors_cache_dir):
            shutil.rmtree(vectors_cache_dir)

        pathdir = os.path.abspath(os.path.join(test_dir_path, 'test_datasets'))
        filename = 'fasttext_sample.vec.zip'
        url_base = urljoin('file:', pathname2url(os.path.join(pathdir, filename)))
        mftb = FastTextBinary(filename, url_base=url_base, cache=vectors_cache_dir)

        if os.path.exists(vectors_cache_dir):
            shutil.rmtree(vectors_cache_dir)

    @raises(OSError)
    def test_init_2(self):
        vectors_cache_dir = '.cache'
        if os.path.exists(vectors_cache_dir):
            shutil.rmtree(vectors_cache_dir)

        pathdir = os.path.abspath(os.path.join(test_dir_path, 'test_datasets'))
        filename = 'fasttext_sample_not_exist.vec.zip'
        url_base = urljoin('file:', pathname2url(os.path.join(pathdir, filename)))
        mftb = FastTextBinary(filename, url_base=url_base, cache=vectors_cache_dir)

        if os.path.exists(vectors_cache_dir):
            shutil.rmtree(vectors_cache_dir)

    @raises(OSError)
    def test_init_3(self):
        vectors_cache_dir = '.cache'
        if os.path.exists(vectors_cache_dir):
            shutil.rmtree(vectors_cache_dir)

        pathdir = os.path.abspath(os.path.join(test_dir_path, 'test_datasets'))
        filename = 'fasttext_sample_not_exist.gz'
        url_base = urljoin('file:', pathname2url(os.path.join(pathdir, filename)))
        mftb = FastTextBinary(filename, url_base=url_base, cache=vectors_cache_dir)

        if os.path.exists(vectors_cache_dir):
            shutil.rmtree(vectors_cache_dir)


class ClassMatchingFieldTestCases(unittest.TestCase):

    def test_init_1(self):
        mf = MatchingField()
        self.assertTrue(mf.sequential)

    def test_init_2(self):
        mf = MatchingField()
        seq = 'Hello, This is a test sequence for tokenizer.'
        tok_seq = [
            'Hello', ',', 'This', 'is', 'a', 'test', 'sequence', 'for', 'tokenizer', '.'
        ]
        self.assertEqual(mf.tokenize(seq), tok_seq)

    @raises(ValueError)
    def test_init_3(self):
        mf = MatchingField(tokenize='random string')

    def test_preprocess_args_1(self):
        mf = MatchingField()
        arg_dict = mf.preprocess_args()
        res_dict = {
            'sequential': True,
            'init_token': None,
            'eos_token': None,
            'init_token': None,
            'lower': False,
            'preprocessing': None,
            'sequential': True,
            'tokenizer_arg': 'nltk',
            'unk_token': '<unk>'
        }
        self.assertEqual(arg_dict, res_dict)

    def test_build_vocab_1(self):
        mf = MatchingField()
        mf.build_vocab()

    @raises(KeyError)
    def test_build_vocab_2(self):
        mf = MatchingField()
        vector_file_name = 'fasttext.wiki_test.vec'
        cache_dir = os.path.join(test_dir_path, 'test_datasets')
        vec_data = mf.build_vocab(vectors=vector_file_name, cache=cache_dir)

    @raises(KeyError)
    def test_build_vocab_3(self):
        mf = MatchingField()
        vector_file_name = 'fasttext.crawl_test.vec'
        cache_dir = os.path.join(test_dir_path, 'test_datasets')
        vec_data = mf.build_vocab(vectors=vector_file_name, cache=cache_dir)
        self.assertIsNone(vec_data)

    def test_get_vector_data(self):
        vectors_cache_dir = '.cache'
        if os.path.exists(vectors_cache_dir):
            shutil.rmtree(vectors_cache_dir)

        pathdir = os.path.abspath(os.path.join(test_dir_path, 'test_datasets'))
        filename = 'fasttext_sample.vec'
        file = os.path.join(pathdir, filename)
        url_base = urljoin('file:', pathname2url(file))
        vecs = Vectors(name=filename, cache=vectors_cache_dir, url=url_base)
        self.assertIsInstance(vecs, Vectors)

        vec_data = MatchingField._get_vector_data(vecs, vectors_cache_dir)
        self.assertEqual(len(vec_data), 1)
        self.assertEqual(vec_data[0].vectors.size(), torch.Size([100, 300]))
        self.assertEqual(vec_data[0].dim, 300)

        if os.path.exists(vectors_cache_dir):
            shutil.rmtree(vectors_cache_dir)

    def test_numericalize_1(self):
        mf = MatchingField(id=True)
        arr = [[1], [2], [3]]
        mf.numericalize(arr)
        self.assertEqual(arr, [[1], [2], [3]])

    @raises(AttributeError)
    def test_numericalize_2(self):
        mf = MatchingField()
        arr = [['a'], ['b'], ['c']]
        mf.numericalize(arr)

    def test_extend_vocab_1(self):
        vectors_cache_dir = '.cache'
        if os.path.exists(vectors_cache_dir):
            shutil.rmtree(vectors_cache_dir)

        mf = MatchingField()
        lf = MatchingField(id=True, sequential=False)
        fields = [('id', lf), ('left_a', mf), ('right_a', mf), ('label', lf)]
        col_naming = {'id': 'id', 'label': 'label', 'left': 'left_', 'right': 'right_'}

        pathdir = os.path.abspath(os.path.join(test_dir_path, 'test_datasets'))
        filename = 'fasttext_sample.vec'
        file = os.path.join(pathdir, filename)
        url_base = urljoin('file:', pathname2url(file))
        vecs = Vectors(name=filename, cache=vectors_cache_dir, url=url_base)

        data_path = os.path.join(test_dir_path, 'test_datasets', 'sample_table_small.csv')
        md = MatchingDataset(fields, col_naming, path=data_path)

        mf.build_vocab()
        mf.vocab.vectors = torch.Tensor(len(mf.vocab.itos), 300)
        mf.extend_vocab(md, vectors=vecs)
        self.assertEqual(len(mf.vocab.itos), 6)
        self.assertEqual(mf.vocab.vectors.size(), torch.Size([6, 300]))


class TestResetVectorCache(unittest.TestCase):

    def test_reset_vector_cache_1(self):
        mf = MatchingField()
        reset_vector_cache()
        self.assertDictEqual(mf._cached_vec_data, {})


class ClassMatchingVocabTestCases(unittest.TestCase):

    def test_extend_vectors_1(self):
        vectors_cache_dir = '.cache'
        if os.path.exists(vectors_cache_dir):
            shutil.rmtree(vectors_cache_dir)

        pathdir = os.path.abspath(os.path.join(test_dir_path, 'test_datasets'))
        filename = 'fasttext_sample.vec'
        file = os.path.join(pathdir, filename)
        url_base = urljoin('file:', pathname2url(file))
        vecs = Vectors(name=filename, cache=vectors_cache_dir, url=url_base)
        self.assertIsInstance(vecs, Vectors)

        vec_data = MatchingField._get_vector_data(vecs, vectors_cache_dir)
        v = MatchingVocab(Counter())
        v.vectors = torch.Tensor(1, vec_data[0].dim)
        v.unk_init = torch.Tensor.zero_
        tokens = {'hello', 'world'}
        v.extend_vectors(tokens, vec_data)
        self.assertEqual(len(v.itos), 4)
        self.assertEqual(v.vectors.size(), torch.Size([4, 300]))
        self.assertEqual(list(v.vectors[2][0:10]), [0.0] * 10)
        self.assertEqual(list(v.vectors[3][0:10]), [0.0] * 10)

        if os.path.exists(vectors_cache_dir):
            shutil.rmtree(vectors_cache_dir)
