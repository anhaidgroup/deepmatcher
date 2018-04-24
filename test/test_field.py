from nose.tools import *

import os
import shutil
import torch
from torchtext.vocab import Vectors
import unittest

from test import test_dir_path

try:
    from urllib.parse import urljoin
    from urllib.request import pathname2url
except ImportError:
    from urlparse import urljoin
    from urllib import path2pathname2url

from deepmatcher.data.field import FastText, MatchingField

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


class ClassMatchingFieldTestCases(unittest.TestCase):
    def test_init_1(self):
        mf = MatchingField()
        self.assertTrue(mf.sequential)

    def test_init_2(self):
        mf = MatchingField()
        seq = 'Hello, This is a test sequence for Moses.'
        tok_seq = ['Hello', ',', 'This', 'is', 'a', 'test', 'sequence',
                   'for', 'Moses', '.']
        self.assertEqual(mf.tokenize(seq), tok_seq)

    @raises(ValueError)
    def test_init_3(self):
        mf = MatchingField(tokenize='random string')

    def test_preprocess_args_1(self):
        mf = MatchingField()
        arg_dict = mf.preprocess_args()
        res_dict = {'sequential': True, 'init_token': None,
                    'eos_token': None, 'init_token': None,
                    'lower': False, 'preprocessing': None,
                    'sequential': True, 'tokenizer_arg': 'moses',
                    'unk_token': '<unk>'}
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
