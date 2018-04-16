from nose.tools import *

import os
import shutil
import torch
import unittest
import urllib

try:
    from urllib.parse import urljoin
    from urllib.request import pathname2url
except ImportError:
    from urlparse import urljoin
    from urllib import path2pathname2url

from deepmatcher.data.field import FastText, MatchingField

class MockFastText(FastText):
    def __init__(self, url_base, name, **kwargs):
        self.url_base = url_base
        super(MockFastText, self).__init__(name, **kwargs)

class ClassFastTextTestCases(unittest.TestCase):
    def test_init_1(self):
        vectors_cache_dir = '.cache'
        if os.path.exists(vectors_cache_dir):
            shutil.rmtree(vectors_cache_dir)

        pathdir = os.path.abspath(os.path.join('.', 'test_datasets'))
        filename = 'fasttext_sample.vec.zip'
        url_base = urljoin('file:', pathname2url(pathdir)) + os.path.sep
        mft = MockFastText(url_base, filename, cache=vectors_cache_dir)
        self.assertEqual(mft.dim, 300)
        self.assertEqual(mft.vectors.size(), torch.Size([100, 300]))


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
        pass
