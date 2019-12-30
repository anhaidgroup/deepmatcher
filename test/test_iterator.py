from nose.tools import *

import os
import torch
import shutil
import unittest
from deepmatcher.data.field import FastText
from deepmatcher.data.process import process
from deepmatcher.data.iterator import MatchingIterator

from test import test_dir_path
from urllib.parse import urljoin
from urllib.request import pathname2url

class ClassMatchingIteratorTestCases(unittest.TestCase):
    def test_splits_1(self):
        vectors_cache_dir = '.cache'
        if os.path.exists(vectors_cache_dir):
            shutil.rmtree(vectors_cache_dir)

        data_dir = os.path.join(test_dir_path, 'test_datasets')
        train_path = 'sample_table_large.csv'
        valid_path = 'sample_table_large.csv'
        test_path = 'sample_table_large.csv'
        cache_file = 'cache.pth'
        cache_path = os.path.join(data_dir, cache_file)
        if os.path.exists(cache_path):
            os.remove(cache_path)

        pathdir = os.path.abspath(os.path.join(test_dir_path, 'test_datasets'))
        filename = 'fasttext_sample.vec.zip'
        url_base = urljoin('file:', pathname2url(pathdir)) + os.path.sep
        ft = FastText(filename, url_base=url_base, cache=vectors_cache_dir)

        datasets = process(data_dir, train=train_path, validation=valid_path,
                           test=test_path, cache=cache_file, embeddings=ft,
                           id_attr='_id', left_prefix='ltable_', right_prefix='rtable_',
                           embeddings_cache_path='',pca=False)

        splits = MatchingIterator.splits(datasets, batch_size=16)
        self.assertEqual(splits[0].batch_size, 16)
        self.assertEqual(splits[1].batch_size, 16)
        self.assertEqual(splits[2].batch_size, 16)
        splits_sorted = MatchingIterator.splits(datasets,
                        batch_sizes=[16, 32, 64], sort_in_buckets=False)
        self.assertEqual(splits_sorted[0].batch_size, 16)
        self.assertEqual(splits_sorted[1].batch_size, 32)
        self.assertEqual(splits_sorted[2].batch_size, 64)

        if os.path.exists(vectors_cache_dir):
            shutil.rmtree(vectors_cache_dir)

        if os.path.exists(cache_path):
            os.remove(cache_path)

    def test_create_batches_1(self):
        vectors_cache_dir = '.cache'
        if os.path.exists(vectors_cache_dir):
            shutil.rmtree(vectors_cache_dir)

        data_dir = os.path.join(test_dir_path, 'test_datasets')
        train_path = 'sample_table_large.csv'
        valid_path = 'sample_table_large.csv'
        test_path = 'sample_table_large.csv'
        cache_file = 'cache.pth'
        cache_path = os.path.join(data_dir, cache_file)
        if os.path.exists(cache_path):
            os.remove(cache_path)

        pathdir = os.path.abspath(os.path.join(test_dir_path, 'test_datasets'))
        filename = 'fasttext_sample.vec.zip'
        url_base = urljoin('file:', pathname2url(pathdir)) + os.path.sep
        ft = FastText(filename, url_base=url_base, cache=vectors_cache_dir)

        datasets = process(data_dir, train=train_path, validation=valid_path,
                           test=test_path, cache=cache_file, embeddings=ft,
                           id_attr='_id', left_prefix='ltable_', right_prefix='rtable_',
                           embeddings_cache_path='',pca=False)

        splits = MatchingIterator.splits(datasets, batch_size=16)
        batch_splits = [split.create_batches() for split in splits]

        sorted_splits = MatchingIterator.splits(datasets,
                        batch_sizes=[16, 32, 64], sort_in_buckets=False)
        batch_sorted_splits = [sorted_split.create_batches()
                                for sorted_split in sorted_splits]
        if os.path.exists(vectors_cache_dir):
            shutil.rmtree(vectors_cache_dir)

        if os.path.exists(cache_path):
            os.remove(cache_path)
