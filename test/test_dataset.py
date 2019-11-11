import io
import os
import shutil
import unittest
from test import test_dir_path

import pandas as pd
from nose.tools import *

import torch
from deepmatcher.data.dataset import *
from deepmatcher.data.field import FastText, MatchingField
from deepmatcher.data.process import _make_fields, process
from torchtext.utils import unicode_csv_reader
from urllib.parse import urljoin
from urllib.request import pathname2url


# import nltk
# nltk.download('perluniprops')
# nltk.download('nonbreaking_prefixes')


class ClassMatchingDatasetTestCases(unittest.TestCase):

    def test_init_1(self):
        fields = [('left_a', MatchingField()), ('right_a', MatchingField())]
        col_naming = {'id': 'id', 'label': 'label', 'left': 'left', 'right': 'right'}
        path = os.path.join(test_dir_path, 'test_datasets', 'sample_table_small.csv')
        md = MatchingDataset(fields, col_naming, path=path)
        self.assertEqual(md.id_field, 'id')
        self.assertEqual(md.label_field, 'label')
        self.assertEqual(md.all_left_fields, ['left_a'])
        self.assertEqual(md.all_right_fields, ['right_a'])
        self.assertEqual(md.all_text_fields, ['left_a', 'right_a'])
        self.assertEqual(md.canonical_text_fields, ['_a'])


class MatchingDatasetSplitsTestCases(unittest.TestCase):

    def setUp(self):
        self.data_dir = os.path.join(test_dir_path, 'test_datasets')
        self.train = 'test_train.csv'
        self.validation = 'test_valid.csv'
        self.test = 'test_test.csv'
        self.cache_name = 'test_cacheddata.pth'
        with io.open(
                os.path.expanduser(os.path.join(self.data_dir, self.train)),
                encoding="utf8") as f:
            header = next(unicode_csv_reader(f))

        id_attr = 'id'
        label_attr = 'label'
        ignore_columns = ['left_id', 'right_id']
        self.fields = _make_fields(header, id_attr, label_attr, ignore_columns, True,
                                   'nltk', False)

        self.column_naming = {
            'id': id_attr,
            'left': 'left_',
            'right': 'right_',
            'label': label_attr
        }

    def tearDown(self):
        cache_name = os.path.join(self.data_dir, self.cache_name)
        if os.path.exists(cache_name):
            os.remove(cache_name)

    def test_splits_1(self):
        datasets = MatchingDataset.splits(
            self.data_dir,
            self.train,
            self.validation,
            self.test,
            self.fields,
            None,
            None,
            self.column_naming,
            self.cache_name,
            train_pca=False)

    @raises(MatchingDataset.CacheStaleException)
    def test_splits_2(self):
        datasets = MatchingDataset.splits(
            self.data_dir,
            self.train,
            self.validation,
            self.test,
            self.fields,
            None,
            None,
            self.column_naming,
            self.cache_name,
            train_pca=False)

        datasets_2 = MatchingDataset.splits(
            self.data_dir,
            'sample_table_small.csv',
            self.validation,
            self.test,
            self.fields,
            None,
            None,
            self.column_naming,
            self.cache_name,
            True,
            False,
            train_pca=False)

    def test_splits_3(self):
        datasets = MatchingDataset.splits(
            self.data_dir,
            self.train,
            self.validation,
            self.test,
            self.fields,
            None,
            None,
            self.column_naming,
            self.cache_name,
            train_pca=False)

        datasets_2 = MatchingDataset.splits(
            self.data_dir,
            self.train,
            self.validation,
            self.test,
            self.fields,
            None,
            None,
            self.column_naming,
            self.cache_name,
            False,
            False,
            train_pca=False)


class DataframeSplitTestCases(unittest.TestCase):

    def test_split_1(self):
        labeled_path = os.path.join(test_dir_path, 'test_datasets',
                                    'sample_table_large.csv')
        labeled_table = pd.read_csv(labeled_path)
        ori_cols = list(labeled_table.columns)
        out_path = os.path.join(test_dir_path, 'test_datasets')
        train_prefix = 'train.csv'
        valid_prefix = 'valid.csv'
        test_prefix = 'test.csv'
        split(labeled_table, out_path, train_prefix, valid_prefix, test_prefix)

        train_path = os.path.join(out_path, train_prefix)
        valid_path = os.path.join(out_path, valid_prefix)
        test_path = os.path.join(out_path, test_prefix)

        train = pd.read_csv(train_path)
        valid = pd.read_csv(valid_path)
        test = pd.read_csv(test_path)

        self.assertEqual(list(train.columns), ori_cols)
        self.assertEqual(list(valid.columns), ori_cols)
        self.assertEqual(list(test.columns), ori_cols)

        if os.path.exists(train_path):
            os.remove(train_path)
        if os.path.exists(valid_path):
            os.remove(valid_path)
        if os.path.exists(test_path):
            os.remove(test_path)

    def test_split_2(self):
        labeled_path = os.path.join(test_dir_path, 'test_datasets',
                                    'sample_table_large.csv')
        labeled_table = pd.read_csv(labeled_path)
        ori_cols = list(labeled_table.columns)
        out_path = os.path.join(test_dir_path, 'test_datasets')
        train_prefix = 'train.csv'
        valid_prefix = 'valid.csv'
        test_prefix = 'test.csv'
        split(labeled_path, out_path, train_prefix, valid_prefix, test_prefix)

        train_path = os.path.join(out_path, train_prefix)
        valid_path = os.path.join(out_path, valid_prefix)
        test_path = os.path.join(out_path, test_prefix)

        train = pd.read_csv(train_path)
        valid = pd.read_csv(valid_path)
        test = pd.read_csv(test_path)

        self.assertEqual(list(train.columns), ori_cols)
        self.assertEqual(list(valid.columns), ori_cols)
        self.assertEqual(list(test.columns), ori_cols)

        if os.path.exists(train_path):
            os.remove(train_path)
        if os.path.exists(valid_path):
            os.remove(valid_path)
        if os.path.exists(test_path):
            os.remove(test_path)


class GetRawTableTestCases(unittest.TestCase):

    def test_get_raw_table(self):
        vectors_cache_dir = '.cache'
        if os.path.exists(vectors_cache_dir):
            shutil.rmtree(vectors_cache_dir)

        data_cache_path = os.path.join(test_dir_path, 'test_datasets', 'cacheddata.pth')
        if os.path.exists(data_cache_path):
            os.remove(data_cache_path)

        vec_dir = os.path.abspath(os.path.join(test_dir_path, 'test_datasets'))
        filename = 'fasttext_sample.vec.zip'
        url_base = urljoin('file:', pathname2url(vec_dir)) + os.path.sep
        ft = FastText(filename, url_base=url_base, cache=vectors_cache_dir)

        train = process(
            path=os.path.join(test_dir_path, 'test_datasets'),
            train='sample_table_small.csv',
            id_attr='id',
            embeddings=ft,
            embeddings_cache_path='',
            pca=False)

        train_raw = train.get_raw_table()
        ori_train = pd.read_csv(
            os.path.join(test_dir_path, 'test_datasets', 'sample_table_small.csv'))
        self.assertEqual(set(train_raw.columns), set(ori_train.columns))

        if os.path.exists(data_cache_path):
            os.remove(data_cache_path)

        if os.path.exists(vectors_cache_dir):
            shutil.rmtree(vectors_cache_dir)
